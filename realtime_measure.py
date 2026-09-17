"""Real-time trouser measurement from a phone camera stream, in millimetres.

The live view classifies every frame, so you can see what the camera thinks it is
looking at. Pressing 's' freezes that frame and runs segmentation only, then puts
the garment on a bright green background and shows it in a review window. From
there you confirm, and the slow stages run on the green image: landmark detection
-> refinement -> derivation -> pixel measurements -> millimetres. Nothing is saved
until you accept.

Millimetres come from the chessboard calibration in garmentiq.calibration (see
src/garmentiq/calibration/calibration.md). Every frame is undistorted with the lens
calibration, and the two end landmarks of each measurement are mapped onto the table
with the table homography. For the numbers to be right:
    - run both calibrations first, at the resolution the stream delivers;
    - keep the camera exactly where it was when calibrate_plane ran;
    - lay the garment flat on the calibrated table.
At 640x480 one pixel covers about 2.7 mm of table, which limits the accuracy.

How to use:
    1. Start DroidCam (or IP Webcam) on the phone, on the same Wi-Fi as this PC.
    2. Set CAMERA_URL in src/garmentiq/calibration/config.py. The calibration and
       this script share it, so they always use the same camera.
    3. Calibrate once (calibration.md, sections 2 and 8).
    4. Run:  python realtime_measure.py

Keys (click the video window first):
    live view       s = capture and segment          q = quit
    review window   1 = captured image   2 = mask   3 = green background
                    ENTER = measure   r = retake   ESC = cancel
"""
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import garmentiq as giq
from garmentiq.calibration.camera import Camera
from garmentiq.calibration.config import CAMERA_URL
from garmentiq.calibration.metrology import PlaneMeasurer
from garmentiq.classification.model_definition import tinyViT
from garmentiq.segmentation.model_definition.birefnet import BiRefNet, load_birefnet_config
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.garment_classes import garment_classes
from garmentiq.landmark.derivation.derivation_dict import derivation_dict


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# Frames are never resized here: the calibration maps pixels of the calibrated
# resolution to mm, so a resized frame would give wrong millimetres. The models
# resize their own inputs internally.

# Segmentation input size. It is the slowest stage.
# (512, 512): ~0.4 s per frame on an RTX 3050, landmarks can move by ~2 px.
# (1024, 1024): ~1.3 s per frame, same result as full_pipeline.ipynb.
SEGMENTATION_SIZE = (512, 512)

# The green the rest of the repo uses for background replacement (RGB).
GREEN_BACKGROUND = (102, 255, 102)

# BiRefNet returns a soft mask (0-255). Anything below this counts as background.
MASK_THRESHOLD = 128

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SAVE_DIR = ROOT / "outputs" / "realtime"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_NAME = "GarmentIQ - live view"
REVIEW_WINDOW_NAME = "GarmentIQ - captured frame"

# Colours are BGR, because OpenCV draws in BGR.
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

# Review window views, in the order the number keys select them.
VIEW_ORIGINAL, VIEW_MASK, VIEW_GREEN = 0, 1, 2


# ----------------------------------------------------------------------------
# Step 1: Camera frames
# ----------------------------------------------------------------------------
def read_undistorted(camera, measurer):
    """Returns the newest camera frame (BGR) with lens distortion removed.

    Every stage runs on this undistorted frame, so the landmark pixels it produces
    can go straight through the table homography to mm. Raises ValueError if the
    stream's resolution differs from the calibrated one.
    """
    return measurer.undistort(camera.read())


# ----------------------------------------------------------------------------
# Step 2: Load models (same settings as full_pipeline.ipynb)
# ----------------------------------------------------------------------------
def load_models():
    print("Loading models on", DEVICE, "...")

    classifier = giq.classification.load_model(
        model_path=str(MODEL_DIR / "tiny_vit_inditex_finetuned.pt"),
        model_class=tinyViT,
        model_args={"num_classes": 9, "img_size": (120, 184), "patch_size": 6},
        device=DEVICE,
    )
    segmenter = giq.segmentation.load_model(
        model_path=str(MODEL_DIR / "birefnet" / "model.safetensors"),
        model_class=BiRefNet,
        model_args=load_birefnet_config(),
        device=DEVICE,
    )
    landmark_model = giq.landmark.detection.load_model(
        model_path=str(MODEL_DIR / "hrnet.pth"),
        model_class=PoseHighResolutionNet(),
        device=DEVICE,
    )
    return classifier, segmenter, landmark_model


# ----------------------------------------------------------------------------
# Step 3: Classification (runs on every live frame)
# ----------------------------------------------------------------------------
def classify_garment(classifier, frame_rgb):
    label, _ = giq.classification.predict(
        model=classifier,
        image_path=frame_rgb,
        classes=sorted(garment_classes),
        resize_dim=(120, 184),
        normalize_mean=[0.8047, 0.7808, 0.7769],
        normalize_std=[0.2957, 0.3077, 0.3081],
        device=DEVICE,
    )
    return label


# ----------------------------------------------------------------------------
# Step 4: Segmentation and the green background (key 's')
# ----------------------------------------------------------------------------
def segment_frame(frame_rgb, segmenter):
    """Returns the garment mask for one frame: (H, W) uint8, soft values 0-255."""
    _, mask = giq.segmentation.extract(
        model=segmenter,
        image_path=frame_rgb,
        resize_dim=SEGMENTATION_SIZE,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        device=DEVICE,
    )
    return mask


def make_green_background(frame_rgb, mask):
    """Puts the segmented garment on a flat bright green background (RGB in, RGB out).

    change_background_color() replaces pixels where the mask is exactly 0, but
    BiRefNet's mask is soft, so the anti-aliased edge and any low-confidence
    background would survive. Binarising first gives a clean key.
    """
    binary_mask = np.where(mask >= MASK_THRESHOLD, 255, 0).astype(np.uint8)
    return giq.segmentation.change_background_color(
        image_np=frame_rgb,
        mask_np=binary_mask,
        background_color=GREEN_BACKGROUND,
    )


# ----------------------------------------------------------------------------
# Step 5: Measure the captured frame
# ----------------------------------------------------------------------------
def measure_frame(green_rgb, mask, label, landmark_model):
    """Runs detection -> refinement -> derivation -> distances on the green image.

    Detection runs on the green-background image, the way giq.tailor does it when
    a background colour is set: the clutter behind the garment is gone, so the
    landmarks sit steadier. Refinement and derivation keep the soft mask, exactly
    as in full_pipeline.ipynb.

    Returns the detection dictionary (landmarks + measurements) and a
    {measurement name: pixel distance} dictionary.
    """
    # Landmark detection
    coords, confidence, detection = giq.landmark.detect(
        class_name=label,
        class_dict=garment_classes,
        image_path=green_rgb,
        model=landmark_model,
        scale_std=200.0,
        resize_dim=[288, 384],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        device=DEVICE,
    )

    # Refinement: move landmarks onto the garment edge using the mask
    refined_coords, detection = giq.landmark.refine(
        class_name=label,
        detection_np=coords,
        detection_conf=confidence,
        detection_dict=detection,
        mask=mask,
    )

    # Derivation: compute the landmarks that are not predicted directly
    _, detection = giq.landmark.derive(
        class_name=label,
        detection_dict=detection,
        derivation_dict=derivation_dict,
        landmark_coords=refined_coords,
        np_mask=mask,
    )

    # Pixel distance for each measurement
    pixel_measurements, detection = giq.utils.compute_measurement_distances(detection)
    return detection, pixel_measurements


def measurements_in_mm(detection, label, measurer):
    """Converts every measurement to millimetres: {measurement name: mm}.

    Each measurement is a straight line between a start and an end landmark. Both
    ends are mapped onto the table with the homography, and the distance is taken
    there. Multiplying the pixel distance by one mm-per-pixel factor would be wrong:
    with perspective, a pixel covers a different length in different parts of the
    frame.
    """
    landmarks = detection[label]["landmarks"]
    distances_mm = {}
    for name, measurement in detection[label]["measurements"].items():
        start = landmarks[measurement["landmarks"]["start"]]
        end = landmarks[measurement["landmarks"]["end"]]
        distances_mm[name] = measurer.distance_mm(
            (start["x"], start["y"]), (end["x"], end["y"])
        )
    return distances_mm


# ----------------------------------------------------------------------------
# Step 6: Drawing
# ----------------------------------------------------------------------------
def put_text(image, text, position, color):
    """Draws text with a dark outline so it stays readable on any background."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_measurements(frame_bgr, detection, label, mm_measurements):
    """Draws measurement lines, landmark points and a list of mm values on the frame."""
    landmarks = detection[label]["landmarks"]
    measurements = detection[label]["measurements"]

    def point_of(landmark_id):
        point = landmarks[landmark_id]
        return int(round(point["x"])), int(round(point["y"]))

    # Lines between the start and end landmark of each measurement
    for measurement in measurements.values():
        start = point_of(measurement["landmarks"]["start"])
        end = point_of(measurement["landmarks"]["end"])
        cv2.line(frame_bgr, start, end, YELLOW, 2)

    # Landmark points (only those that have coordinates)
    for landmark_id, point in landmarks.items():
        if "x" in point and "y" in point:
            x, y = point_of(landmark_id)
            cv2.circle(frame_bgr, (x, y), 5, GREEN, -1)
            put_text(frame_bgr, landmark_id, (x + 8, y - 8), GREEN)

    # Measurement values, listed in the top-left corner
    put_text(frame_bgr, f"Class: {label}", (10, 60), GREEN)
    for row, (name, distance_mm) in enumerate(mm_measurements.items()):
        put_text(frame_bgr, f"{name}: {distance_mm:.1f} mm", (10, 90 + row * 30), GREEN)


# ----------------------------------------------------------------------------
# Step 7: The review window
# ----------------------------------------------------------------------------
def show_review(views, view_index, title, hint):
    """Draws one view of the captured frame, with its caption, into the review window."""
    display = views[view_index].copy()
    put_text(display, title, (10, 30), YELLOW)
    put_text(display, hint, (10, display.shape[0] - 20), YELLOW)
    cv2.imshow(REVIEW_WINDOW_NAME, display)


def review_capture(views, label, measured=False):
    """Shows the captured frame and waits for a decision. All views share one size.

    views is [original BGR, mask BGR, green BGR]. Returns "measure", "retake" or
    "cancel"; after measuring, "measure" is not offered and the window just closes.
    """
    measured_tag = "   MEASURED" if measured else ""
    titles = [
        f"Captured image   ({label}){measured_tag}",
        "Segmentation mask",
        f"Green background{measured_tag}",
    ]
    hint = (
        "1/2/3 = view    ENTER = close    r = retake    ESC = quit review"
        if measured
        else "1/2/3 = view    ENTER = measure    r = retake    ESC = cancel"
    )

    # Before measuring, open on the green image, since that is what gets measured.
    # After measuring, open on the real captured photo with the measurements drawn.
    view_index = VIEW_ORIGINAL if measured else VIEW_GREEN
    show_review(views, view_index, titles[view_index], hint)

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("1"), ord("2"), ord("3")):
            view_index = key - ord("1")
            show_review(views, view_index, titles[view_index], hint)
        elif key in (13, ord("m")):  # 13 = Enter
            return "close" if measured else "measure"
        elif key == ord("r"):
            return "retake"
        elif key in (27, ord("q")):  # 27 = Esc
            return "close" if measured else "cancel"
        elif cv2.getWindowProperty(REVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "close" if measured else "cancel"  # window closed with the X button


# ----------------------------------------------------------------------------
# Step 8: Save a snapshot (after a successful measurement)
# ----------------------------------------------------------------------------
def save_snapshot(frame_bgr, green_bgr, mask, detection, label, mm_measurements):
    name = datetime.now().strftime("frame_%Y%m%d_%H%M%S")
    image_path = SAVE_DIR / f"{name}.png"
    green_path = SAVE_DIR / f"{name}_green.png"
    mask_path = SAVE_DIR / f"{name}_mask.png"
    json_path = SAVE_DIR / f"{name}_measurement.json"

    cv2.imwrite(str(image_path), frame_bgr)
    cv2.imwrite(str(green_path), green_bgr)
    cv2.imwrite(str(mask_path), mask)
    clean = giq.utils.clean_detection_dict(
        class_name=label, image_name=image_path.name, detection_dict=detection
    )

    # clean_detection_dict keeps only the pixel "distance", so the mm value is added
    # next to it afterwards.
    measurements = clean[image_path.name]["measurements"]
    for measurement_name, distance_mm in mm_measurements.items():
        measurements[measurement_name]["distance_mm"] = round(distance_mm, 1)

    giq.utils.export_dict_to_json(data=clean, filename=str(json_path))
    print("Saved", image_path.name, ",", green_path.name, ",", mask_path.name,
          "and", json_path.name)


# ----------------------------------------------------------------------------
# Step 9: Capture, review and measure (key 's')
# ----------------------------------------------------------------------------
def capture_and_measure(frame_bgr, label, segmenter, landmark_model, measurer):
    """Segments the captured frame, shows it for review, then measures on confirmation.

    Returns True if the user asked for another capture straight away ('r').
    """
    print(f"Segmenting one frame as '{label}' ...")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # models expect RGB

    try:
        mask = segment_frame(frame_rgb, segmenter)
        green_rgb = make_green_background(frame_rgb, mask)
    except Exception as error:
        print("Could not segment this frame:", error)
        return False

    green_bgr = cv2.cvtColor(green_rgb, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    views = [frame_bgr.copy(), mask_bgr, green_bgr]

    action = review_capture(views, label)
    if action != "measure":
        cv2.destroyWindow(REVIEW_WINDOW_NAME)
        return action == "retake"

    print("Measuring ...")
    try:
        detection, pixel_measurements = measure_frame(
            green_rgb, mask, label, landmark_model
        )
    except Exception as error:
        # Usually no garment in view, so the landmarks cannot be derived.
        print("Could not measure this frame:", error)
        cv2.destroyWindow(REVIEW_WINDOW_NAME)
        return False

    # Pixels -> millimetres, using the calibration
    mm_measurements = measurements_in_mm(detection, label, measurer)

    # The landmarks are in the frame's own pixel space, so the same overlay fits
    # the captured image and the green version.
    measured_original = frame_bgr.copy()
    measured_green = green_bgr.copy()
    draw_measurements(measured_original, detection, label, mm_measurements)
    draw_measurements(measured_green, detection, label, mm_measurements)

    for name, distance_mm in mm_measurements.items():
        print(f"  {name}: {distance_mm:.1f} mm  ({pixel_measurements[name]:.1f} px)")
    save_snapshot(
        measured_original, measured_green, mask, detection, label, mm_measurements
    )

    action = review_capture([measured_original, mask_bgr, measured_green], label, measured=True)
    cv2.destroyWindow(REVIEW_WINDOW_NAME)
    return action == "retake"


# ----------------------------------------------------------------------------
# Step 10: Live loop
# ----------------------------------------------------------------------------
def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load the calibration first: if it is missing, fail now, not after the models.
    measurer = PlaneMeasurer()
    print("Loaded calibration: lens + table homography at "
          f"{measurer.image_size[0]}x{measurer.image_size[1]}")

    classifier, segmenter, landmark_model = load_models()

    print("Connecting to", CAMERA_URL, "...")
    try:
        with Camera(CAMERA_URL) as camera:
            print("Running. Press 's' to capture the current frame, 'q' to quit.")

            while True:
                start_time = time.perf_counter()

                frame_bgr = read_undistorted(camera, measurer)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # models expect RGB

                # Classification is fast, so it runs on every frame
                label = classify_garment(classifier, frame_rgb)

                # The live view shows the class only; capturing happens on 's'
                live_view = frame_bgr.copy()
                fps = 1.0 / (time.perf_counter() - start_time)
                put_text(live_view, f"FPS: {fps:.1f}", (10, 30), YELLOW)
                put_text(live_view, f"Class: {label}", (10, 60), GREEN)
                put_text(live_view, "s = capture   q = quit", (10, 90), YELLOW)
                cv2.imshow(WINDOW_NAME, live_view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    # 'r' in the review window means "capture again", so loop until
                    # the user is done with this garment.
                    while capture_and_measure(
                        frame_bgr, label, segmenter, landmark_model, measurer
                    ):
                        frame_bgr = read_undistorted(camera, measurer)
                        label = classify_garment(
                            classifier, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        )
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
