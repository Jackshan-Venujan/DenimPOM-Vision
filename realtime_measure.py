"""Real-time trouser measurement from a phone camera stream.

The live view classifies every frame, so you can see what the camera thinks it is
looking at. Pressing 's' freezes that frame and runs segmentation only, then puts
the garment on a bright green background and shows it in a review window. From
there you confirm, and the slow stages run on the green image: landmark detection
-> refinement -> derivation -> pixel measurements. Nothing is saved until you accept.

How to use:
    1. Start IP Webcam (or DroidCam) on the phone, on the same Wi-Fi as this PC.
    2. Set CAMERA_URL below to the address the app shows.
    3. Run:  python realtime_measure.py

Keys (click the video window first):
    live view       s = capture and segment          q = quit
    review window   1 = captured image   2 = mask   3 = green background
                    ENTER = measure   r = retake   ESC = cancel
"""
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.segmentation.model_definition.birefnet import BiRefNet, load_birefnet_config
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.garment_classes import garment_classes
from garmentiq.landmark.derivation.derivation_dict import derivation_dict


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
# IP Webcam: http://<phone-ip>:8080/video    DroidCam: http://<phone-ip>:4747/video
CAMERA_URL = "http://192.168.8.170:4747/video"

# Frames wider than this are scaled down before processing, to keep it fast.
FRAME_WIDTH = 1280

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
# Step 1: Camera reader
# ----------------------------------------------------------------------------
class LatestFrameReader:
    """Reads the phone stream in a background thread and keeps only the newest frame.

    A network stream queues up frames. The pipeline is slower than the camera, so
    reading frames in order would make the video fall further and further behind.
    Keeping only the newest frame keeps the display live.
    """

    def __init__(self, url):
        self.capture = cv2.VideoCapture(url)
        if not self.capture.isOpened():
            # DroidCam allows only one client at a time, so a browser tab or VLC
            # still showing the stream will lock this script out.
            raise RuntimeError(
                f"Could not open camera stream: {url}\n"
                f"Check that the phone app is streaming, and that no browser tab, "
                f"VLC or DroidCam client is already connected to it."
            )

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._keep_reading, daemon=True)
        self.thread.start()

    def _keep_reading(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)  # stream hiccup: wait briefly instead of spinning the CPU

    def read(self):
        """Returns a copy of the newest frame (BGR), or None if nothing has arrived yet."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


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
# Step 3: Frame helpers
# ----------------------------------------------------------------------------
def resize_frame(frame_bgr):
    """Scales the frame down to FRAME_WIDTH, keeping its aspect ratio."""
    height, width = frame_bgr.shape[:2]
    if width <= FRAME_WIDTH:
        return frame_bgr
    new_height = int(height * FRAME_WIDTH / width)
    return cv2.resize(frame_bgr, (FRAME_WIDTH, new_height))


def wait_for_first_frame(reader, timeout_seconds=10):
    """Waits until the stream delivers its first frame."""
    start = time.time()
    frame = reader.read()
    while frame is None:
        if time.time() - start > timeout_seconds:
            raise RuntimeError("No frames received. Check CAMERA_URL and the phone app.")
        time.sleep(0.05)
        frame = reader.read()
    return frame


# ----------------------------------------------------------------------------
# Step 4: Classification (runs on every live frame)
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
# Step 5: Segmentation and the green background (key 's')
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
# Step 6: Measure the captured frame
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


# ----------------------------------------------------------------------------
# Step 7: Drawing
# ----------------------------------------------------------------------------
def put_text(image, text, position, color):
    """Draws text with a dark outline so it stays readable on any background."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_measurements(frame_bgr, detection, label, pixel_measurements):
    """Draws measurement lines, landmark points and a list of values on the frame."""
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
    for row, (name, distance) in enumerate(pixel_measurements.items()):
        put_text(frame_bgr, f"{name}: {distance:.1f} px", (10, 90 + row * 30), GREEN)


# ----------------------------------------------------------------------------
# Step 8: The review window
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
    titles = [
        f"Captured image   ({label})",
        "Segmentation mask",
        "Green background" + ("   MEASURED" if measured else ""),
    ]
    hint = (
        "1/2/3 = view    ENTER = close    r = retake    ESC = quit review"
        if measured
        else "1/2/3 = view    ENTER = measure    r = retake    ESC = cancel"
    )

    view_index = VIEW_GREEN  # the green version is what the next stage runs on
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
# Step 9: Save a snapshot (after a successful measurement)
# ----------------------------------------------------------------------------
def save_snapshot(frame_bgr, green_bgr, mask, detection, label):
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
    giq.utils.export_dict_to_json(data=clean, filename=str(json_path))
    print("Saved", image_path.name, ",", green_path.name, ",", mask_path.name,
          "and", json_path.name)


# ----------------------------------------------------------------------------
# Step 10: Capture, review and measure (key 's')
# ----------------------------------------------------------------------------
def capture_and_measure(frame_bgr, label, segmenter, landmark_model):
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

    # The landmarks are in the frame's own pixel space, so the same overlay fits
    # the captured image and the green version.
    measured_original = frame_bgr.copy()
    measured_green = green_bgr.copy()
    draw_measurements(measured_original, detection, label, pixel_measurements)
    draw_measurements(measured_green, detection, label, pixel_measurements)

    for name, distance in pixel_measurements.items():
        print(f"  {name}: {distance:.1f} px")
    save_snapshot(measured_original, measured_green, mask, detection, label)

    action = review_capture([measured_original, mask_bgr, measured_green], label, measured=True)
    cv2.destroyWindow(REVIEW_WINDOW_NAME)
    return action == "retake"


# ----------------------------------------------------------------------------
# Step 11: Live loop
# ----------------------------------------------------------------------------
def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    classifier, segmenter, landmark_model = load_models()

    print("Connecting to", CAMERA_URL, "...")
    reader = LatestFrameReader(CAMERA_URL)

    try:
        wait_for_first_frame(reader)
        print("Running. Press 's' to capture the current frame, 'q' to quit.")

        while True:
            start_time = time.perf_counter()

            frame_bgr = resize_frame(reader.read())
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
                while capture_and_measure(frame_bgr, label, segmenter, landmark_model):
                    frame_bgr = resize_frame(reader.read())
                    label = classify_garment(
                        classifier, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
