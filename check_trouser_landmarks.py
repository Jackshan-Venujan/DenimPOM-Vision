r"""Draws all 14 trouser landmarks the detection model predicts, labelled with their IDs.

Use it to check which ID is which point on your own images before using an ID in
src/garmentiq/instruction/trousers.json. The model always predicts all 14 points,
whatever the instruction file lists, so this shows every point it can give you.

How to use:
    python check_trouser_landmarks.py                      # uses test_images/trouser.jpeg
    python check_trouser_landmarks.py C:\Users\H.S.Thishon\OneDrive\Desktop\ELIoT_Assigment\Resources\output.jpeg  # your own images

The labelled images are saved to outputs/landmark_check/.
"""
import sys
from pathlib import Path

import cv2
import torch

import garmentiq as giq
from garmentiq.garment_classes import garment_classes
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hrnet.pth"
SAVE_DIR = ROOT / "outputs" / "landmark_check"
DEFAULT_IMAGE = ROOT / "test_images" / "trouser.jpeg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colours are BGR, because OpenCV draws in BGR.
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def detect_all_points(image_rgb, model):
    """Returns the 14 predicted (x, y) points and their confidences, in ID order."""
    coords, confidence, _ = giq.landmark.detect(
        class_name="trousers",
        class_dict=garment_classes,
        image_path=image_rgb,
        model=model,
        scale_std=200.0,
        resize_dim=[288, 384],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        device=DEVICE,
    )
    # coords is (1, 14, 2) and confidence is (1, 14, 1): row i is landmark ID i + 1.
    return coords[0], confidence[0, :, 0]


def draw_ids(image_bgr, points):
    """Draws each point with its ID next to it. Text size follows the image size."""
    scale = max(image_bgr.shape[:2]) / 800  # 1.0 for an 800 px image
    radius = max(3, int(5 * scale))
    for landmark_id, (x, y) in enumerate(points, start=1):
        x, y = int(round(x)), int(round(y))
        cv2.circle(image_bgr, (x, y), radius, GREEN, -1)
        position = (x + 2 * radius, y - radius)
        # Dark outline first so the red ID stays readable on any background
        cv2.putText(image_bgr, str(landmark_id), position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (0, 0, 0), int(5 * scale) + 1)
        cv2.putText(image_bgr, str(landmark_id), position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, RED, int(2 * scale) + 1)


def main():
    image_paths = [Path(p) for p in sys.argv[1:]] or [DEFAULT_IMAGE]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model = giq.landmark.detection.load_model(
        model_path=str(MODEL_PATH),
        model_class=PoseHighResolutionNet(),
        device=DEVICE,
    )

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print("Could not read", image_path)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # the model expects RGB
        points, confidence = detect_all_points(image_rgb, model)

        print(f"\n{image_path.name}")
        for landmark_id, ((x, y), conf) in enumerate(zip(points, confidence), start=1):
            print(f"  ID {landmark_id:>2}: x={x:7.1f}  y={y:7.1f}  confidence={conf:.2f}")

        draw_ids(image_bgr, points)
        save_path = SAVE_DIR / f"{image_path.stem}_ids.png"
        cv2.imwrite(str(save_path), image_bgr)
        print("  saved", save_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
