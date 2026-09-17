# garmentiq/calibration/capture_lens.py
"""Step A1: saving chessboard photos for the lens calibration.

Hold the board (or move the phone) so the board is seen at many angles and in every
part of the frame, including the corners and edges, where lens distortion is
strongest. Tilted views are what let the focal length be measured, so 20 flat
overhead shots calibrate worse than 15 varied ones.

The dots on the preview show where corners have been saved so far. Aim to cover the
whole frame with them.

Run::

    python -m garmentiq.calibration.capture_lens

Keys: s = save (only when the board is found)   q = quit
"""
import cv2
import numpy as np

from garmentiq.calibration.camera import (
    GREEN,
    RED,
    YELLOW,
    Camera,
    check_frame_size,
    open_window,
    put_text,
)
from garmentiq.calibration.chessboard import draw_corners, find_corners
from garmentiq.calibration.config import LENS_IMAGE_DIR, LENS_MIN_IMAGES


def next_image_path(image_dir):
    """The first unused name img_000.png, img_001.png, ... in `image_dir`."""
    index = 0
    while (image_dir / f"img_{index:03d}.png").exists():
        index += 1
    return image_dir / f"img_{index:03d}.png"


def main():
    LENS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = len(list(LENS_IMAGE_DIR.glob("img_*.png")))
    saved_corners = []  # corners of photos saved in this session, for the coverage dots

    window = "Lens capture - s = save, q = quit"
    open_window(window)
    print(f"Saving to {LENS_IMAGE_DIR} ({saved_count} photos already there)")

    with Camera() as camera:
        check_frame_size(camera.read())

        while True:
            frame = camera.read()
            corners = find_corners(frame)

            preview = frame.copy()
            for point in np.concatenate(saved_corners) if saved_corners else []:
                cv2.circle(preview, (int(point[0]), int(point[1])), 2, YELLOW, -1)
            if corners is None:
                put_text(preview, "Board NOT found", 0, RED)
            else:
                draw_corners(preview, corners)
                put_text(preview, "Board found - press s to save")
            put_text(preview, f"Saved: {saved_count} (aim for {LENS_MIN_IMAGES}+)", 1)
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if corners is None:
                    print("Not saved: the board is not found in this frame.")
                    continue
                path = next_image_path(LENS_IMAGE_DIR)
                cv2.imwrite(str(path), frame)  # the clean frame, not the preview
                saved_corners.append(corners)
                saved_count += 1
                print(f"Saved {path.name}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
