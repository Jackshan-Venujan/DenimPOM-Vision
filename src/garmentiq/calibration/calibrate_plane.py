# garmentiq/calibration/calibrate_plane.py
"""Step B: the table homography, which turns pixels into millimetres.

Lay the chessboard flat on the table where the garment will be measured, with the
camera in its final position. Every inner corner is then a point whose position is
known both in pixels (detected) and in mm (from the square size). `cv2.findHomography`
fits one 3x3 matrix H to all of them, and H then maps any pixel on the table to mm.

Frames are undistorted first, so H belongs to the undistorted image: measure only
points taken from undistorted frames (or pass raw points to
`PlaneMeasurer.raw_to_mm`).

Run::

    python -m garmentiq.calibration.calibrate_plane

Keys: c = average PLANE_AVERAGE_FRAMES frames, fit H and save   q = quit

Run it again whenever the camera, its zoom or the table moves.
"""
import cv2
import numpy as np

from garmentiq.calibration.camera import (
    RED,
    YELLOW,
    Camera,
    open_window,
    put_text,
)
from garmentiq.calibration.chessboard import board_points_mm, draw_corners, find_corners
from garmentiq.calibration.config import (
    HOMOGRAPHY_PATH,
    INNER_CORNERS,
    PLANE_AVERAGE_FRAMES,
    PLANE_MAX_ERROR_MM,
    SQUARE_MM,
)
from garmentiq.calibration.undistort import Undistorter


def average_corners(corner_frames):
    """Averages the same corners over several frames to smooth out stream noise.

    Returns:
        tuple: ((N, 2) mean corners, jitter_px). Jitter is the largest standard
        deviation of any corner; above ~0.5 px the board or camera probably moved.
    """
    stack = np.stack(corner_frames)  # (frames, N, 2)
    jitter_px = float(np.linalg.norm(stack.std(axis=0), axis=1).max())
    return stack.mean(axis=0).astype(np.float32), jitter_px


def compute_homography(corners_px, points_mm):
    """Fits H so that H * pixel = mm, using every corner (plain least squares).

    Returns:
        tuple: (H, errors_mm). errors_mm is, for each corner, the distance in mm
        between where H puts it and where it really is on the board.
    """
    H, _ = cv2.findHomography(corners_px, points_mm, 0)
    if H is None:
        raise ValueError("findHomography failed: the corners may be degenerate.")
    mapped = cv2.perspectiveTransform(corners_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors_mm = np.linalg.norm(mapped - points_mm, axis=1)
    return H, errors_mm


def save_homography(H, image_size, errors_mm, path=HOMOGRAPHY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        H=H,
        image_size=np.array(image_size),
        rms_error_mm=float(np.sqrt(np.mean(errors_mm**2))),
        max_error_mm=float(errors_mm.max()),
        square_mm=SQUARE_MM,
        inner_corners=np.array(INNER_CORNERS),
    )


def print_report(corners_px, errors_mm, jitter_px):
    columns = INNER_CORNERS[0]
    width_mm = (columns - 1) * SQUARE_MM
    width_px = np.linalg.norm(corners_px[columns - 1] - corners_px[0])
    rms = float(np.sqrt(np.mean(errors_mm**2)))
    verdict = "PASS" if errors_mm.max() < PLANE_MAX_ERROR_MM else "FAIL"

    print("\n=== Table homography ===")
    print(f"Frames averaged : {PLANE_AVERAGE_FRAMES} (jitter {jitter_px:.2f} px)")
    print(f"Scale on board  : {width_mm / width_px:.3f} mm per pixel")
    print(f"Fit error       : RMS {rms:.3f} mm, max {errors_mm.max():.3f} mm")
    print(f"                  (limit {PLANE_MAX_ERROR_MM} mm) -> {verdict}")
    if jitter_px > 0.5:
        print("WARNING: corners moved between frames. Keep camera and board still.")


def main():
    undistorter = Undistorter()
    points_mm = board_points_mm()
    window = "Table homography - c = calibrate, q = quit"
    open_window(window)

    collected = None  # a list while averaging frames, None otherwise
    status = "Lay the board flat on the table, then press c"

    with Camera() as camera:
        while True:
            frame = undistorter.apply(camera.read())
            corners = find_corners(frame)

            if collected is not None:
                if corners is None:
                    collected = None
                    status = "Board lost while averaging - press c again"
                    print(status)
                else:
                    collected.append(corners)
                    if len(collected) == PLANE_AVERAGE_FRAMES:
                        corners_px, jitter_px = average_corners(collected)
                        H, errors_mm = compute_homography(corners_px, points_mm)
                        print_report(corners_px, errors_mm, jitter_px)
                        save_homography(H, undistorter.image_size, errors_mm)

                        snapshot = frame.copy()
                        draw_corners(snapshot, corners_px)
                        snapshot_path = HOMOGRAPHY_PATH.with_suffix(".png")
                        cv2.imwrite(str(snapshot_path), snapshot)
                        print(f"Saved {HOMOGRAPHY_PATH} and {snapshot_path.name}")

                        status = f"Saved. Max error {errors_mm.max():.2f} mm"
                        collected = None

            preview = frame.copy()
            if corners is None:
                put_text(preview, "Board NOT found", 0, RED)
            else:
                draw_corners(preview, corners)
                put_text(preview, "Board found")
            if collected is not None:
                status = f"Averaging {len(collected)}/{PLANE_AVERAGE_FRAMES}"
            put_text(preview, status, 1, YELLOW)
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and collected is None:
                if corners is None:
                    print("Board not found - cannot calibrate.")
                else:
                    collected = []

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
