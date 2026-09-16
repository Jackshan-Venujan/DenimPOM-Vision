# garmentiq/calibration/calibrate_lens.py
"""Step A2: measuring the lens from the saved chessboard photos.

Finds the board in every photo from Step A1 and runs `cv2.calibrateCamera`, which
returns:

* K, the camera matrix: focal lengths (fx, fy) and image centre (cx, cy), in pixels;
* dist, the lens distortion coefficients (k1, k2, p1, p2, k3).

Both are saved to `camera_intrinsics.npz`. They are valid only for this phone, this
zoom and focus, and this resolution.

Run::

    python -m garmentiq.calibration.calibrate_lens

If a photo is flagged as an outlier (usually motion blur), delete it and run again.
"""
import cv2
import numpy as np

from garmentiq.calibration.chessboard import board_points_mm, find_corners
from garmentiq.calibration.config import (
    INTRINSICS_PATH,
    LENS_IMAGE_DIR,
    LENS_MAX_RMS_PX,
    LENS_MIN_IMAGES,
)


def find_corners_in_images(image_dir):
    """Detects the board in every img_*.png in `image_dir`.

    Returns:
        tuple: (corners_per_image, image_names, image_size). Images where the board is
        not found are skipped and reported.

    Raises:
        ValueError: If the images do not all have the same size.
    """
    corners_per_image, image_names, image_size = [], [], None

    for path in sorted(image_dir.glob("img_*.png")):
        image = cv2.imread(str(path))
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"{path.name} is {size}, the others are {image_size}.")

        corners = find_corners(image)
        if corners is None:
            print(f"  skipped {path.name}: board not found")
            continue
        corners_per_image.append(corners)
        image_names.append(path.name)

    return corners_per_image, image_names, image_size


def solve_intrinsics(corners_per_image, image_size):
    """Runs the lens calibration on detected corners.

    k3 is fixed at 0: with a phone lens and a few dozen views it is poorly determined
    and tends to take wild values that break undistortion near the frame edges.

    Args:
        corners_per_image (list): (N, 2) pixel corners, one array per image.
        image_size (tuple): (width, height).

    Returns:
        dict: K, dist, image_size, rms_px and per_view_rms_px (one value per image).
    """
    board = board_points_mm()
    board_3d = np.hstack([board, np.zeros((len(board), 1), np.float32)])  # z = 0
    object_points = [board_3d] * len(corners_per_image)
    image_points = [corners.reshape(-1, 1, 2) for corners in corners_per_image]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=cv2.CALIB_FIX_K3
    )

    per_view = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residuals = (projected - img).reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))))

    return {
        "K": K,
        "dist": dist.ravel(),
        "image_size": np.array(image_size),
        "rms_px": float(rms),
        "per_view_rms_px": np.array(per_view),
    }


def save_intrinsics(result, path=INTRINSICS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)


def print_report(result, image_names):
    K = result["K"]
    print("\n=== Lens calibration ===")
    print(f"Images used : {len(image_names)}")
    print(f"Image size  : {result['image_size'][0]}x{result['image_size'][1]}")
    print(f"fx, fy      : {K[0, 0]:.1f}, {K[1, 1]:.1f} px")
    print(f"cx, cy      : {K[0, 2]:.1f}, {K[1, 2]:.1f} px")
    print(f"dist        : {np.array2string(result['dist'], precision=5)}")

    per_view = result["per_view_rms_px"]
    outlier_limit = max(2 * np.median(per_view), LENS_MAX_RMS_PX)
    print("\nPer-image error (px):")
    for name, error in zip(image_names, per_view):
        flag = "  <- outlier, consider deleting" if error > outlier_limit else ""
        print(f"  {name}: {error:.3f}{flag}")

    rms = result["rms_px"]
    verdict = "PASS" if rms < LENS_MAX_RMS_PX else "FAIL"
    print(f"\nOverall RMS : {rms:.3f} px (limit {LENS_MAX_RMS_PX}) -> {verdict}")
    if len(image_names) < LENS_MIN_IMAGES:
        print(f"WARNING: {len(image_names)} usable images, aim for {LENS_MIN_IMAGES}+")


def main():
    print(f"Reading photos from {LENS_IMAGE_DIR}")
    corners_per_image, image_names, image_size = find_corners_in_images(LENS_IMAGE_DIR)
    if len(corners_per_image) < 3:
        raise SystemExit(
            f"Only {len(corners_per_image)} usable photos. Run capture_lens first."
        )

    result = solve_intrinsics(corners_per_image, image_size)
    print_report(result, image_names)
    save_intrinsics(result)
    print(f"\nSaved {INTRINSICS_PATH}")


if __name__ == "__main__":
    main()
