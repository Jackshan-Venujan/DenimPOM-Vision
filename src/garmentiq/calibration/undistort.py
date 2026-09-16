# garmentiq/calibration/undistort.py
"""Removing lens distortion while staying in the same pixel frame.

A homography assumes a perfect pinhole camera. A phone lens bends straight lines,
most strongly toward the image edges — which is exactly where a 1200 mm garment
reaches — so distortion is removed before anything is converted to millimetres.

Both helpers keep the original camera matrix, so undistorted points stay in the same
pixel coordinate system as the raw image and a homography fitted on one works on the
other. With no intrinsics they pass their input straight through, so the whole
pipeline still runs before the lens has been calibrated; it is just less accurate.
"""
import cv2
import numpy as np

from .metric import as_points


def undistort_points(points_px, intrinsics):
    """Removes lens distortion from pixel points.

    Args:
        points_px (array-like): (N, 2) raw image points.
        intrinsics (CameraIntrinsics | None): Calibrated lens, or None for a no-op.

    Returns:
        numpy.ndarray: (N, 2) float64 undistorted points, in the same pixel frame.
    """
    points = as_points(points_px)
    if intrinsics is None or not np.any(intrinsics.dist_coeffs):
        return points.copy()

    K = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist, None, None, K)
    undistorted = undistorted.reshape(-1, 2).astype(np.float64)

    # cv2.undistortPoints inverts the distortion with a fixed, small number of
    # iterations, which leaves a visible residual at the frame edges. Polish it until
    # re-distorting the answer reproduces the input to well below a pixel.
    no_rotation = np.zeros(3)
    focal = np.array([K[0, 0], K[1, 1]])
    centre = K[:2, 2]
    for _ in range(50):
        normalized = np.column_stack(
            [(undistorted - centre) / focal, np.ones(len(undistorted))]
        )
        redistorted = cv2.projectPoints(normalized, no_rotation, no_rotation, K, dist)[0]
        correction = points - redistorted.reshape(-1, 2)
        undistorted = undistorted + correction
        if np.max(np.abs(correction)) < 1e-9:
            break
    return undistorted


def undistort_marker_corners(corners_px, intrinsics):
    """Applies `undistort_points` to a {marker id: (4, 2)} dictionary."""
    if intrinsics is None:
        return {int(i): np.asarray(c, dtype=np.float64) for i, c in corners_px.items()}
    return {
        int(marker_id): undistort_points(corners, intrinsics)
        for marker_id, corners in corners_px.items()
    }


def undistort_image(image, intrinsics):
    """Removes lens distortion from a whole image, keeping the same camera matrix.

    Straightening the image is only needed for display and for detecting markers on
    an already-undistorted frame; measuring undistorts the handful of points it needs,
    which is far cheaper and avoids a resampling step.

    Args:
        image (numpy.ndarray): BGR or grayscale image.
        intrinsics (CameraIntrinsics | None): Calibrated lens, or None for a no-op.

    Returns:
        numpy.ndarray: Undistorted image of the same size.

    Raises:
        ValueError: If the image size differs from the one the lens was calibrated at,
            because intrinsics are only valid at their own resolution.
    """
    if intrinsics is None or not np.any(intrinsics.dist_coeffs):
        return image.copy()

    height, width = image.shape[:2]
    if (width, height) != tuple(intrinsics.image_size):
        raise ValueError(
            f"Image is {width}x{height} but the lens was calibrated at "
            f"{intrinsics.image_size[0]}x{intrinsics.image_size[1]}. Intrinsics are "
            f"only valid at their own resolution; recalibrate, or capture at the "
            f"calibrated size."
        )
    return cv2.undistort(image, intrinsics.camera_matrix, intrinsics.dist_coeffs)
