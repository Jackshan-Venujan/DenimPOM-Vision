# garmentiq/calibration/metric.py
"""Converting pixel points to millimetres through a planar homography.

The homography handles perspective exactly, so there is no single
pixels-per-millimetre ratio anywhere in this package: the scale is different at every
point in the frame, and `local_mm_per_px` is how you read it at the point you care
about.

Nothing here calibrates anything; the homography comes from `homography.py`.
"""
import math

import cv2
import numpy as np


def as_points(points):
    """Normalises any reasonable point input to an (N, 2) float64 array.

    Accepts (2,), (N, 2) and OpenCV's (N, 1, 2).

    Raises:
        ValueError: If the shape is not usable, or the values are not finite.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim == 1 and array.shape[0] == 2:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Points must have shape (N, 2), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Points contain NaN or infinite values.")
    return array


def pixel_to_mm(points_px, H):
    """Maps pixel points onto the measuring plane.

    Args:
        points_px (array-like): (N, 2) points in the pixel frame `H` expects —
            undistorted pixels when the homography came from a calibrated lens.
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography.

    Returns:
        numpy.ndarray: (N, 2) float64 millimetre coordinates.
    """
    points = as_points(points_px)
    mapped = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), np.asarray(H, dtype=np.float64)
    )
    return mapped.reshape(-1, 2)


def mm_to_pixel(points_mm, H):
    """Maps millimetre points back into the pixel frame, for drawing overlays."""
    points = as_points(points_mm)
    inverse = np.linalg.inv(np.asarray(H, dtype=np.float64))
    mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), inverse)
    return mapped.reshape(-1, 2)


def distance_mm(point_a_mm, point_b_mm):
    """Straight-line distance between two millimetre points."""
    return math.hypot(
        float(point_b_mm[0]) - float(point_a_mm[0]),
        float(point_b_mm[1]) - float(point_a_mm[1]),
    )


def homography_jacobian(point_px, H):
    """Derivative d(X_mm, Y_mm) / d(x_px, y_px) of a homography at one point.

    Returns:
        numpy.ndarray: (2, 2) float64 Jacobian.
    """
    H = np.asarray(H, dtype=np.float64)
    x, y = float(point_px[0]), float(point_px[1])
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(w) < 1e-12:
        raise ValueError(
            f"Point ({x:.1f}, {y:.1f}) maps to the horizon of this homography; it is "
            f"outside the measurable area."
        )
    X = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    Y = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    return (
        np.array(
            [
                [H[0, 0] - X * H[2, 0], H[0, 1] - X * H[2, 1]],
                [H[1, 0] - Y * H[2, 0], H[1, 1] - Y * H[2, 1]],
            ]
        )
        / w
    )


def local_mm_per_px(points_px, H):
    """Millimetres one pixel spans at each point: the square root of the area scale.

    Perspective makes this vary across the frame. A landmark sitting where this value
    is much larger than elsewhere is a landmark whose pixel noise costs more
    millimetres, which is worth knowing before trusting the measurement.

    Returns:
        numpy.ndarray: (N,) float64 millimetres per pixel.
    """
    return np.array(
        [
            math.sqrt(abs(np.linalg.det(homography_jacobian(point, H))))
            for point in as_points(points_px)
        ]
    )


def measure_distance(point_a_px, point_b_px, H):
    """Distance in millimetres between two pixel points.

    Args:
        point_a_px (array-like): (x, y) of the first point.
        point_b_px (array-like): (x, y) of the second point.
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography.

    Returns:
        float: Distance in millimetres.
    """
    a_mm, b_mm = pixel_to_mm([point_a_px, point_b_px], H)
    return distance_mm(a_mm, b_mm)


def measurement_uncertainty_mm(point_a_px, point_b_px, H, point_sigma_px=2.0, coverage_k=2.0):
    """Expanded uncertainty of a distance, from the pixel noise of its two endpoints.

    The per-coordinate landmark error is pushed through the local homography
    derivative J at each endpoint, and only the component along the measured line
    matters::

        variance  = u^T (J_a J_a^T + J_b J_b^T) u * sigma_px^2
        tolerance = k * sqrt(variance)

    This is the measurement's own noise; it does not include calibration error.

    Args:
        point_a_px (array-like): (x, y) of the first point.
        point_b_px (array-like): (x, y) of the second point.
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography.
        point_sigma_px (float): One-sigma error of each landmark coordinate, in pixels.
        coverage_k (float): Coverage factor; 2 is roughly 95 %.

    Returns:
        float: Expanded uncertainty in millimetres.
    """
    points = as_points([point_a_px, point_b_px])
    a_mm, b_mm = pixel_to_mm(points, H)
    covariance = sum(
        homography_jacobian(point, H) @ homography_jacobian(point, H).T
        for point in points
    ) * float(point_sigma_px) ** 2

    distance = distance_mm(a_mm, b_mm)
    if distance > 0.0:
        direction = (b_mm - a_mm) / distance
        variance = float(direction @ covariance @ direction)
    else:
        # A zero-length measurement has no direction; use the mean of both axes.
        variance = float(np.trace(covariance)) / 2.0
    return float(coverage_k) * math.sqrt(max(variance, 0.0))
