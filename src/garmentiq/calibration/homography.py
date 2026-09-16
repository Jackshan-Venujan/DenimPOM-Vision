# garmentiq/calibration/homography.py
"""Fitting the pixel-to-millimetre homography of one frame.

Every marker corner the detector finds is a correspondence between a pixel and a known
millimetre position on the board, so a frame showing several markers gives far more
constraints than a homography needs. The extra ones are what make it robust: RANSAC
drops a marker whose corners were misdetected, and the residuals say in millimetres
how well the rest agree.

The fit runs board -> image, because the noise lives in the detected pixel corners and
that is what should be minimised; the result is inverted for measuring. The frame's
markers are whatever the garment did not cover, so this is re-fitted per capture.
"""
import cv2
import numpy as np


class HomographyError(ValueError):
    """Raised when a frame's homography cannot be estimated reliably."""


def _check_spread(points, name):
    """Rejects points that lie on (nearly) one line, which no homography can use."""
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular_values[0] <= 0.0 or singular_values[1] / singular_values[0] < 1e-3:
        raise HomographyError(
            f"The {name} lie almost on a single line. A homography needs markers "
            f"spread in two directions — use markers from more than one edge of the "
            f"board, not a single row."
        )


def estimate_homography(corners_px, board, ransac_threshold_px=3.0, min_markers=2):
    """Fits the homography mapping frame pixels to board millimetres.

    Args:
        corners_px (dict): Marker id -> (4, 2) detected corners, undistorted if a lens
            calibration is in use.
        board (BoardSpec): The learned board layout.
        ransac_threshold_px (float): Inlier threshold, in pixels.
        min_markers (int): Fewest known markers required. Two markers give eight
            points, which is a comfortable margin over the four a homography needs.

    Returns:
        tuple:
            - H (numpy.ndarray): (3, 3) float64 pixel -> millimetre homography.
            - info (dict): `marker_ids`, `n_markers`, `n_points`, `inlier_ids`,
              `rmse_mm`, `max_mm`, `rmse_px` and per-marker `marker_errors_mm`.

    Raises:
        HomographyError: If too few known markers are visible, if they are collinear,
            or if OpenCV cannot produce a usable homography.
    """
    known = sorted(set(corners_px) & set(board.corners_mm))
    if len(known) < min_markers:
        unknown = sorted(set(corners_px) - set(board.corners_mm))
        extra = f" Detected ids not on the board: {unknown}." if unknown else ""
        raise HomographyError(
            f"Only {len(known)} board markers visible ({known}), need at least "
            f"{min_markers}. Move the garment off the markers, or add board markers "
            f"outside the garment area.{extra}"
        )

    image_points = np.vstack([np.asarray(corners_px[i], dtype=np.float64) for i in known])
    board_points = board.points_mm(known)
    _check_spread(board_points, "board markers")
    _check_spread(image_points, "detected markers")

    # With exactly four points RANSAC has nothing to vote with, so solve directly.
    method = 0 if len(board_points) == 4 else cv2.RANSAC
    board_to_image, inlier_mask = cv2.findHomography(
        board_points, image_points, method, float(ransac_threshold_px)
    )
    if board_to_image is None or not np.all(np.isfinite(board_to_image)):
        raise HomographyError(
            "OpenCV could not fit a homography to these markers. Check that the board "
            "file matches the board in the frame."
        )
    if abs(np.linalg.det(board_to_image)) < 1e-12:
        raise HomographyError("The fitted homography is degenerate.")

    inliers = (
        np.ones(len(board_points), dtype=bool)
        if inlier_mask is None
        else inlier_mask.ravel().astype(bool)
    )
    if int(inliers.sum()) < 4:
        raise HomographyError(
            f"Only {int(inliers.sum())} of {len(board_points)} corners agree with the "
            f"fitted homography. The board layout and the frame disagree."
        )

    H = np.linalg.inv(board_to_image)
    H = H / H[2, 2]

    info = homography_residuals(corners_px, board, H, marker_ids=known)
    info["inlier_ids"] = sorted(
        {known[index // 4] for index in np.flatnonzero(inliers)}
    )
    projected = cv2.perspectiveTransform(
        board_points.reshape(-1, 1, 2), board_to_image
    ).reshape(-1, 2)
    info["rmse_px"] = float(
        np.sqrt(np.mean(np.linalg.norm(projected - image_points, axis=1) ** 2))
    )
    return H, info


def homography_residuals(corners_px, board, H, marker_ids=None):
    """Reports how far each detected corner lands from its known board position.

    This is the health check on every capture: if these millimetres are small, the
    frame's geometry is understood; if one marker stands out, that marker moved, was
    misdetected, or is not where the board file says it is.

    Args:
        corners_px (dict): Marker id -> (4, 2) detected corners.
        board (BoardSpec): The learned board layout.
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography.
        marker_ids (iterable, optional): Restrict to these ids.

    Returns:
        dict: `marker_ids`, `n_markers`, `n_points`, `rmse_mm`, `max_mm` and
        `marker_errors_mm` (id -> that marker's own RMS error, worst first).
    """
    ids = sorted(set(corners_px) & set(board.corners_mm)) if marker_ids is None else list(marker_ids)
    image_points = np.vstack([np.asarray(corners_px[i], dtype=np.float64) for i in ids])
    board_points = board.points_mm(ids)

    mapped = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), np.asarray(H, dtype=np.float64)
    ).reshape(-1, 2)
    errors = np.linalg.norm(mapped - board_points, axis=1)

    per_marker = {
        int(marker_id): float(np.sqrt(np.mean(errors[index * 4 : index * 4 + 4] ** 2)))
        for index, marker_id in enumerate(ids)
    }
    return {
        "marker_ids": [int(i) for i in ids],
        "n_markers": len(ids),
        "n_points": len(errors),
        "rmse_mm": float(np.sqrt(np.mean(errors**2))),
        "max_mm": float(errors.max()),
        "marker_errors_mm": dict(
            sorted(per_marker.items(), key=lambda item: -item[1])
        ),
    }
