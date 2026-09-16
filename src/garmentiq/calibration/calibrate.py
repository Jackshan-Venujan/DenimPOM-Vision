# garmentiq/calibration/calibrate.py
"""Turning one frame into a pixel-to-millimetre transform.

This is the single call the rest of the application makes. It runs the whole metrology
chain for one capture — detect the markers the garment left visible, fit the
homography to the learned board, lift it to the top of the fabric — and hands back an
object that converts points and measures distances.

The homography is re-fitted per frame, so the camera and the board may move between
captures; only the lens calibration and the board layout carry over.
"""
from dataclasses import dataclass, field

import numpy as np

from .detect import detect_markers, load_image
from .homography import estimate_homography
from .metric import (
    as_points,
    distance_mm,
    local_mm_per_px,
    measurement_uncertainty_mm,
    mm_to_pixel,
    pixel_to_mm,
)
from .surface import lift_to_surface
from .undistort import undistort_marker_corners, undistort_points


@dataclass
class FrameCalibration:
    """The pixel-to-millimetre transform of one captured frame.

    Attributes:
        H (numpy.ndarray): (3, 3) homography from undistorted pixels to millimetres,
            already lifted to the measuring surface.
        board (BoardSpec): The board layout it was fitted to.
        intrinsics (CameraIntrinsics | None): Lens used to undistort points.
        surface_offset_mm (float): Fabric thickness the plane was lifted by.
        corners_px (dict): Marker id -> detected corners, as found in the raw frame.
        info (dict): Fit quality — `n_markers`, `marker_ids`, `rmse_mm`, `max_mm`,
            `rmse_px`, `marker_errors_mm`, `inlier_ids`.
    """

    H: np.ndarray
    board: object
    intrinsics: object = None
    surface_offset_mm: float = 0.0
    corners_px: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def n_markers(self):
        """How many board markers this frame's fit rests on."""
        return int(self.info.get("n_markers", len(self.corners_px)))

    @property
    def rmse_mm(self):
        """How far the board markers land from where the board file says they are."""
        return float(self.info.get("rmse_mm", float("nan")))

    def to_mm(self, points_px):
        """Converts raw frame pixels to millimetres, undistorting on the way.

        Args:
            points_px (array-like): (N, 2) points in the raw frame, e.g. landmark
                coordinates straight from the pipeline.

        Returns:
            numpy.ndarray: (N, 2) float64 millimetre coordinates.
        """
        return pixel_to_mm(undistort_points(points_px, self.intrinsics), self.H)

    def to_pixels(self, points_mm):
        """Converts millimetre points back to undistorted pixels, for drawing."""
        return mm_to_pixel(points_mm, self.H)

    def distance(self, point_a_px, point_b_px):
        """Straight-line distance in millimetres between two raw frame points."""
        a_mm, b_mm = self.to_mm([point_a_px, point_b_px])
        return distance_mm(a_mm, b_mm)

    def uncertainty(self, point_a_px, point_b_px, point_sigma_px=2.0, coverage_k=2.0):
        """Expanded uncertainty of `distance`, from landmark pixel noise alone.

        It does not include the calibration's own error; compare `rmse_mm` for that.
        """
        points = undistort_points([point_a_px, point_b_px], self.intrinsics)
        return measurement_uncertainty_mm(
            points[0], points[1], self.H, point_sigma_px, coverage_k
        )

    def mm_per_px(self, points_px):
        """Millimetres one pixel spans at each raw frame point."""
        return local_mm_per_px(undistort_points(points_px, self.intrinsics), self.H)

    def landmarks_to_mm(self, landmarks):
        """Converts a `{id: {"x": px, "y": px}}` landmark dictionary to millimetres.

        This is the shape the GarmentIQ landmark stages produce. Landmarks without
        coordinates, which the derivation stage can leave behind, are skipped.

        Returns:
            dict: Landmark id -> `{"x_mm": float, "y_mm": float}`.
        """
        usable = [
            (landmark_id, point)
            for landmark_id, point in landmarks.items()
            if "x" in point and "y" in point
        ]
        if not usable:
            return {}
        points_mm = self.to_mm([[point["x"], point["y"]] for _, point in usable])
        return {
            landmark_id: {"x_mm": float(x), "y_mm": float(y)}
            for (landmark_id, _), (x, y) in zip(usable, points_mm)
        }

    def __str__(self):
        return (
            f"FrameCalibration({self.n_markers} markers, residual "
            f"{self.rmse_mm:.3f} mm, surface +{self.surface_offset_mm:.1f} mm)"
        )


def calibrate_frame(
    image,
    board,
    intrinsics=None,
    surface_offset_mm=0.0,
    detector=None,
    ransac_threshold_px=3.0,
    min_markers=2,
):
    """Builds the pixel-to-millimetre transform for one frame.

    Args:
        image (str | pathlib.Path | numpy.ndarray): The captured frame, raw (not
            undistorted) — points are undistorted individually, which is both cheaper
            and more accurate than resampling the whole image.
        board (BoardSpec): The learned board layout.
        intrinsics (CameraIntrinsics, optional): Calibrated lens. Strongly recommended;
            without it, lens distortion is left in the measurements.
        surface_offset_mm (float): Fabric thickness, so the plane is lifted to the top
            of the garment. Needs `intrinsics`; ignored without it.
        detector (cv2.aruco.ArucoDetector, optional): Reuse a detector across frames.
        ransac_threshold_px (float): Inlier threshold when fitting, in pixels.
        min_markers (int): Fewest board markers the frame must show.

    Returns:
        FrameCalibration: The transform, with fit diagnostics.

    Raises:
        DetectionError: If a marker id is detected twice.
        HomographyError: If too few markers are visible, or they are collinear.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = load_image(image)

    corners_px = detect_markers(
        image,
        dictionary=board.dictionary,
        keep_ids=board.marker_ids,
        detector=detector,
    )
    undistorted = undistort_marker_corners(corners_px, intrinsics)

    H, info = estimate_homography(
        undistorted, board, ransac_threshold_px=ransac_threshold_px, min_markers=min_markers
    )
    if intrinsics is not None and surface_offset_mm:
        H = lift_to_surface(H, intrinsics, surface_offset_mm)
    elif surface_offset_mm:
        info["surface_offset_skipped"] = (
            "A surface offset was given but no camera intrinsics, so the plane was not "
            "lifted. Calibrate the lens to apply it."
        )

    return FrameCalibration(
        H=H,
        board=board,
        intrinsics=intrinsics,
        surface_offset_mm=float(surface_offset_mm) if intrinsics is not None else 0.0,
        corners_px=corners_px,
        info=info,
    )


def measure_pixel_distances(distances_px, calibration, points_px):
    """Converts a `{name: pixel distance}` dictionary to millimetres.

    Pixel distances cannot be rescaled by a single factor — perspective makes the
    scale different at every point — so each measurement is re-derived from its two
    endpoints instead of being multiplied by anything.

    Args:
        distances_px (dict): Measurement name -> pixel distance (used only for its
            keys and to report the pixel value alongside).
        calibration (FrameCalibration): The frame's transform.
        points_px (dict): Measurement name -> ((x1, y1), (x2, y2)) endpoints in raw
            frame pixels.

    Returns:
        dict: Measurement name -> `{"distance_px", "distance_mm", "tolerance_mm"}`.

    Raises:
        KeyError: If a measurement in `distances_px` has no endpoints in `points_px`.
    """
    results = {}
    for name in distances_px:
        start, end = points_px[name]
        results[name] = {
            "distance_px": float(distances_px[name]),
            "distance_mm": calibration.distance(start, end),
            "tolerance_mm": calibration.uncertainty(start, end),
        }
    return results


def board_area_px(calibration, image_size):
    """Fraction of the frame that lies inside the board's marker hull.

    A measurement taken outside the markers is extrapolated, not interpolated, and is
    the first thing to suspect when a number looks wrong.
    """
    points = np.vstack(list(calibration.corners_px.values()))
    hull_width = points[:, 0].max() - points[:, 0].min()
    hull_height = points[:, 1].max() - points[:, 1].min()
    return float(hull_width * hull_height) / float(image_size[0] * image_size[1])


def is_inside_markers(calibration, points_px, margin_px=0.0):
    """Flags which points fall inside the detected markers' bounding box.

    Returns:
        numpy.ndarray: (N,) booleans, True where the point is safely interpolated.
    """
    corners = np.vstack(list(calibration.corners_px.values()))
    low = corners.min(axis=0) - margin_px
    high = corners.max(axis=0) + margin_px
    points = as_points(points_px)
    return np.all((points >= low) & (points <= high), axis=1)
