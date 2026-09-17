# garmentiq/calibration/metrology.py
"""Measuring in millimetres: the part the rest of GarmentIQ uses.

Loads the results of Step A (lens) and Step B (table homography) and turns pixel
positions on the table into millimetres::

    from garmentiq.calibration.metrology import PlaneMeasurer

    measurer = PlaneMeasurer()
    frame = measurer.undistort(raw_frame)
    length_mm = measurer.distance_mm((120, 200), (400, 210))

Points must lie on the calibrated table surface. Anything above it (a thick seam, a
folded hem) is nearer the camera, looks bigger and measures slightly long.
"""
import cv2
import numpy as np

from garmentiq.calibration.config import HOMOGRAPHY_PATH, INTRINSICS_PATH
from garmentiq.calibration.undistort import Undistorter


class PlaneMeasurer:
    """Pixel -> mm conversion for points on the calibrated table."""

    def __init__(
        self, homography_path=HOMOGRAPHY_PATH, intrinsics_path=INTRINSICS_PATH
    ):
        if not homography_path.exists():
            raise FileNotFoundError(
                f"{homography_path} not found. Run calibrate_plane first."
            )
        data = np.load(homography_path)
        self.H = data["H"]
        self.image_size = tuple(int(value) for value in data["image_size"])
        self.undistorter = Undistorter(intrinsics_path)

        if self.image_size != self.undistorter.image_size:
            raise ValueError(
                f"Homography was made at {self.image_size} but the lens calibration at "
                f"{self.undistorter.image_size}. Redo calibrate_plane."
            )

    def undistort(self, frame):
        """Returns the undistorted frame. Pick points to measure from this image."""
        return self.undistorter.apply(frame)

    def to_mm(self, pixel_points):
        """Converts points from the UNDISTORTED frame to table positions.

        Args:
            pixel_points: (x, y) pairs, e.g. [(120, 200), (400, 210)].

        Returns:
            numpy.ndarray: (N, 2) positions in mm on the table.
        """
        points = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.H).reshape(-1, 2)

    def raw_to_mm(self, pixel_points):
        """Same as `to_mm`, for points taken from the RAW (distorted) camera frame."""
        return self.to_mm(self.undistorter.points(pixel_points))

    def distance_mm(self, point_1, point_2):
        """Distance in mm between two points of the UNDISTORTED frame."""
        a, b = self.to_mm([point_1, point_2])
        return float(np.linalg.norm(a - b))

    def mm_per_pixel(self, point):
        """How many mm one pixel covers at `point` (undistorted frame).

        This is the resolution limit: a landmark that is 1 px off is this many mm off.
        """
        x, y = point
        horizontal = self.distance_mm((x, y), (x + 1, y))
        vertical = self.distance_mm((x, y), (x, y + 1))
        return (horizontal + vertical) / 2
