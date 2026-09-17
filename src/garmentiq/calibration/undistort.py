# garmentiq/calibration/undistort.py
"""Removing lens distortion, using the result of Step A.

The undistortion maps are built once, so correcting each frame is a single fast
`cv2.remap`. The corrected image keeps the same camera matrix K, so it has the same
size and scale as the raw frame; only the bending of straight lines is removed.

Run on its own to compare raw and corrected frames side by side::

    python -m garmentiq.calibration.undistort
"""
import cv2
import numpy as np

from garmentiq.calibration.camera import Camera, check_frame_size, open_window, put_text
from garmentiq.calibration.config import INTRINSICS_PATH


class Undistorter:
    """Undistorts frames and pixel points for one calibrated camera."""

    def __init__(self, intrinsics_path=INTRINSICS_PATH):
        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"{intrinsics_path} not found. Run capture_lens, then calibrate_lens."
            )
        data = np.load(intrinsics_path)
        self.K = data["K"]
        self.dist = data["dist"]
        self.image_size = tuple(int(value) for value in data["image_size"])
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, self.image_size, cv2.CV_32FC1
        )

    def apply(self, frame):
        """Returns the undistorted frame.

        Raises:
            ValueError: If the frame is not the size the lens was calibrated at.
        """
        check_frame_size(frame, self.image_size, what="the lens calibration")
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def points(self, pixel_points):
        """Moves points from the raw frame to where they are in the undistorted frame.

        Args:
            pixel_points: (x, y) pairs from the raw camera frame.

        Returns:
            numpy.ndarray: (N, 2) positions in the undistorted frame.
        """
        points = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(points, self.K, self.dist, P=self.K).reshape(-1, 2)


def main():
    undistorter = Undistorter()
    print("K =\n", undistorter.K)
    print("dist =", undistorter.dist)

    window = "Raw (left) vs undistorted (right) - q to quit"
    open_window(window)
    cv2.resizeWindow(window, undistorter.image_size[0] * 2, undistorter.image_size[1])

    with Camera() as camera:
        while True:
            raw = camera.read()
            corrected = undistorter.apply(raw)
            put_text(raw, "Raw")
            put_text(corrected, "Undistorted")
            cv2.imshow(window, np.hstack([raw, corrected]))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
