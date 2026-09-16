# garmentiq/calibration/camera.py
"""Calibrating the camera's intrinsics and lens distortion from board photos.

A homography assumes a perfect pinhole camera, and a phone lens is not one. The
distortion is measured once here, from photos of the board taken at many angles, and
removed from every point before it becomes millimetres. Tilted views are what make
the focal length observable, so a set of flat overhead shots calibrates badly no
matter how many there are.

The result is valid only for the lens, zoom, focus and **resolution** it was captured
at. Change any of those and calibrate again.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


class CameraCalibrationError(ValueError):
    """Raised when camera intrinsics cannot be estimated reliably."""


@dataclass
class CameraIntrinsics:
    """A calibrated lens.

    Attributes:
        camera_matrix (numpy.ndarray): (3, 3) [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        dist_coeffs (numpy.ndarray): Distortion coefficients (k1, k2, p1, p2, k3).
        image_size (tuple): (width, height) the calibration is valid for.
        rms_px (float): Overall reprojection RMS, in pixels.
        diagnostics (dict): Per-view errors, frame coverage, recovered camera height
            and any capture warnings.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple
    rms_px: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).ravel()
        self.image_size = tuple(int(value) for value in self.image_size)
        self.rms_px = float(self.rms_px)

    @property
    def focal_length_px(self):
        """(fx, fy) in pixels."""
        return float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])

    @property
    def principal_point_px(self):
        """(cx, cy) in pixels."""
        return float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])

    def to_dict(self):
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "rms_px": self.rms_px,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            camera_matrix=data["camera_matrix"],
            dist_coeffs=data["dist_coeffs"],
            image_size=data["image_size"],
            rms_px=data.get("rms_px", 0.0),
            diagnostics=data.get("diagnostics", {}),
        )

    def save_json(self, path):
        """Writes the intrinsics to JSON, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path):
        """Reads intrinsics written by `save_json`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No camera file at {path}. Run the camera calibration step first "
                f"(python -m garmentiq.calibration.bootstrap)."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __str__(self):
        fx, fy = self.focal_length_px
        return (
            f"CameraIntrinsics(fx={fx:.1f}, fy={fy:.1f}, "
            f"{self.image_size[0]}x{self.image_size[1]}, RMS {self.rms_px:.3f} px)"
        )


def _coverage_fraction(points_px, image_size, grid):
    """Share of the frame's grid cells that hold at least one corner."""
    width, height = image_size
    columns = np.clip((points_px[:, 0] / width * grid[0]).astype(int), 0, grid[0] - 1)
    rows = np.clip((points_px[:, 1] / height * grid[1]).astype(int), 0, grid[1] - 1)
    return len(set(zip(columns.tolist(), rows.tolist()))) / (grid[0] * grid[1])


def calibrate_camera(
    views,
    board,
    flags=cv2.CALIB_FIX_K3,
    min_views=8,
    min_markers_per_view=2,
    max_view_rms_px=None,
    coverage_grid=(8, 6),
):
    """Estimates intrinsics and distortion from photos of the board.

    Views at a different resolution, or showing too few markers, are dropped first.
    The worst-reprojecting view is then removed and the calibration repeated while it
    exceeds `max_view_rms_px` (by default three times the median, and at least 0.5 px)
    and more than `min_views` remain — one blurred photo otherwise drags the whole
    lens model with it.

    Args:
        views (list): Detections from `detect.detect_in_images`, with `name`,
            `corners` and `image_size`.
        board (BoardSpec): The learned board layout, used as the 3D reference.
        flags (int): `cv2.calibrateCamera` flags. `CALIB_FIX_K3` keeps the model
            stable when each photo contributes only a handful of markers.
        min_views (int): Fewest usable views accepted.
        min_markers_per_view (int): Fewest board markers a view must show.
        max_view_rms_px (float, optional): Fixed per-view rejection threshold.
        coverage_grid (tuple): (columns, rows) grid for the frame-coverage check.

    Returns:
        CameraIntrinsics: The calibrated lens, with diagnostics and warnings.

    Raises:
        CameraCalibrationError: If too few usable views remain, or OpenCV fails.
    """
    usable, rejected = [], []
    sizes = [tuple(view["image_size"]) for view in views]
    if not sizes:
        raise CameraCalibrationError("No views were given for camera calibration.")
    image_size = max(set(sizes), key=sizes.count)

    for view in views:
        ids = sorted(set(view["corners"]) & set(board.corners_mm))
        if tuple(view["image_size"]) != image_size:
            rejected.append((view["name"], f"resolution {view['image_size']} differs"))
        elif len(ids) < min_markers_per_view:
            rejected.append((view["name"], f"only {len(ids)} board markers visible"))
        else:
            usable.append({"name": view["name"], "ids": ids, "corners": view["corners"]})

    if len(usable) < min_views:
        raise CameraCalibrationError(
            f"Camera calibration needs at least {min_views} usable views, got "
            f"{len(usable)} of {len(views)}. Rejected: {rejected}"
        )

    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    while True:
        object_points = [
            board.object_points(view["ids"]).astype(np.float32) for view in usable
        ]
        image_points = [
            np.vstack([view["corners"][i] for i in view["ids"]]).astype(np.float32)
            for view in usable
        ]
        try:
            rms, K, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
                object_points, image_points, image_size, None, None,
                flags=flags, criteria=criteria,
            )
        except cv2.error as error:
            raise CameraCalibrationError(
                f"OpenCV could not calibrate the camera: {error}"
            ) from error

        per_view = per_view.ravel().astype(np.float64)
        limit = (
            float(max_view_rms_px)
            if max_view_rms_px is not None
            else max(3.0 * float(np.median(per_view)), 0.5)
        )
        worst = int(np.argmax(per_view))
        if per_view[worst] <= limit or len(usable) <= min_views:
            break
        rejected.append(
            (usable[worst]["name"], f"reprojection RMS {per_view[worst]:.2f} px > {limit:.2f} px")
        )
        usable.pop(worst)

    all_points = np.vstack(image_points)
    coverage = _coverage_fraction(all_points, image_size, coverage_grid)

    # Camera centre in board coordinates; its Z is the perpendicular board distance.
    heights, tilts = [], []
    for rvec, tvec in zip(rvecs, tvecs):
        R = cv2.Rodrigues(rvec)[0]
        centre = -R.T @ np.asarray(tvec, dtype=np.float64).reshape(3)
        heights.append(abs(float(centre[2])))
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(float(R[2, 2])))))))

    diagnostics = {
        "n_views_used": len(usable),
        "view_names": [view["name"] for view in usable],
        "per_view_rms_px": {
            view["name"]: round(float(value), 4)
            for view, value in zip(usable, per_view)
        },
        "coverage_fraction": round(float(coverage), 4),
        "board_distance_mm": round(float(np.median(heights)), 1),
        "board_distance_range_mm": [round(min(heights), 1), round(max(heights), 1)],
        "n_views_tilted_20deg": int(np.sum(np.array(tilts) >= 20.0)),
        "rejected": rejected,
    }
    diagnostics["warnings"] = _warnings(rms, diagnostics, len(usable))

    return CameraIntrinsics(
        camera_matrix=K,
        dist_coeffs=dist,
        image_size=image_size,
        rms_px=float(rms),
        diagnostics=diagnostics,
    )


def _warnings(rms, diagnostics, n_views):
    """Capture problems worth fixing before trusting the intrinsics."""
    warnings = []
    if rms > 1.0:
        warnings.append(
            f"Reprojection RMS is {rms:.2f} px (expect under 1 px): check focus, "
            f"motion blur and that the board is flat."
        )
    if diagnostics["coverage_fraction"] < 0.7:
        warnings.append(
            f"Markers cover only {diagnostics['coverage_fraction']:.0%} of the frame: "
            f"add photos with the board near the frame's corners and edges, where "
            f"distortion is strongest and currently unmeasured."
        )
    if diagnostics["n_views_tilted_20deg"] < 5:
        warnings.append(
            f"Only {diagnostics['n_views_tilted_20deg']} views are tilted 20 degrees "
            f"or more. Without tilt the focal length is poorly determined — add "
            f"angled shots."
        )
    if n_views < 15:
        warnings.append(f"Only {n_views} views kept; 20-40 give a more stable result.")
    return warnings
