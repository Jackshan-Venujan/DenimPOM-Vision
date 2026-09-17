"""Tests for garmentiq.calibration, on synthetic geometry.

Everything here is generated: a virtual camera photographs a virtual chessboard, and
the calibration has to recover what was put in. No camera or network is needed.

One test class per module, so a failing stage can be run on its own::

    pytest test/test_calibration.py::TestMetrology -v -o addopts=""
"""
import cv2
import numpy as np
import pytest

from garmentiq.calibration.calibrate_lens import solve_intrinsics
from garmentiq.calibration.calibrate_plane import average_corners, compute_homography
from garmentiq.calibration.chessboard import board_points_mm, find_corners
from garmentiq.calibration.config import INNER_CORNERS, SQUARE_MM
from garmentiq.calibration.metrology import PlaneMeasurer
from garmentiq.calibration.undistort import Undistorter
from garmentiq.calibration.verify_plane import check_board

IMAGE_SIZE = (640, 480)
K_TRUE = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
NO_DISTORTION = np.zeros(5)


def board_3d():
    points = board_points_mm()
    return np.hstack([points, np.zeros((len(points), 1), np.float32)])


def photograph_board(rotation_deg=(0, 0, 0), distance_mm=700):
    """Pixel positions of the board corners seen by the virtual camera."""
    columns, rows = INNER_CORNERS
    centre = np.array([(columns - 1) * SQUARE_MM / 2, (rows - 1) * SQUARE_MM / 2, 0])
    rvec = np.deg2rad(np.array(rotation_deg, dtype=np.float64))
    R, _ = cv2.Rodrigues(rvec)
    tvec = np.array([0, 0, distance_mm]) - R @ centre  # board centre on optical axis
    pixels, _ = cv2.projectPoints(board_3d(), rvec, tvec, K_TRUE, NO_DISTORTION)
    return pixels.reshape(-1, 2).astype(np.float32)


def render_board(square_px=40, margin_px=60):
    """A grayscale image of the printed board: (columns + 1) x (rows + 1) squares."""
    columns, rows = INNER_CORNERS[0] + 1, INNER_CORNERS[1] + 1
    height = rows * square_px + 2 * margin_px
    width = columns * square_px + 2 * margin_px
    image = np.full((height, width), 255, np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                y, x = margin_px + row * square_px, margin_px + column * square_px
                image[y : y + square_px, x : x + square_px] = 0
    return cv2.GaussianBlur(image, (3, 3), 0)


def save_calibration_files(tmp_path, H, homography_size=IMAGE_SIZE):
    intrinsics_path = tmp_path / "camera_intrinsics.npz"
    homography_path = tmp_path / "table_homography.npz"
    np.savez(
        intrinsics_path, K=K_TRUE, dist=NO_DISTORTION, image_size=np.array(IMAGE_SIZE)
    )
    np.savez(homography_path, H=H, image_size=np.array(homography_size))
    return homography_path, intrinsics_path


class TestChessboard:
    def test_board_points_are_row_by_row_in_mm(self):
        points = board_points_mm()
        columns, rows = INNER_CORNERS
        assert points.shape == (columns * rows, 2)
        np.testing.assert_allclose(points[0], [0, 0])
        np.testing.assert_allclose(points[1], [SQUARE_MM, 0])
        np.testing.assert_allclose(points[columns], [0, SQUARE_MM])

    def test_finds_every_corner_on_a_rendered_board(self):
        corners = find_corners(render_board(square_px=40))
        columns, rows = INNER_CORNERS
        assert corners is not None and corners.shape == (columns * rows, 2)

        grid = corners.reshape(rows, columns, 2)
        steps = np.linalg.norm(np.diff(grid, axis=1), axis=-1)
        np.testing.assert_allclose(steps, 40, atol=0.3)

    def test_returns_none_without_a_board(self):
        assert find_corners(np.full((480, 640), 128, np.uint8)) is None


class TestCalibrateLens:
    def test_recovers_the_focal_length(self):
        views = [
            (0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 25, 0), (0, -25, 0),
            (20, 20, 10), (-20, 20, -10), (20, -20, 5), (-20, -20, 0), (15, 0, 30),
        ]  # fmt: skip
        corners = [photograph_board(view) for view in views]

        result = solve_intrinsics(corners, IMAGE_SIZE)

        assert result["rms_px"] < 1e-3
        np.testing.assert_allclose(result["K"], K_TRUE, rtol=1e-3, atol=0.5)
        assert len(result["per_view_rms_px"]) == len(views)


class TestCalibratePlane:
    def test_homography_maps_tilted_board_back_to_mm(self):
        corners_px = photograph_board((20, -15, 5))
        H, errors_mm = compute_homography(corners_px, board_points_mm())
        assert errors_mm.max() < 1e-3

    def test_average_corners_reduces_noise_and_reports_jitter(self):
        truth = photograph_board()
        rng = np.random.default_rng(0)
        frames = [
            truth + rng.normal(0, 0.3, truth.shape).astype(np.float32)
            for _ in range(50)
        ]

        mean, jitter_px = average_corners(frames)

        assert np.abs(mean - truth).max() < 0.2
        assert 0.3 < jitter_px < 0.8


class TestMetrology:
    def test_distance_and_scale(self, tmp_path):
        H = np.diag([0.5, 0.5, 1.0])  # every pixel is 0.5 mm
        measurer = PlaneMeasurer(*save_calibration_files(tmp_path, H))

        assert measurer.distance_mm((0, 0), (100, 0)) == pytest.approx(50)
        assert measurer.mm_per_pixel((320, 240)) == pytest.approx(0.5)
        np.testing.assert_allclose(
            measurer.raw_to_mm([(10, 20)]), measurer.to_mm([(10, 20)]), atol=1e-6
        )

    def test_real_homography_measures_the_board(self, tmp_path):
        corners_px = photograph_board((25, 10, 0))
        H, _ = compute_homography(corners_px, board_points_mm())
        measurer = PlaneMeasurer(*save_calibration_files(tmp_path, H))

        columns = INNER_CORNERS[0]
        width = measurer.distance_mm(corners_px[0], corners_px[columns - 1])
        assert width == pytest.approx((columns - 1) * SQUARE_MM, abs=1e-3)

    def test_rejects_mismatched_resolutions(self, tmp_path):
        paths = save_calibration_files(tmp_path, np.eye(3), homography_size=(1280, 720))
        with pytest.raises(ValueError):
            PlaneMeasurer(*paths)

    def test_undistorter_rejects_wrong_frame_size(self, tmp_path):
        _, intrinsics_path = save_calibration_files(tmp_path, np.eye(3))
        with pytest.raises(ValueError):
            Undistorter(intrinsics_path).apply(np.zeros((720, 1280, 3), np.uint8))


class TestVerifyPlane:
    def test_exact_calibration_passes(self):
        corners_px = photograph_board((15, 15, 0))
        H, _ = compute_homography(corners_px, board_points_mm())

        class Measurer:
            def to_mm(self, points):
                points = np.asarray(points, np.float64).reshape(-1, 1, 2)
                return cv2.perspectiveTransform(points, H).reshape(-1, 2)

        result = check_board(corners_px, Measurer())
        assert result["passed"]
        assert result["error_per_metre_mm"] < 0.01

    def test_one_percent_scale_error_fails(self):
        class OnePercentLong:
            def to_mm(self, points):
                return board_points_mm() * 1.01

        result = check_board(photograph_board(), OnePercentLong())
        assert result["error_per_metre_mm"] == pytest.approx(10, rel=1e-3)
        assert not result["passed"]
