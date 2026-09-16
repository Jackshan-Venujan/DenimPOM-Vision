"""Tests for garmentiq.calibration, on synthetic geometry.

Everything here is generated: a ground-truth board is invented, a virtual camera
photographs it, and the calibration has to recover what was put in. That makes the
answers exactly knowable, which no real photo allows, and it runs in seconds with no
camera, no models and no network.

One test class per module, so a failing stage can be run on its own::

    pytest test/test_calibration.py::TestLearnBoard -v
"""
import json
import math

import cv2
import numpy as np
import pytest

from garmentiq.calibration.board import BoardSpec
from garmentiq.calibration.calibrate import calibrate_frame, is_inside_markers
from garmentiq.calibration.camera import CameraIntrinsics, calibrate_camera
from garmentiq.calibration.detect import (
    detect_markers,
    dictionary_name,
    identify_dictionary,
)
from garmentiq.calibration.homography import HomographyError, estimate_homography
from garmentiq.calibration.learn_board import (
    BoardLearningError,
    compare_boards,
    learn_board,
)
from garmentiq.calibration.metric import (
    distance_mm,
    local_mm_per_px,
    measure_distance,
    measurement_uncertainty_mm,
    pixel_to_mm,
)
from garmentiq.calibration.surface import (
    expected_scale_error,
    lift_to_surface,
    plane_pose,
)
from garmentiq.calibration.undistort import undistort_points

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 50.0
IMAGE_SIZE = (1600, 1200)  # width, height


# ----------------------------------------------------------------------------
# Synthetic world
# ----------------------------------------------------------------------------
def make_intrinsics(focal_px=1800.0, distortion=None):
    """A plausible phone camera; `distortion` as (k1, k2, p1, p2, k3)."""
    width, height = IMAGE_SIZE
    K = np.array(
        [[focal_px, 0.0, width / 2], [0.0, focal_px, height / 2], [0.0, 0.0, 1.0]]
    )
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    return CameraIntrinsics(camera_matrix=K, dist_coeffs=dist, image_size=IMAGE_SIZE)


def make_ground_truth_board(n_markers=9, area_mm=(600.0, 400.0), seed=0, sizes=None):
    """An irregular hand-made board: markers at jittered positions and free rotations.

    Nothing about this is a grid, which is the point — the solver must not need one.
    """
    rng = np.random.default_rng(seed)
    columns = int(math.ceil(math.sqrt(n_markers)))
    rows = int(math.ceil(n_markers / columns))
    corners = {}

    for index in range(n_markers):
        column, row = index % columns, index // columns
        # Regular cell, then shoved around inside it so no two gaps are equal.
        x = (column + 0.5) / columns * area_mm[0] + rng.uniform(-25.0, 25.0)
        y = (row + 0.5) / rows * area_mm[1] + rng.uniform(-20.0, 20.0)
        angle = rng.uniform(-math.pi, math.pi)
        size = MARKER_SIZE_MM if sizes is None else sizes[index]

        template = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * size
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        corners[index] = template @ rotation.T + np.array([x, y])

    return BoardSpec(
        dictionary=DICTIONARY, marker_size_mm=MARKER_SIZE_MM, corners_mm=corners
    )


def make_pose(board, distance_mm=1200.0, tilt_x=0.0, tilt_y=0.0, roll=0.0, shift=(0.0, 0.0)):
    """Places the camera above the board's centre, then tilts and shifts it.

    Returns:
        tuple: (rvec, tvec) taking board coordinates into the camera frame.
    """
    points = np.vstack(list(board.corners_mm.values()))
    centre = points.mean(axis=0)

    def rotation_x(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    def rotation_y(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def rotation_z(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    R = rotation_z(roll) @ rotation_y(tilt_y) @ rotation_x(tilt_x)
    # Camera sits at negative Z in board coordinates, so the board is in front of it.
    camera_centre = np.array([centre[0] + shift[0], centre[1] + shift[1], -distance_mm])
    tvec = -R @ camera_centre
    return cv2.Rodrigues(R)[0].ravel(), tvec


def project(points_mm, rvec, tvec, intrinsics, z_mm=0.0):
    """Projects board-plane points (optionally lifted by `z_mm`) into the image."""
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
    object_points = np.column_stack([points, np.full(len(points), float(z_mm))])
    projected, _ = cv2.projectPoints(
        object_points, np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        intrinsics.camera_matrix, intrinsics.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def make_view(board, rvec, tvec, intrinsics, noise_px=0.0, rng=None, name="view",
              marker_ids=None):
    """Photographs the board, returning a detection as `detect_in_images` would."""
    ids = board.marker_ids if marker_ids is None else list(marker_ids)
    corners = {}
    for marker_id in ids:
        points = project(board.corners_mm[marker_id], rvec, tvec, intrinsics)
        if noise_px and rng is not None:
            points = points + rng.normal(0.0, noise_px, points.shape)
        corners[marker_id] = points
    return {"name": name, "corners": corners, "image_size": IMAGE_SIZE}


def make_view_set(board, intrinsics, n_views=15, noise_px=0.05, seed=1):
    """A spread of viewpoints: varied tilt, roll, distance and framing."""
    rng = np.random.default_rng(seed)
    views = []
    for index in range(n_views):
        rvec, tvec = make_pose(
            board,
            distance_mm=rng.uniform(1000.0, 1500.0),
            tilt_x=rng.uniform(-0.35, 0.35),
            tilt_y=rng.uniform(-0.35, 0.35),
            roll=rng.uniform(-0.3, 0.3),
            shift=(rng.uniform(-80.0, 80.0), rng.uniform(-60.0, 60.0)),
        )
        views.append(make_view(board, rvec, tvec, intrinsics, noise_px, rng, f"view_{index}"))
    return views


def render_board_image(board, intrinsics, rvec, tvec, pixels_per_mm=4.0, margin_mm=40.0):
    """Renders a real photograph of the board: actual marker bitmaps, warped.

    Needed wherever the ArUco detector is in the loop, because it needs pixels, not
    projected corner coordinates.
    """
    points = np.vstack(list(board.corners_mm.values()))
    low = points.min(axis=0) - margin_mm
    high = points.max(axis=0) + margin_mm
    canvas_size = np.ceil((high - low) * pixels_per_mm).astype(int)
    canvas = np.full((int(canvas_size[1]), int(canvas_size[0])), 255, dtype=np.uint8)

    aruco_dict = cv2.aruco.getPredefinedDictionary(board.dictionary)
    for marker_id, corners_mm in board.corners_mm.items():
        side = 240
        marker = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), side)
        source = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float64)
        target = (corners_mm - low) * pixels_per_mm
        M = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
        cv2.warpPerspective(
            marker, M, (canvas.shape[1], canvas.shape[0]), dst=canvas,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    # Board millimetres -> image pixels, then canvas pixels -> board millimetres.
    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    board_to_image = intrinsics.camera_matrix @ np.column_stack(
        [R[:, 0], R[:, 1], np.asarray(tvec, dtype=np.float64).reshape(3)]
    )
    canvas_to_board = np.array(
        [[1.0 / pixels_per_mm, 0.0, low[0]], [0.0, 1.0 / pixels_per_mm, low[1]], [0.0, 0.0, 1.0]]
    )
    image = cv2.warpPerspective(
        canvas, board_to_image @ canvas_to_board, IMAGE_SIZE,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------
class TestBoardSpec:
    def test_json_round_trip(self, tmp_path):
        board = make_ground_truth_board(seed=3)
        path = board.save_json(tmp_path / "board.json")
        loaded = BoardSpec.load_json(path)

        assert loaded.marker_ids == board.marker_ids
        assert loaded.dictionary == board.dictionary
        for marker_id in board.marker_ids:
            np.testing.assert_allclose(
                loaded.corners_mm[marker_id], board.corners_mm[marker_id]
            )

    def test_points_and_object_points_agree(self):
        board = make_ground_truth_board(n_markers=4, seed=4)
        ids = board.marker_ids
        flat, spatial = board.points_mm(ids), board.object_points(ids)

        assert flat.shape == (16, 2)
        assert spatial.shape == (16, 3)
        np.testing.assert_allclose(spatial[:, :2], flat)
        np.testing.assert_allclose(spatial[:, 2], 0.0)

    def test_measured_sizes_match_the_drawn_squares(self):
        board = make_ground_truth_board(n_markers=5, seed=5)
        for size in board.measured_marker_sizes_mm().values():
            assert size == pytest.approx(MARKER_SIZE_MM, abs=1e-9)

    def test_missing_file_names_the_next_step(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="bootstrap"):
            BoardSpec.load_json(tmp_path / "absent.json")

    def test_empty_board_is_refused(self):
        with pytest.raises(ValueError, match="at least one marker"):
            BoardSpec(dictionary=DICTIONARY, marker_size_mm=50.0, corners_mm={})


# ----------------------------------------------------------------------------
class TestDetect:
    def test_detects_every_marker_with_correct_corner_order(self):
        board = make_ground_truth_board(n_markers=6, seed=6)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board)
        image = render_board_image(board, intrinsics, rvec, tvec)

        found = detect_markers(image, dictionary=DICTIONARY)
        assert sorted(found) == board.marker_ids

        # Each detected corner should sit where that marker's corner was projected.
        for marker_id, corners in found.items():
            expected = project(board.corners_mm[marker_id], rvec, tvec, intrinsics)
            np.testing.assert_allclose(corners, expected, atol=1.5)

    def test_identify_dictionary_finds_the_right_one(self):
        board = make_ground_truth_board(n_markers=4, seed=7)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board)
        image = render_board_image(board, intrinsics, rvec, tvec)

        matches = identify_dictionary(image)
        assert matches, "no dictionary matched a rendered board"
        assert matches[0]["dictionary"] == DICTIONARY
        assert matches[0]["marker_ids"] == board.marker_ids

    def test_keep_ids_ignores_strangers(self):
        board = make_ground_truth_board(n_markers=6, seed=8)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board)
        image = render_board_image(board, intrinsics, rvec, tvec)

        found = detect_markers(image, dictionary=DICTIONARY, keep_ids=[0, 1])
        assert sorted(found) == [0, 1]

    def test_dictionary_name_is_readable(self):
        assert dictionary_name(DICTIONARY) == "DICT_4X4_50"


# ----------------------------------------------------------------------------
class TestLearnBoard:
    def test_recovers_an_irregular_hand_made_layout(self):
        truth = make_ground_truth_board(n_markers=9, seed=10)
        intrinsics = make_intrinsics()
        views = make_view_set(truth, intrinsics, n_views=15, noise_px=0.05, seed=11)

        learned = learn_board(views, MARKER_SIZE_MM, DICTIONARY)

        assert learned.marker_ids == truth.marker_ids
        difference = compare_boards(truth, learned)
        # 0.05 px of corner noise on a 600 x 400 mm board. The joint fit gets this to
        # hundredths of a millimetre; the alternating passes alone stall near 1 mm,
        # so this threshold is what guards the joint fit against regressing.
        assert difference["rms_mm"] < 0.05, difference
        assert difference["max_mm"] < 0.1, difference
        assert learned.diagnostics["jointly_refined"]

    def test_stays_accurate_with_noisier_corners(self):
        truth = make_ground_truth_board(n_markers=9, seed=10)
        intrinsics = make_intrinsics()
        views = make_view_set(truth, intrinsics, n_views=15, noise_px=0.2, seed=11)

        learned = learn_board(views, MARKER_SIZE_MM, DICTIONARY)

        assert compare_boards(truth, learned)["rms_mm"] < 0.15

    def test_noise_free_capture_is_recovered_exactly(self):
        truth = make_ground_truth_board(n_markers=9, seed=10)
        views = make_view_set(truth, make_intrinsics(), n_views=12, noise_px=0.0, seed=11)

        learned = learn_board(views, MARKER_SIZE_MM, DICTIONARY)

        assert compare_boards(truth, learned)["max_mm"] < 0.001

    def test_origin_choice_does_not_change_distances(self):
        """The millimetre origin is arbitrary, so measurements must ignore it."""
        truth = make_ground_truth_board(n_markers=6, seed=12)
        intrinsics = make_intrinsics()
        views = make_view_set(truth, intrinsics, n_views=12, noise_px=0.0, seed=13)

        first = learn_board(views, MARKER_SIZE_MM, DICTIONARY, reference_id=0)
        second = learn_board(views, MARKER_SIZE_MM, DICTIONARY, reference_id=3)

        def span(board):
            points = np.vstack([board.corners_mm[i] for i in board.marker_ids])
            return np.linalg.norm(points[0] - points[-1])

        assert span(first) == pytest.approx(span(second), abs=0.05)

    def test_flags_a_marker_seen_only_once(self):
        truth = make_ground_truth_board(n_markers=6, seed=14)
        intrinsics = make_intrinsics()
        views = make_view_set(truth, intrinsics, n_views=10, noise_px=0.05, seed=15)
        # Marker 5 appears in one view only.
        for view in views[1:]:
            view["corners"].pop(5, None)

        learned = learn_board(views, MARKER_SIZE_MM, DICTIONARY)

        assert 5 in learned.diagnostics["weakly_constrained_ids"]
        assert any("fewer than 3" in w for w in learned.diagnostics["warnings"])

    def test_detects_a_board_with_mixed_marker_sizes(self):
        sizes = [50.0, 50.0, 50.0, 40.0, 50.0, 50.0]
        truth = make_ground_truth_board(n_markers=6, seed=16, sizes=sizes)
        intrinsics = make_intrinsics()
        views = make_view_set(truth, intrinsics, n_views=12, noise_px=0.05, seed=17)

        learned = learn_board(views, MARKER_SIZE_MM, DICTIONARY)

        assert not learned.diagnostics["uniform_marker_size"]
        assert any("different sizes" in w for w in learned.diagnostics["warnings"])
        measured = learned.measured_marker_sizes_mm()
        assert measured[3] == pytest.approx(40.0, abs=0.5)

    def test_warns_when_solved_without_a_lens_calibration(self):
        truth = make_ground_truth_board(n_markers=6, seed=18)
        views = make_view_set(truth, make_intrinsics(), n_views=10, seed=19)

        learned = learn_board(views, MARKER_SIZE_MM, DICTIONARY, intrinsics=None)

        assert any("lens calibration" in w for w in learned.diagnostics["warnings"])

    def test_marker_never_seen_with_the_others_is_reported(self):
        truth = make_ground_truth_board(n_markers=6, seed=20)
        intrinsics = make_intrinsics()
        views = make_view_set(truth, intrinsics, n_views=8, seed=21)
        for view in views:
            view["corners"].pop(5, None)
        # One view shows marker 5 and nothing else, so it can never be placed.
        rvec, tvec = make_pose(truth)
        views.append(make_view(truth, rvec, tvec, intrinsics, name="lonely", marker_ids=[5]))

        with pytest.raises(BoardLearningError, match=r"\[5\]"):
            learn_board(views, MARKER_SIZE_MM, DICTIONARY)

    def test_rejects_a_nonsense_marker_size(self):
        views = make_view_set(make_ground_truth_board(seed=22), make_intrinsics(), 4)
        with pytest.raises(BoardLearningError, match="must be positive"):
            learn_board(views, 0.0, DICTIONARY)


# ----------------------------------------------------------------------------
class TestHomography:
    def _board_and_view(self, seed=30, noise_px=0.0):
        board = make_ground_truth_board(n_markers=6, seed=seed)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board, tilt_x=0.2, tilt_y=-0.15)
        rng = np.random.default_rng(seed)
        view = make_view(board, rvec, tvec, intrinsics, noise_px, rng)
        return board, view, (rvec, tvec), intrinsics

    def test_noise_free_fit_is_exact(self):
        board, view, _, _ = self._board_and_view()

        H, info = estimate_homography(view["corners"], board)

        assert info["rmse_mm"] < 0.01
        assert info["n_markers"] == 6
        for marker_id, corners in view["corners"].items():
            np.testing.assert_allclose(
                pixel_to_mm(corners, H), board.corners_mm[marker_id], atol=0.01
            )

    def test_ransac_rejects_a_misdetected_marker(self):
        board, view, _, _ = self._board_and_view(seed=31, noise_px=0.05)
        view["corners"][2] = view["corners"][2] + np.array([40.0, -25.0])

        H, info = estimate_homography(view["corners"], board)

        assert 2 not in info["inlier_ids"]
        assert info["marker_errors_mm"][2] > 5.0
        # The other markers still agree with each other.
        others = [e for i, e in info["marker_errors_mm"].items() if i != 2]
        assert max(others) < 0.5

    def test_needs_more_than_one_marker(self):
        board, view, _, _ = self._board_and_view(seed=32)
        one = {0: view["corners"][0]}

        with pytest.raises(HomographyError, match="at least 2"):
            estimate_homography(one, board)

    def test_collinear_markers_are_refused(self):
        line = {
            index: np.array([[x, 0.0], [x + 50.0, 0.0], [x + 50.0, 50.0], [x, 50.0]])
            for index, x in enumerate([0.0, 200.0, 400.0])
        }
        # A board whose markers all sit in one row, seen straight on.
        board = BoardSpec(DICTIONARY, MARKER_SIZE_MM, line)
        flat = {i: c * 2.0 + 100.0 for i, c in line.items()}
        for corners in flat.values():
            corners[:, 1] = 500.0  # squash every corner onto one image line

        with pytest.raises(HomographyError, match="single line"):
            estimate_homography(flat, board)

    def test_reports_ids_that_are_not_on_the_board(self):
        board, view, _, _ = self._board_and_view(seed=33)
        stranger = {99: view["corners"][0]}

        with pytest.raises(HomographyError, match="not on the board"):
            estimate_homography(stranger, board)


# ----------------------------------------------------------------------------
class TestSurface:
    def _setup(self, height_mm=1170.0):
        board = make_ground_truth_board(n_markers=9, seed=40)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board, distance_mm=height_mm)
        view = make_view(board, rvec, tvec, intrinsics)
        H, _ = estimate_homography(view["corners"], board)
        return board, intrinsics, rvec, tvec, H

    def test_plane_pose_recovers_the_camera_height(self):
        _, intrinsics, _, _, H = self._setup(height_mm=1170.0)

        _, _, camera_height = plane_pose(H, intrinsics)

        assert camera_height == pytest.approx(1170.0, rel=1e-3)

    def test_lifting_corrects_the_fabric_thickness(self):
        board, intrinsics, rvec, tvec, H = self._setup(height_mm=1170.0)
        offset = 5.0

        # Two points 300 mm apart, lying on top of 5 mm of fabric. The camera is at
        # negative Z in board coordinates, so "up off the board" is negative Z.
        centre = np.vstack(list(board.corners_mm.values())).mean(axis=0)
        a_mm = centre - np.array([150.0, 0.0])
        b_mm = centre + np.array([150.0, 0.0])
        a_px, b_px = project([a_mm, b_mm], rvec, tvec, intrinsics, z_mm=-offset)

        on_board = measure_distance(a_px, b_px, H)
        lifted = lift_to_surface(H, intrinsics, offset)
        on_fabric = measure_distance(a_px, b_px, lifted)

        assert on_fabric == pytest.approx(300.0, abs=0.05)
        # Uncorrected, the fabric reads too long by about offset / height.
        assert on_board == pytest.approx(300.0 * 1170.0 / 1165.0, rel=1e-3)
        assert on_board - on_fabric == pytest.approx(1.29, abs=0.1)

    def test_expected_scale_error_matches_what_lifting_removes(self):
        assert expected_scale_error(5.0, 1170.0) == pytest.approx(0.00427, rel=0.01)
        # 4.3 mm on a metre: the number that makes this module necessary.
        assert expected_scale_error(5.0, 1170.0) * 1000.0 == pytest.approx(4.3, abs=0.1)

    def test_no_intrinsics_is_an_exact_no_op(self):
        _, _, _, _, H = self._setup()
        np.testing.assert_array_equal(lift_to_surface(H, None, 5.0), H)

    def test_zero_offset_is_an_exact_no_op(self):
        _, intrinsics, _, _, H = self._setup()
        np.testing.assert_array_equal(lift_to_surface(H, intrinsics, 0.0), H)


# ----------------------------------------------------------------------------
class TestMetric:
    def _similarity(self, mm_per_px=0.5, angle=0.3, offset=(120.0, -40.0)):
        c, s = math.cos(angle), math.sin(angle)
        return np.array(
            [
                [mm_per_px * c, -mm_per_px * s, offset[0]],
                [mm_per_px * s, mm_per_px * c, offset[1]],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_distances_through_a_pure_similarity(self):
        H = self._similarity(mm_per_px=0.5)
        # 400 px apart at 0.5 mm per px is 200 mm, whatever the rotation.
        assert measure_distance([0.0, 0.0], [400.0, 0.0], H) == pytest.approx(200.0)
        assert measure_distance([0.0, 0.0], [0.0, 400.0], H) == pytest.approx(200.0)
        assert measure_distance([10.0, 10.0], [310.0, 410.0], H) == pytest.approx(250.0)

    def test_local_scale_is_constant_without_perspective(self):
        H = self._similarity(mm_per_px=0.37)
        scales = local_mm_per_px([[0, 0], [1500, 1100], [800, 20]], H)
        np.testing.assert_allclose(scales, 0.37, rtol=1e-9)

    def test_local_scale_varies_under_real_perspective(self):
        board = make_ground_truth_board(n_markers=9, seed=50)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board, tilt_x=0.5)
        view = make_view(board, rvec, tvec, intrinsics)
        H, _ = estimate_homography(view["corners"], board)

        scales = local_mm_per_px([[200, 150], [1400, 1050]], H)
        assert scales.max() / scales.min() > 1.05, "a 0.5 rad tilt should change the scale"

    def test_round_trip_through_millimetres(self):
        H = self._similarity()
        points = np.array([[12.0, 34.0], [560.0, 780.0]])
        from garmentiq.calibration.metric import mm_to_pixel

        np.testing.assert_allclose(mm_to_pixel(pixel_to_mm(points, H), H), points, atol=1e-9)

    def test_uncertainty_grows_with_landmark_noise(self):
        H = self._similarity(mm_per_px=0.5)
        a, b = [0.0, 0.0], [400.0, 0.0]

        one = measurement_uncertainty_mm(a, b, H, point_sigma_px=1.0, coverage_k=2.0)
        two = measurement_uncertainty_mm(a, b, H, point_sigma_px=2.0, coverage_k=2.0)

        # Two independent endpoints, each 1 px sigma, at 0.5 mm/px, k=2:
        # 2 * 0.5 * sqrt(2) = 1.414 mm.
        assert one == pytest.approx(math.sqrt(2.0), rel=1e-6)
        assert two == pytest.approx(2.0 * one, rel=1e-9)

    def test_distance_mm_is_plain_euclidean(self):
        assert distance_mm((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# ----------------------------------------------------------------------------
class TestUndistort:
    def test_undistort_inverts_the_lens_model(self):
        intrinsics = make_intrinsics(distortion=(-0.28, 0.12, 0.001, -0.0005, 0.0))
        # Points out at the frame edges, where distortion is strongest.
        ideal = np.array([[100.0, 80.0], [1500.0, 1120.0], [800.0, 60.0], [60.0, 600.0]])

        normalized = np.column_stack(
            [
                (ideal - intrinsics.principal_point_px) / np.array(intrinsics.focal_length_px),
                np.ones(len(ideal)),
            ]
        )
        distorted = cv2.projectPoints(
            normalized, np.zeros(3), np.zeros(3),
            intrinsics.camera_matrix, intrinsics.dist_coeffs,
        )[0].reshape(-1, 2)

        recovered = undistort_points(distorted, intrinsics)
        np.testing.assert_allclose(recovered, ideal, atol=1e-6)

    def test_no_intrinsics_passes_points_through(self):
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(undistort_points(points, None), points)


# ----------------------------------------------------------------------------
class TestCameraCalibration:
    def test_recovers_focal_length_and_distortion(self):
        truth_intrinsics = make_intrinsics(focal_px=1800.0, distortion=(-0.12, 0.05, 0.0, 0.0, 0.0))
        board = make_ground_truth_board(n_markers=9, seed=60)
        views = make_view_set(board, truth_intrinsics, n_views=24, noise_px=0.05, seed=61)

        estimated = calibrate_camera(views, board, min_views=8)

        fx, fy = estimated.focal_length_px
        assert fx == pytest.approx(1800.0, rel=0.02)
        assert fy == pytest.approx(1800.0, rel=0.02)
        assert estimated.dist_coeffs[0] == pytest.approx(-0.12, abs=0.05)
        assert estimated.rms_px < 1.0
        assert estimated.diagnostics["board_distance_mm"] == pytest.approx(1250.0, rel=0.2)

    def test_too_few_views_is_refused_with_the_count(self):
        board = make_ground_truth_board(n_markers=6, seed=62)
        views = make_view_set(board, make_intrinsics(), n_views=3, seed=63)

        with pytest.raises(Exception, match="at least 8 usable views"):
            calibrate_camera(views, board, min_views=8)

    def test_json_round_trip(self, tmp_path):
        intrinsics = make_intrinsics(distortion=(-0.2, 0.05, 0.0, 0.0, 0.0))
        path = intrinsics.save_json(tmp_path / "camera.json")
        loaded = CameraIntrinsics.load_json(path)

        np.testing.assert_allclose(loaded.camera_matrix, intrinsics.camera_matrix)
        np.testing.assert_allclose(loaded.dist_coeffs, intrinsics.dist_coeffs)
        assert loaded.image_size == intrinsics.image_size
        assert json.loads(path.read_text())["image_size"] == list(IMAGE_SIZE)


# ----------------------------------------------------------------------------
class TestCalibrateFrame:
    def _render(self, seed=70, height_mm=1200.0):
        board = make_ground_truth_board(n_markers=9, seed=seed)
        intrinsics = make_intrinsics()
        rvec, tvec = make_pose(board, distance_mm=height_mm, tilt_x=0.12, tilt_y=-0.08)
        image = render_board_image(board, intrinsics, rvec, tvec)
        return board, intrinsics, rvec, tvec, image

    def test_measures_a_known_distance_from_a_rendered_photo(self):
        board, intrinsics, rvec, tvec, image = self._render()

        calibration = calibrate_frame(image, board, intrinsics)

        assert calibration.n_markers == len(board.marker_ids)
        assert calibration.rmse_mm < 1.0

        centre = np.vstack(list(board.corners_mm.values())).mean(axis=0)
        a_mm = centre - np.array([150.0, 0.0])
        b_mm = centre + np.array([150.0, 0.0])
        a_px, b_px = project([a_mm, b_mm], rvec, tvec, intrinsics)

        assert calibration.distance(a_px, b_px) == pytest.approx(300.0, abs=0.5)

    def test_still_works_when_the_garment_hides_most_markers(self):
        board, intrinsics, rvec, tvec, image = self._render(seed=71)
        visible = board.marker_ids[:3]
        hidden = BoardSpec(board.dictionary, board.marker_size_mm,
                           {i: board.corners_mm[i] for i in visible})

        calibration = calibrate_frame(image, hidden, intrinsics)

        assert calibration.n_markers == 3
        centre = np.vstack([board.corners_mm[i] for i in visible]).mean(axis=0)
        a_px, b_px = project(
            [centre - np.array([100.0, 0.0]), centre + np.array([100.0, 0.0])],
            rvec, tvec, intrinsics,
        )
        assert calibration.distance(a_px, b_px) == pytest.approx(200.0, abs=1.0)

    def test_landmarks_to_mm_handles_the_pipeline_shape(self):
        board, intrinsics, rvec, tvec, image = self._render(seed=72)
        calibration = calibrate_frame(image, board, intrinsics)

        centre = np.vstack(list(board.corners_mm.values())).mean(axis=0)
        a_px, b_px = project(
            [centre, centre + np.array([200.0, 0.0])], rvec, tvec, intrinsics
        )
        landmarks = {
            "1": {"x": float(a_px[0]), "y": float(a_px[1]), "conf": 0.9},
            "2": {"x": float(b_px[0]), "y": float(b_px[1]), "conf": 0.8},
            "3": {"conf": 0.0},  # derivation can leave a landmark without coordinates
        }

        millimetres = calibration.landmarks_to_mm(landmarks)

        assert set(millimetres) == {"1", "2"}
        separation = math.dist(
            (millimetres["1"]["x_mm"], millimetres["1"]["y_mm"]),
            (millimetres["2"]["x_mm"], millimetres["2"]["y_mm"]),
        )
        assert separation == pytest.approx(200.0, abs=0.5)

    def test_flags_points_outside_the_markers(self):
        board, intrinsics, _, _, image = self._render(seed=73)
        calibration = calibrate_frame(image, board, intrinsics)

        corners = np.vstack(list(calibration.corners_px.values()))
        inside = corners.mean(axis=0)
        outside = corners.min(axis=0) - 300.0

        flags = is_inside_markers(calibration, [inside, outside])
        assert flags[0] and not flags[1]

    def test_surface_offset_without_intrinsics_is_reported_not_silent(self):
        board, _, _, _, image = self._render(seed=74)

        calibration = calibrate_frame(image, board, None, surface_offset_mm=5.0)

        assert calibration.surface_offset_mm == 0.0
        assert "surface_offset_skipped" in calibration.info


# ----------------------------------------------------------------------------
class TestEndToEnd:
    def test_bootstrap_beats_leaving_distortion_in(self):
        """The whole point of the bootstrap: correcting the lens improves the board."""
        truth_intrinsics = make_intrinsics(distortion=(-0.15, 0.04, 0.0, 0.0, 0.0))
        truth_board = make_ground_truth_board(n_markers=6, seed=80)
        views = make_view_set(truth_board, truth_intrinsics, n_views=14, noise_px=0.05, seed=81)

        # Pass one: no lens correction, so the layout absorbs the distortion.
        first = learn_board(views, MARKER_SIZE_MM, DICTIONARY)
        lens = calibrate_camera(views, first, min_views=8)
        # Pass two: the same photos, now undistorted.
        second = learn_board(views, MARKER_SIZE_MM, DICTIONARY, intrinsics=lens)

        before = compare_boards(truth_board, first)["rms_mm"]
        after = compare_boards(truth_board, second)["rms_mm"]

        assert after < before, f"correcting the lens made it worse: {before} -> {after}"
        assert after < 0.5
