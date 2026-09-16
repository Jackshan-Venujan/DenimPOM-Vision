# garmentiq/calibration/__init__.py
"""Turning pixel measurements into millimetres.

The AI stages of GarmentIQ work in pixels. This one supplies the physical scale, so a
measurement can be judged against a real tolerance. It does not change any of them:
landmarks arrive as pixel coordinates and leave as millimetres.

The scale comes from a printed ArUco board the garment lies on. Because the board is
flat, every photo of it relates to it by an exact homography, so perspective is handled
properly rather than by assuming one pixels-per-millimetre ratio — a 50 mm marker does
not span the same number of pixels at the centre of the frame as at the corner.

The board may be hand-made. Its layout is not typed in but solved from photos by
`learn_board`; the only number a person supplies is the calipered size of one marker,
because a photograph cannot reveal absolute scale. Lens distortion is measured once by
`calibrate_camera`, and `bootstrap` resolves the circular dependency between the two.

Calibrating, once::

    python -m garmentiq.calibration.inspect_board photo.jpg   # which dictionary, which ids
    python -m garmentiq.calibration.capture --mode board      # photos of the bare board
    python -m garmentiq.calibration.capture --mode camera     # tilted photos for the lens
    python -m garmentiq.calibration.bootstrap                 # writes board.json, camera.json

Measuring, per frame::

    board = BoardSpec.load_json("calibration/board.json")
    lens = CameraIntrinsics.load_json("calibration/camera.json")
    frame_cal = calibrate_frame(frame, board, lens, surface_offset_mm=5.0)
    waist_mm = frame_cal.distance(waist_left_px, waist_right_px)
"""
from .board import BoardSpec
from .calibrate import (
    FrameCalibration,
    board_area_px,
    calibrate_frame,
    is_inside_markers,
    measure_pixel_distances,
)
from .camera import CameraCalibrationError, CameraIntrinsics, calibrate_camera
from .config import RigConfig, load_rig
from .detect import (
    DetectionError,
    create_detector,
    detect_in_images,
    detect_markers,
    dictionary_name,
    identify_dictionary,
    load_image,
)
from .homography import HomographyError, estimate_homography, homography_residuals
from .learn_board import BoardLearningError, compare_boards, learn_board
from .metric import (
    distance_mm,
    local_mm_per_px,
    measure_distance,
    measurement_uncertainty_mm,
    mm_to_pixel,
    pixel_to_mm,
)
from .plot import draw_board_layout, draw_markers, draw_mm_grid
from .surface import SurfaceError, expected_scale_error, lift_to_surface, plane_pose
from .undistort import undistort_image, undistort_marker_corners, undistort_points

# capture, inspect_board and bootstrap are command line tools, run with `python -m`.
# They are not imported here: importing a module that is also about to run as __main__
# makes Python execute it twice, and it warns about exactly that.
