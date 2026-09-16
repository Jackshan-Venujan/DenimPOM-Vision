# garmentiq/calibration/config.py
"""Every setting the calibration scripts share, in one place.

Change a value here and every script picks it up. Nothing else in the package holds a
number that describes the rig.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
# DroidCam stream. DroidCam allows one client at a time, so close any browser tab or
# other script that is showing it before running a calibration script.
CAMERA_URL = "http://192.168.8.170:4747/video"

# (width, height) the stream delivers. The resolution is chosen in the DroidCam app;
# cv2 cannot change it on an HTTP stream. Every calibration is valid only at this size.
FRAME_SIZE = (640, 480)

# The frame is small, so windows are shown this many times larger. Mouse clicks are
# still reported in frame pixels.
DISPLAY_SCALE = 1

# ----------------------------------------------------------------------------
# Chessboard: Checkerboard-A3-40mm-9x6
# ----------------------------------------------------------------------------
# OpenCV counts INNER corners (where four squares meet), not squares. This print has
# 10 x 7 squares, so 9 x 6 inner corners. If the board is never found, count the
# squares on the print and fix this first: inner corners = squares - 1 each way.
INNER_CORNERS = (9, 6)  # (columns, rows)

# Side of one square in millimetres. Measure the print with a steel rule across as
# many squares as possible and divide: printers scale pages by 1-3 %, and that error
# goes straight into every measurement. Example: 8 squares measure 319.2 mm -> 39.9.
SQUARE_MM = 40.0

# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration"
LENS_IMAGE_DIR = OUTPUT_DIR / "lens_images"  # Step A photos
INTRINSICS_PATH = OUTPUT_DIR / "camera_intrinsics.npz"  # Step A result
HOMOGRAPHY_PATH = OUTPUT_DIR / "table_homography.npz"  # Step B result

# ----------------------------------------------------------------------------
# Quality limits (each script reports PASS/FAIL against these)
# ----------------------------------------------------------------------------
LENS_MIN_IMAGES = 15  # fewer views than this calibrates the lens poorly
LENS_MAX_RMS_PX = 0.5  # lens reprojection error
PLANE_AVERAGE_FRAMES = 10  # frames averaged before fitting the table homography
PLANE_MAX_ERROR_MM = 0.5  # homography fit error on the board corners
TOLERANCE_MM = 3.0  # allowed measurement error on a garment
