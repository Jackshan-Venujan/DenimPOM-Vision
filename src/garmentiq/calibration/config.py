# garmentiq/calibration/config.py
"""Holding the handful of rig numbers that cannot be detected from a photo.

Everything about the board itself — which dictionary, which marker ids, where each one
sits, how big the board is — is solved by `learn_board` from photos. Only the values
below have to come from a human, because no photograph contains them:

* a photo has no absolute scale, so one marker must be measured with a caliper;
* fabric thickness is not visible from above.

Fill in every `FILL_THE_DATA` before running the calibration. `load_rig()` refuses to
run while any of them are unfilled, and says which.
"""
from dataclasses import dataclass
from pathlib import Path

# Sentinel for a value that still needs measuring. Nothing accepts it.
FILL_THE_DATA = None


# ----------------------------------------------------------------------------
# Required: measure these on the real rig
# ----------------------------------------------------------------------------

# Side of ONE printed marker in millimetres, including its black border but not the
# white quiet zone around it. Measure it with a caliper (or a steel rule, carefully) to
# 0.1 mm and write the number you actually read, not the number you asked the printer
# for — printers scale by a few tenths of a percent and it is never exactly 50.
#
#   HOW TO MEASURE: caliper across the black square of one marker, corner to corner
#   along an edge. Measure two or three different markers; if they disagree by more
#   than ~0.2 mm, say so, because learn_board will detect that and it changes how the
#   board is solved.
#
#   WHY IT MATTERS: this is the only absolute scale in the whole system. A 1 % error
#   here is a 1 % error in every measurement forever — 10 mm on a 1 m leg.
#
# Example: MARKER_SIZE_MM = 50.1
MARKER_SIZE_MM = 49.0

# How far the top surface of the garment sits above the board, in millimetres. The
# landmarks are detected on top of the fabric, but the homography measures on the
# board, and anything nearer the camera looks bigger.
#
#   HOW TO MEASURE: lay the garment flat as it will be measured and put a steel rule
#   on edge beside it; read the height of the fabric's upper surface. For folded denim
#   this is usually 4-6 mm. Use the thickness at the points you actually measure
#   across, not at the thickest seam.
#
#   WHY IT MATTERS: at a camera height of 1170 mm, 5 mm of fabric is a 0.43 % error,
#   which is 4.3 mm on a 1000 mm leg — larger than the whole tolerance.
#
# Example: SURFACE_OFFSET_MM = 5.0
SURFACE_OFFSET_MM = 3.0


# ----------------------------------------------------------------------------
# Optional: cross-checks and defaults
# ----------------------------------------------------------------------------

# Distance from the lens to the board, in millimetres. Only a sanity check: camera
# calibration recovers this independently, and a big disagreement means something is
# wrong with the calibration rather than with the tape measure.
# Example: CAMERA_TO_BOARD_MM = 1170.0
CAMERA_TO_BOARD_MM = 1850.0

# Predefined ArUco dictionary id, e.g. cv2.aruco.DICT_4X4_50. Leave as None and
# `inspect_board` will identify it from a photo; pin it afterwards so detection is
# fast and cannot be confused by an overlapping dictionary.
ARUCO_DICTIONARY = None

# Capture resolution the rig runs at, as (width, height). Only used to warn when a
# frame arrives at a different size from the one the lens was calibrated at, because
# intrinsics are valid only at their own resolution.
# Example: CAMERA_RESOLUTION = (4606, 3456)
CAMERA_RESOLUTION = None

# Required tolerance for the QC verdict, in millimetres.
TOLERANCE_LIMIT_MM = 3.0

# One-sigma error of a landmark coordinate, in pixels. Feeds the reported tolerance.
LANDMARK_SIGMA_PX = 2.0

# Where calibration photos and results live, relative to the repository root.
ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_DIR = ROOT / "calibration"
BOARD_IMAGE_DIR = CALIBRATION_DIR / "board_views"
CAMERA_IMAGE_DIR = CALIBRATION_DIR / "camera_views"
BOARD_JSON = CALIBRATION_DIR / "board.json"
CAMERA_JSON = CALIBRATION_DIR / "camera.json"


@dataclass(frozen=True)
class RigConfig:
    """The measured rig values, once they have all been filled in."""

    marker_size_mm: float
    surface_offset_mm: float
    camera_to_board_mm: float = None
    dictionary: int = None
    camera_resolution: tuple = None
    tolerance_limit_mm: float = TOLERANCE_LIMIT_MM
    landmark_sigma_px: float = LANDMARK_SIGMA_PX


# What each required field is, for the error message when it is still unfilled.
_REQUIRED = {
    "MARKER_SIZE_MM": "the calipered side of one printed marker, in mm (e.g. 50.1)",
    "SURFACE_OFFSET_MM": "how far the fabric's top surface sits above the board, in mm (e.g. 5.0)",
}


def load_rig():
    """Returns the rig configuration, or explains what is still missing.

    Returns:
        RigConfig: The filled-in values.

    Raises:
        ValueError: If any required value is still `FILL_THE_DATA`, naming each one
            and how to measure it.
    """
    values = globals()
    missing = [name for name in _REQUIRED if values.get(name) is None]
    if missing:
        lines = "\n".join(f"  {name} = {_REQUIRED[name]}" for name in missing)
        raise ValueError(
            f"Fill these in before calibrating, in {Path(__file__).name}:\n{lines}\n"
            f"Each one is commented in that file with how to measure it."
        )
    return RigConfig(
        marker_size_mm=float(MARKER_SIZE_MM),
        surface_offset_mm=float(SURFACE_OFFSET_MM),
        camera_to_board_mm=None if CAMERA_TO_BOARD_MM is None else float(CAMERA_TO_BOARD_MM),
        dictionary=None if ARUCO_DICTIONARY is None else int(ARUCO_DICTIONARY),
        camera_resolution=None if CAMERA_RESOLUTION is None else tuple(CAMERA_RESOLUTION),
    )
