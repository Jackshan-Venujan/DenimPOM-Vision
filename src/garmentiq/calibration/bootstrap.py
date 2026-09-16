# garmentiq/calibration/bootstrap.py
"""Breaking the circle between lens calibration and board learning.

Camera calibration needs a board whose geometry is known, and learning the board's
geometry wants the lens distortion removed first. Neither can go first, so both are
solved by iteration:

1. Learn the board with no lens correction. The layout absorbs some distortion and is
   slightly wrong, but it is close enough to calibrate against.
2. Calibrate the lens using that layout.
3. Re-learn the board with the distortion removed. This layout is the real one.
4. Re-calibrate the lens against the corrected layout.

Two passes are enough in practice, and `compare_boards` says by how much the layout
moved between them: if pass two barely changed anything, distortion was not hurting
this rig and you know it rather than assuming it.

Run it as `python -m garmentiq.calibration.bootstrap`.
"""
import argparse
from pathlib import Path

from .camera import calibrate_camera
from .config import (
    BOARD_IMAGE_DIR,
    BOARD_JSON,
    CAMERA_IMAGE_DIR,
    CAMERA_JSON,
    load_rig,
)
from .detect import detect_in_images, dictionary_name, identify_dictionary, load_image
from .learn_board import compare_boards, learn_board

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_images(directory):
    """Lists the image files in a directory, sorted by name.

    Raises:
        FileNotFoundError: If the directory is missing or holds no images.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No such directory: {directory}. Capture photos first with "
            f"python -m garmentiq.calibration.capture"
        )
    images = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images in {directory}.")
    return images


def resolve_dictionary(image_path, dictionary=None):
    """Returns the board's ArUco dictionary, identifying it from a photo if needed.

    Raises:
        ValueError: If no dictionary matches the photo.
    """
    if dictionary is not None:
        return int(dictionary)
    matches = identify_dictionary(load_image(image_path))
    if not matches:
        raise ValueError(
            f"No ArUco dictionary matched {Path(image_path).name}. Check the photo is "
            f"sharp, in focus and shows the markers unobstructed."
        )
    best = matches[0]
    print(
        f"Dictionary identified as {best['name']} "
        f"({best['n_markers']} markers: {best['marker_ids']})"
    )
    if len(matches) > 1:
        others = ", ".join(f"{m['name']} ({m['n_markers']})" for m in matches[1:])
        print(f"  Other dictionaries also matched: {others}. Pin the right one in config.py.")
    return best["dictionary"]


def bootstrap_calibration(
    board_images,
    camera_images,
    marker_size_mm,
    dictionary=None,
    passes=2,
    board_json=None,
    camera_json=None,
    verbose=True,
):
    """Solves the board layout and the lens together, then writes both to JSON.

    Args:
        board_images (list): Photos of the bare board, from varied angles.
        camera_images (list): Photos for lens calibration, well spread across the
            frame and tilted. May be the same list as `board_images`.
        marker_size_mm (float): Calipered side of one marker.
        dictionary (int, optional): ArUco dictionary; identified from a photo if None.
        passes (int): Learn/calibrate rounds. Two is enough; one skips the correction.
        board_json (pathlib.Path, optional): Where to write the board.
        camera_json (pathlib.Path, optional): Where to write the intrinsics.
        verbose (bool): Print progress and diagnostics.

    Returns:
        tuple: `(BoardSpec, CameraIntrinsics, history)` where `history` lists how far
        the board moved on each pass after the first.

    Raises:
        BoardLearningError: If the board layout cannot be solved.
        CameraCalibrationError: If the lens cannot be calibrated.
    """
    board_images = list(board_images)
    camera_images = list(camera_images)
    dictionary = resolve_dictionary(board_images[0], dictionary)

    def say(*message):
        if verbose:
            print(*message)

    say(f"Detecting markers in {len(board_images)} board photos ...")
    board_views = detect_in_images(board_images, dictionary)
    say(f"Detecting markers in {len(camera_images)} camera photos ...")
    camera_views = (
        board_views
        if camera_images == board_images
        else detect_in_images(camera_images, dictionary)
    )

    board, intrinsics, history = None, None, []
    for index in range(int(passes)):
        say(f"\n--- pass {index + 1} of {passes} ---")
        previous = board
        board = learn_board(
            board_views, marker_size_mm, dictionary, intrinsics=intrinsics
        )
        say(f"Board: {board}")
        say(f"  worst marker scatter {board.diagnostics['max_marker_scatter_mm']:.3f} mm, "
            f"worst view {board.diagnostics['max_view_rms_px']:.3f} px")
        if previous is not None:
            moved = compare_boards(previous, board)
            history.append(moved)
            say(f"  layout moved {moved['rms_mm']:.3f} mm RMS "
                f"({moved['max_mm']:.3f} mm worst) from the previous pass")

        intrinsics = calibrate_camera(camera_views, board)
        say(f"Camera: {intrinsics}")
        say(f"  coverage {intrinsics.diagnostics['coverage_fraction']:.0%}, "
            f"board distance {intrinsics.diagnostics['board_distance_mm']:.0f} mm")

    for warning in board.diagnostics["warnings"]:
        say(f"BOARD WARNING: {warning}")
    for warning in intrinsics.diagnostics["warnings"]:
        say(f"CAMERA WARNING: {warning}")

    if board_json is not None:
        say(f"\nWrote {board.save_json(board_json)}")
    if camera_json is not None:
        say(f"Wrote {intrinsics.save_json(camera_json)}")
    return board, intrinsics, history


def main(argv=None):
    """Command line entry point: `python -m garmentiq.calibration.bootstrap`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--board-dir", type=Path, default=BOARD_IMAGE_DIR,
                        help="photos of the bare board")
    parser.add_argument("--camera-dir", type=Path, default=CAMERA_IMAGE_DIR,
                        help="photos for lens calibration")
    parser.add_argument("--board-json", type=Path, default=BOARD_JSON)
    parser.add_argument("--camera-json", type=Path, default=CAMERA_JSON)
    parser.add_argument("--marker-size-mm", type=float, default=None,
                        help="overrides MARKER_SIZE_MM from config.py")
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args(argv)

    marker_size_mm = args.marker_size_mm
    dictionary = None
    if marker_size_mm is None:
        rig = load_rig()  # raises with instructions if config.py is unfilled
        marker_size_mm, dictionary = rig.marker_size_mm, rig.dictionary

    board_images = find_images(args.board_dir)
    camera_images = (
        board_images if args.camera_dir == args.board_dir else find_images(args.camera_dir)
    )
    board, intrinsics, _ = bootstrap_calibration(
        board_images,
        camera_images,
        marker_size_mm,
        dictionary=dictionary,
        passes=args.passes,
        board_json=args.board_json,
        camera_json=args.camera_json,
    )
    print(f"\nDictionary: {dictionary_name(board.dictionary)}")
    print(f"Board markers: {board.marker_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
