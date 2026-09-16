# garmentiq/calibration/capture.py
"""Capturing calibration photos from the rig, with live feedback on what is missing.

A calibration is only as good as the photos it was given, and the two jobs want
opposite things:

* **board** views feed `learn_board`. Every marker must be seen from several angles,
  and pairs of markers must share frames so the layout can be chained together. The
  live view tracks how many angles each marker has been seen from.
* **camera** views feed `calibrate_camera`. The markers must reach the corners and
  edges of the frame, where distortion is strongest, and enough views must be tilted
  or the focal length stays ambiguous. The live view tracks frame coverage and tilt.

Both are captured the same way, so this is one script with a `--mode`. Shooting blind
and finding out later that the corners were never covered is the failure this avoids.

Press `s` to save each frame yourself, or pass `--auto N` to capture N frames on a
timer while you reposition the board. Automatic capture refuses frames that are blurred
or show too few markers and simply tries again, so an unattended run cannot quietly
fill up with unusable photos.

    python -m garmentiq.calibration.capture --mode board
    python -m garmentiq.calibration.capture --mode board --auto 20 --interval 5
    python -m garmentiq.calibration.capture --mode camera --auto 30
"""
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import ARUCO_DICTIONARY, BOARD_IMAGE_DIR, CAMERA_IMAGE_DIR
from .detect import create_detector, detect_markers, identify_dictionary
from .plot import draw_markers

WINDOW_NAME = "GarmentIQ - calibration capture"

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# Frame-coverage grid for camera mode: how many cells have ever held a marker corner.
COVERAGE_GRID = (8, 6)

# Targets the live view counts down to. They mirror the thresholds the calibration
# modules warn about, so a capture session that satisfies these will not be scolded.
TARGETS = {
    "board": {"views": 20, "angles_per_marker": 4},
    "camera": {"views": 30, "coverage": 0.7},
}


class LatestFrameReader:
    """Reads a camera stream in a background thread, keeping only the newest frame.

    A network stream queues frames up. Detection is slower than the camera, so reading
    in order would fall further and further behind; keeping only the newest keeps the
    preview live.
    """

    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {source}\n"
                f"Check the phone app is streaming, and that no browser tab, VLC or "
                f"DroidCam client is already connected to it."
            )
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._keep_reading, daemon=True)
        self.thread.start()

    def _keep_reading(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        """Returns a copy of the newest frame (BGR), or None if none has arrived."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


def put_text(image, text, position, color):
    """Draws text with a dark outline so it stays readable on any background."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class CaptureProgress:
    """Tracks what a capture session still needs, and says it in one line."""

    def __init__(self, mode, image_size):
        self.mode = mode
        self.image_size = image_size
        self.n_saved = 0
        self.marker_views = {}
        self.marker_angles = {}
        self.covered_cells = set()

    def record(self, corners_px):
        """Folds one saved frame into the running totals."""
        self.n_saved += 1
        width, height = self.image_size
        for marker_id, corners in corners_px.items():
            corners = np.asarray(corners, dtype=np.float64)
            self.marker_views[marker_id] = self.marker_views.get(marker_id, 0) + 1
            # A marker's apparent aspect ratio changes with viewing angle, so binning
            # it is a cheap stand-in for "how many different angles have I seen this
            # from" without solving a pose per frame.
            sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
            shape = round(float(sides[0] / max(sides[1], 1e-6)), 1)
            self.marker_angles.setdefault(marker_id, set()).add(shape)

            columns, rows = COVERAGE_GRID
            for x, y in corners:
                cell = (
                    int(np.clip(x / width * columns, 0, columns - 1)),
                    int(np.clip(y / height * rows, 0, rows - 1)),
                )
                self.covered_cells.add(cell)

    @property
    def coverage(self):
        """Share of frame cells that have held a marker corner in some saved frame."""
        return len(self.covered_cells) / (COVERAGE_GRID[0] * COVERAGE_GRID[1])

    def status(self):
        """One line saying what is still missing."""
        target = TARGETS[self.mode]
        if self.mode == "board":
            thin = sorted(
                marker_id
                for marker_id, angles in self.marker_angles.items()
                if len(angles) < target["angles_per_marker"]
            )
            need = f"few angles on {thin}" if thin else "all markers well covered"
            return f"saved {self.n_saved}/{target['views']}   {need}"
        return (
            f"saved {self.n_saved}/{target['views']}   "
            f"frame coverage {self.coverage:.0%} (want {target['coverage']:.0%})"
        )

    def is_complete(self):
        target = TARGETS[self.mode]
        if self.n_saved < target["views"]:
            return False
        if self.mode == "camera":
            return self.coverage >= target["coverage"]
        return all(
            len(angles) >= target["angles_per_marker"]
            for angles in self.marker_angles.values()
        )


def resolve_source(source):
    """Turns a camera argument into what cv2.VideoCapture wants.

    A bare number is a local webcam index; anything else is a URL or file path.
    """
    text = str(source)
    return int(text) if text.isdigit() else text


def capture_session(source, mode, output_dir, dictionary=None, frame_width=1280):
    """Runs the live capture window until you quit.

    Args:
        source (str | int): Camera URL, device index, or video file.
        mode (str): `"board"` or `"camera"`, which changes what progress is tracked.
        output_dir (str | pathlib.Path): Where saved frames go.
        dictionary (int, optional): ArUco dictionary. Identified from the first frame
            that contains markers when None.
        frame_width (int): Preview width; frames are saved at full resolution.

    Returns:
        CaptureProgress: What was captured.

    Raises:
        RuntimeError: If the stream cannot be opened or delivers no frames.
    """
    if mode not in TARGETS:
        raise ValueError(f"Mode must be 'board' or 'camera', got {mode!r}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = LatestFrameReader(resolve_source(source))
    detector, progress = None, None
    try:
        frame = _wait_for_first_frame(reader)
        height, width = frame.shape[:2]
        progress = CaptureProgress(mode, (width, height))
        print(f"Capturing {mode} views at {width}x{height} into {output_dir}")
        print("Keys:  s = save frame    q = quit")

        while True:
            frame = reader.read()
            if frame is None:
                continue

            if detector is None:
                dictionary = dictionary or _identify(frame)
                if dictionary is not None:
                    detector = create_detector(dictionary)

            corners = detect_markers(frame, detector=detector) if detector else {}
            preview = _make_preview(frame, corners, progress, dictionary, frame_width)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not corners:
                    print("No markers in view; not saving.")
                    continue
                name = datetime.now().strftime(f"{mode}_%Y%m%d_%H%M%S_%f")[:-3] + ".png"
                cv2.imwrite(str(output_dir / name), frame)
                progress.record(corners)
                print(f"Saved {name}   {progress.status()}")
                if progress.is_complete():
                    print("This set now meets the targets; more views still help.")
    finally:
        reader.release()
        cv2.destroyAllWindows()
    return progress


def sharpness(image):
    """How sharp a frame is: the variance of its Laplacian.

    A blurred photo still detects markers, but its corners land in the wrong place and
    quietly poison the calibration, so blurred frames are refused rather than saved.
    The number has no absolute meaning — it depends on resolution and on how much
    detail the scene has — so it is reported, not just tested.
    """
    return float(cv2.Laplacian(as_gray_frame(image), cv2.CV_64F).var())


def as_gray_frame(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def auto_capture_session(
    source,
    mode,
    output_dir,
    count=20,
    interval=4.0,
    dictionary=None,
    min_markers=2,
    min_sharpness=40.0,
    headless=True,
    settle_seconds=1.0,
):
    """Captures a set unattended, on a timer, while you reposition the board.

    Nobody has to watch a window or press a key: it counts down out loud, grabs a
    frame, checks it is worth keeping, and says what it still needs. That makes the
    capture runnable from a terminal — including by someone who cannot see the preview.

    A frame is only kept when enough markers are detected and the frame is sharp
    enough. A rejected frame is retried on the next tick rather than counted, so a
    hand crossing the board or a moment of motion blur costs a few seconds, not a
    ruined calibration.

    Args:
        source (str | int): Camera URL, device index, or video file.
        mode (str): `"board"` or `"camera"`.
        output_dir (str | pathlib.Path): Where saved frames go.
        count (int): How many good frames to keep.
        interval (float): Seconds between attempts — your time to move the board.
        dictionary (int, optional): ArUco dictionary; identified from the stream if None.
        min_markers (int): Fewest markers a frame must show to be kept.
        min_sharpness (float): Laplacian variance below which a frame counts as blurred.
        headless (bool): Skip the preview window. Leave it on when running over a
            terminal; turn it off to watch what the camera sees.
        settle_seconds (float): Pause after the countdown before grabbing, so the board
            has stopped moving and the phone has re-focused.

    Returns:
        CaptureProgress: What was captured.

    Raises:
        RuntimeError: If the stream cannot be opened or delivers no frames.
        ValueError: If `mode` is not one of the two.
    """
    if mode not in TARGETS:
        raise ValueError(f"Mode must be 'board' or 'camera', got {mode!r}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = LatestFrameReader(resolve_source(source))
    detector = None
    try:
        frame = _wait_for_first_frame(reader)
        height, width = frame.shape[:2]
        progress = CaptureProgress(mode, (width, height))

        print(f"Capturing {count} {mode} views at {width}x{height} into {output_dir}")
        print(_advice(mode))
        print(f"One frame every {interval:.0f} s. Move the board between beeps.\n")

        attempts, rejected = 0, 0
        while progress.n_saved < count and attempts < count * 4:
            attempts += 1
            for remaining in range(int(interval), 0, -1):
                print(f"\r  next in {remaining} s ... ", end="", flush=True)
                time.sleep(1.0)
            time.sleep(settle_seconds)

            frame = reader.read()
            if frame is None:
                print("\r  no frame from the camera; retrying.        ")
                continue

            if detector is None:
                dictionary = dictionary or _identify(frame)
                if dictionary is None:
                    print("\r  no markers found yet; is the board in view?   ")
                    continue
                detector = create_detector(dictionary)

            corners = detect_markers(frame, detector=detector)
            focus = sharpness(frame)
            if len(corners) < min_markers:
                rejected += 1
                print(f"\r  SKIP: only {len(corners)} markers in view.            ")
                continue
            if focus < min_sharpness:
                rejected += 1
                print(f"\r  SKIP: blurred (sharpness {focus:.0f} < {min_sharpness:.0f}).  ")
                continue

            name = datetime.now().strftime(f"{mode}_%Y%m%d_%H%M%S_%f")[:-3] + ".png"
            cv2.imwrite(str(output_dir / name), frame)
            progress.record(corners)
            print(
                f"\r  [{progress.n_saved}/{count}] {name}  "
                f"{len(corners)} markers, sharpness {focus:.0f}\n"
                f"      {progress.status()}"
            )

            if not headless:
                cv2.imshow(WINDOW_NAME, _make_preview(frame, corners, progress,
                                                      dictionary, 1280))
                cv2.waitKey(1)

        print(f"\nKept {progress.n_saved} frames, rejected {rejected}.")
        if progress.n_saved < count:
            print(
                f"Stopped after {attempts} attempts without reaching {count}. "
                f"Check the board is lit, in focus and in frame."
            )
    finally:
        reader.release()
        cv2.destroyAllWindows()
    return progress


def _advice(mode):
    """What to do with the board between shots, which differs entirely by mode."""
    if mode == "board":
        return (
            "BOARD views teach it your board's geometry. Keep the WHOLE board in frame\n"
            "and no garment on it. Between shots, change the angle and distance — every\n"
            "marker needs to be seen from several directions, and shots must overlap."
        )
    return (
        "CAMERA views teach it your lens. TILT the board 20-40 degrees in most shots,\n"
        "and move it into the CORNERS and EDGES of the frame — that is where distortion\n"
        "lives and it cannot be measured where the board never went."
    )


def _wait_for_first_frame(reader, timeout_seconds=10):
    start = time.time()
    frame = reader.read()
    while frame is None:
        if time.time() - start > timeout_seconds:
            raise RuntimeError("No frames received. Check the camera source.")
        time.sleep(0.05)
        frame = reader.read()
    return frame


def _identify(frame):
    matches = identify_dictionary(frame)
    if not matches:
        return None
    print(f"Dictionary identified as {matches[0]['name']}")
    return matches[0]["dictionary"]


def _make_preview(frame, corners, progress, dictionary, frame_width):
    """Builds the on-screen view: detections, progress and what is still missing."""
    preview = draw_markers(frame, corners, label=False) if corners else frame.copy()

    height, width = preview.shape[:2]
    if width > frame_width:
        preview = cv2.resize(preview, (frame_width, int(height * frame_width / width)))

    if dictionary is None:
        put_text(preview, "Looking for markers ...", (10, 30), RED)
    else:
        colour = GREEN if corners else RED
        put_text(preview, f"{len(corners)} markers in view", (10, 30), colour)
    put_text(preview, progress.status(), (10, 55), YELLOW)
    put_text(preview, "s = save    q = quit", (10, preview.shape[0] - 15), YELLOW)
    return preview


def main(argv=None):
    """Command line entry point: `python -m garmentiq.calibration.capture`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("board", "camera"), required=True)
    parser.add_argument("--source", default="http://192.168.8.170:4747/video",
                        help="camera URL, device index, or video file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to the directory for this mode in config.py")
    parser.add_argument("--dictionary", type=int, default=ARUCO_DICTIONARY)
    parser.add_argument("--auto", type=int, default=None, metavar="N",
                        help="capture N frames on a timer instead of on keypresses")
    parser.add_argument("--interval", type=float, default=4.0,
                        help="seconds between automatic captures (default 4)")
    parser.add_argument("--min-sharpness", type=float, default=40.0,
                        help="reject frames blurrier than this (Laplacian variance)")
    parser.add_argument("--show", action="store_true",
                        help="show the preview window during automatic capture")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (
        BOARD_IMAGE_DIR if args.mode == "board" else CAMERA_IMAGE_DIR
    )
    if args.auto:
        progress = auto_capture_session(
            args.source, args.mode, output_dir,
            count=args.auto, interval=args.interval, dictionary=args.dictionary,
            min_sharpness=args.min_sharpness, headless=not args.show,
        )
    else:
        progress = capture_session(args.source, args.mode, output_dir, args.dictionary)

    print(f"\nFinished: {progress.status()}")
    print("Next: python -m garmentiq.calibration.bootstrap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
