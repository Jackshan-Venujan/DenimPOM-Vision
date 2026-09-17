# garmentiq/calibration/verify_plane.py
"""Checking the calibration's accuracy with the chessboard as a known ruler.

After Step B, move the board to other places on the table: the centre, each corner of
the garment area and the edges of the frame. At each place press v. The board's
squares and spans are measured through the saved calibration and compared with their
true size.

Errors are also given **per metre** (error / true length * 1000), because a 1 % scale
error that is 3 mm across the board becomes 12 mm across a 1200 mm trouser. A check
passes when the error per metre is within TOLERANCE_MM.

Run::

    python -m garmentiq.calibration.verify_plane

Keys: v = verify the board where it is now   c = clear outlines   q = quit
"""
import cv2
import numpy as np

from garmentiq.calibration.camera import GREEN, RED, Camera, open_window, put_text
from garmentiq.calibration.chessboard import draw_corners, find_corners
from garmentiq.calibration.config import INNER_CORNERS, SQUARE_MM, TOLERANCE_MM
from garmentiq.calibration.metrology import PlaneMeasurer


def check_board(
    corners_px, measurer, inner_corners=INNER_CORNERS, square_mm=SQUARE_MM
):
    """Measures the board's squares and spans and compares them with the truth.

    Args:
        corners_px (numpy.ndarray): (N, 2) corners from an undistorted frame.
        measurer: Anything with a `to_mm(points)` method, normally a PlaneMeasurer.

    Returns:
        dict: Errors in mm (measured - true). The span errors are the worst row
        (width) or column (height); `error_per_metre_mm` is the worst span error
        scaled to a 1000 mm length.
    """
    columns, rows = inner_corners
    mm = measurer.to_mm(corners_px).reshape(rows, columns, 2)

    steps_x = np.linalg.norm(mm[:, 1:] - mm[:, :-1], axis=-1).ravel()
    steps_y = np.linalg.norm(mm[1:, :] - mm[:-1, :], axis=-1).ravel()
    square_errors = np.concatenate([steps_x, steps_y]) - square_mm

    true_width = (columns - 1) * square_mm
    true_height = (rows - 1) * square_mm
    true_diagonal = float(np.hypot(true_width, true_height))
    width_errors = np.linalg.norm(mm[:, -1] - mm[:, 0], axis=-1) - true_width
    height_errors = np.linalg.norm(mm[-1, :] - mm[0, :], axis=-1) - true_height
    diagonal_error = float(np.linalg.norm(mm[-1, -1] - mm[0, 0])) - true_diagonal

    worst_width = float(width_errors[np.argmax(np.abs(width_errors))])
    worst_height = float(height_errors[np.argmax(np.abs(height_errors))])
    per_metre = 1000 * max(
        abs(worst_width) / true_width,
        abs(worst_height) / true_height,
        abs(diagonal_error) / true_diagonal,
    )

    return {
        "square_mean_abs_mm": float(np.mean(np.abs(square_errors))),
        "square_max_abs_mm": float(np.max(np.abs(square_errors))),
        "width_error_mm": worst_width,
        "height_error_mm": worst_height,
        "diagonal_error_mm": diagonal_error,
        "error_per_metre_mm": float(per_metre),
        "passed": per_metre <= TOLERANCE_MM,
    }


def print_check(number, result):
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"#{number}: squares mean {result['square_mean_abs_mm']:.2f} / "
        f"max {result['square_max_abs_mm']:.2f} mm | "
        f"width {result['width_error_mm']:+.2f}  "
        f"height {result['height_error_mm']:+.2f}  "
        f"diagonal {result['diagonal_error_mm']:+.2f} mm | "
        f"per metre {result['error_per_metre_mm']:.2f} mm -> {verdict}"
    )


def board_outline(corners_px, inner_corners=INNER_CORNERS):
    """The four outer corners, in drawing order, as an int32 polygon."""
    columns = inner_corners[0]
    outline = corners_px[[0, columns - 1, -1, -columns]]
    return outline.round().astype(np.int32)


def main():
    measurer = PlaneMeasurer()
    window = "Verify - move the board, v = verify, c = clear, q = quit"
    open_window(window)

    checks = []  # (outline, result) for every check so far

    with Camera() as camera:
        while True:
            frame = measurer.undistort(camera.read())
            corners = find_corners(frame)

            preview = frame.copy()
            for outline, result in checks:
                color = GREEN if result["passed"] else RED
                cv2.polylines(preview, [outline], True, color, 2)
            if corners is None:
                put_text(preview, "Board NOT found", 0, RED)
            else:
                draw_corners(preview, corners)
                put_text(preview, "Board found - press v to verify here")
            put_text(preview, f"Checks: {len(checks)}", 1)
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                checks = []
            if key == ord("v"):
                if corners is None:
                    print("Board not found - cannot verify.")
                    continue
                result = check_board(corners, measurer)
                checks.append((board_outline(corners), result))
                print_check(len(checks), result)

    cv2.destroyAllWindows()

    if checks:
        worst = max(result["error_per_metre_mm"] for _, result in checks)
        failed = sum(not result["passed"] for _, result in checks)
        print(
            f"\n{len(checks)} checks, {failed} failed. "
            f"Worst: {worst:.2f} mm per metre (tolerance {TOLERANCE_MM} mm)."
        )


if __name__ == "__main__":
    main()
