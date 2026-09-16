# garmentiq/calibration/chessboard.py
"""Finding the chessboard in an image, and where its corners really are in mm.

The two functions here are the two halves of every calibration:

* `find_corners` gives the inner corners in **pixels**;
* `board_points_mm` gives the same corners in **millimetres**, in the same order.

Pairing them row by row is what turns pixels into millimetres.

Run on its own to check the board is detected live::

    python -m garmentiq.calibration.chessboard
"""
import cv2
import numpy as np

from garmentiq.calibration.camera import RED, Camera, open_window, put_text
from garmentiq.calibration.config import INNER_CORNERS, SQUARE_MM


def board_points_mm(inner_corners=INNER_CORNERS, square_mm=SQUARE_MM):
    """Positions of the inner corners on the board, in mm.

    The first corner is (0, 0). Points go left to right along each row, then row by
    row, which is the order `find_corners` returns them in.

    Returns:
        numpy.ndarray: (N, 2) float32 array of (x, y) in mm, N = columns * rows.
    """
    columns, rows = inner_corners
    xs, ys = np.meshgrid(np.arange(columns), np.arange(rows))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return (points * square_mm).astype(np.float32)


def find_corners(image, inner_corners=INNER_CORNERS):
    """Finds the chessboard's inner corners with sub-pixel accuracy.

    Args:
        image (numpy.ndarray): BGR or grayscale image.
        inner_corners (tuple): (columns, rows) of inner corners.

    Returns:
        numpy.ndarray or None: (N, 2) float32 pixel positions, or None if the whole
        board was not found.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, inner_corners, flags=flags)
    if not found:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def draw_corners(image, corners, inner_corners=INNER_CORNERS):
    """Draws the detected corners on `image` in place (coloured rows, OpenCV style)."""
    cv2.drawChessboardCorners(image, inner_corners, corners.reshape(-1, 1, 2), True)


def main():
    window = "Chessboard check - q to quit"
    open_window(window)
    expected = INNER_CORNERS[0] * INNER_CORNERS[1]

    with Camera() as camera:
        while True:
            frame = camera.read()
            corners = find_corners(frame)

            if corners is None:
                put_text(frame, "Board NOT found", 0, RED)
                put_text(frame, f"Looking for {INNER_CORNERS} inner corners", 1, RED)
            else:
                draw_corners(frame, corners)
                put_text(frame, f"Board found: {len(corners)}/{expected} corners")
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
