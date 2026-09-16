# garmentiq/calibration/plot.py
"""Drawing what the calibration believes, so a wrong one is visible.

Residuals in millimetres say a calibration is wrong; these overlays say *how*. A
millimetre grid back-projected onto the photo is the quickest check there is: if the
squares are not square and do not lie flat on the board, the geometry is wrong, and no
amount of staring at numbers shows it as fast.
"""
import cv2
import numpy as np

from .metric import mm_to_pixel

GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


def _thickness(image):
    """Line thickness that stays visible on both a 1280 px preview and a 4606 px photo."""
    return max(1, round(max(image.shape[:2]) / 800))


def draw_markers(image, corners_px, color=GREEN, label=True):
    """Outlines every detected marker and marks its first corner.

    The red dot sits on corner 0 (the marker's top-left). If those dots are not all on
    the same side of their markers, some markers were printed or stuck down rotated —
    which is fine for a learned board, but worth knowing.

    Args:
        image (numpy.ndarray): BGR frame.
        corners_px (dict): Marker id -> (4, 2) corners.
        color (tuple): Outline colour, BGR.
        label (bool): Draw the marker id.

    Returns:
        numpy.ndarray: A copy of the frame with the overlay drawn.
    """
    canvas = image.copy()
    thickness = _thickness(canvas)
    for marker_id, corners in corners_px.items():
        corners = np.asarray(corners, dtype=np.float64)
        cv2.polylines(
            canvas, [np.round(corners).astype(np.int32)], True, color, thickness
        )
        first = np.round(corners[0]).astype(int)
        cv2.circle(canvas, tuple(first.tolist()), thickness * 3, RED, -1)
        if label:
            cv2.putText(
                canvas,
                str(marker_id),
                tuple((first - [0, thickness * 6]).tolist()),
                cv2.FONT_HERSHEY_SIMPLEX,
                thickness * 0.6,
                RED,
                thickness,
            )
    return canvas


def draw_mm_grid(image, H, spacing_mm=100.0, extent_mm=None, color=CYAN, label=True):
    """Back-projects a millimetre grid onto the frame.

    Every line is drawn as a chain of short segments, because perspective bends a
    straight metric line into a straight image line only if the geometry is right —
    drawing it segment by segment means a wrong calibration shows up as a visibly
    curved or skewed grid rather than a plausible straight one.

    Args:
        image (numpy.ndarray): BGR frame.
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography.
        spacing_mm (float): Grid pitch in millimetres.
        extent_mm (tuple, optional): ((x_min, x_max), (y_min, y_max)) to cover.
            Defaults to whatever area of the board the frame can see.
        color (tuple): Line colour, BGR.
        label (bool): Write the millimetre value on each axis line.

    Returns:
        numpy.ndarray: A copy of the frame with the grid drawn.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, _thickness(canvas) // 2)

    if extent_mm is None:
        frame_corners = [[0, 0], [width, 0], [width, height], [0, height]]
        visible = cv2.perspectiveTransform(
            np.asarray(frame_corners, dtype=np.float64).reshape(-1, 1, 2),
            np.asarray(H, dtype=np.float64),
        ).reshape(-1, 2)
        extent_mm = (
            (visible[:, 0].min(), visible[:, 0].max()),
            (visible[:, 1].min(), visible[:, 1].max()),
        )

    (x_min, x_max), (y_min, y_max) = extent_mm
    x_lines = np.arange(np.ceil(x_min / spacing_mm), np.floor(x_max / spacing_mm) + 1) * spacing_mm
    y_lines = np.arange(np.ceil(y_min / spacing_mm), np.floor(y_max / spacing_mm) + 1) * spacing_mm

    def polyline(points_mm, text=None):
        points_px = np.round(mm_to_pixel(points_mm, H)).astype(np.int32)
        cv2.polylines(canvas, [points_px], False, color, thickness)
        if text is not None and len(points_px):
            cv2.putText(
                canvas, text, tuple(points_px[0].tolist()), cv2.FONT_HERSHEY_SIMPLEX,
                thickness * 0.5, color, thickness,
            )

    steps = 20
    for x in x_lines:
        ys = np.linspace(y_min, y_max, steps)
        polyline(np.column_stack([np.full(steps, x), ys]), f"{x:.0f}" if label else None)
    for y in y_lines:
        xs = np.linspace(x_min, x_max, steps)
        polyline(np.column_stack([xs, np.full(steps, y)]), f"{y:.0f}" if label else None)
    return canvas


def draw_board_layout(board, pixels_per_mm=1.0, margin_mm=20.0):
    """Renders the learned board layout as a plan view, to eyeball against the real one.

    Args:
        board (BoardSpec): The learned layout.
        pixels_per_mm (float): Rendering scale.
        margin_mm (float): Blank border around the markers.

    Returns:
        numpy.ndarray: A BGR image of the board as the calibration understands it.
    """
    points = np.vstack(list(board.corners_mm.values()))
    low = points.min(axis=0) - margin_mm
    high = points.max(axis=0) + margin_mm
    size = np.ceil((high - low) * pixels_per_mm).astype(int)
    canvas = np.full((int(size[1]), int(size[0]), 3), 255, dtype=np.uint8)

    for marker_id, corners in sorted(board.corners_mm.items()):
        pixels = np.round((corners - low) * pixels_per_mm).astype(np.int32)
        cv2.fillPoly(canvas, [pixels], (30, 30, 30))
        cv2.circle(canvas, tuple(pixels[0].tolist()), 3, RED, -1)
        centre = pixels.mean(axis=0).astype(int)
        cv2.putText(
            canvas, str(marker_id), tuple((centre + [6, 0]).tolist()),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1,
        )

    width, height = high - low
    cv2.putText(
        canvas, f"{width:.0f} x {height:.0f} mm, {len(board.corners_mm)} markers",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
    )
    return canvas
