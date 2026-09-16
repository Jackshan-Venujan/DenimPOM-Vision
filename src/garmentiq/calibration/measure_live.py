# garmentiq/calibration/measure_live.py
"""Measuring by clicking two points on the live, undistorted view.

Use it to sanity-check the calibration with a steel ruler or tape laid on the table.

Run::

    python -m garmentiq.calibration.measure_live

Mouse: left click = set point (two clicks make one measurement)
Keys : space = freeze / unfreeze the frame   c = clear   q = quit
"""
import cv2
import numpy as np

from garmentiq.calibration.camera import GREEN, YELLOW, Camera, open_window, put_text
from garmentiq.calibration.metrology import PlaneMeasurer


class ClickState:
    """What the mouse has done so far."""

    def __init__(self, measurer):
        self.measurer = measurer
        self.points = []  # clicked points, in frame pixels
        self.cursor = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) == 2:
            self.points = []  # a third click starts a new measurement
        self.points.append((x, y))
        print(f"Clicked pixel ({x}, {y})")
        if len(self.points) == 2:
            length_mm = self.measurer.distance_mm(*self.points)
            print(f"  -> distance {length_mm:.1f} mm")


def draw_measurement(image, point_1, point_2, length_mm, color):
    cv2.line(image, point_1, point_2, color, 1)
    for point in (point_1, point_2):
        cv2.circle(image, point, 3, color, -1)
    label = f"{length_mm:.1f} mm"
    middle = ((point_1[0] + point_2[0]) // 2, (point_1[1] + point_2[1]) // 2 - 8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, middle, font, 0.5, (0, 0, 0), 3)
    cv2.putText(image, label, middle, font, 0.5, color, 1)


def main():
    measurer = PlaneMeasurer()
    clicks = ClickState(measurer)
    window = "Measure - click two points, space = freeze, c = clear, q = quit"
    open_window(window)
    cv2.setMouseCallback(window, clicks.on_mouse)

    frozen = None  # the frozen frame, or None while live

    with Camera() as camera:
        while True:
            frame = measurer.undistort(camera.read()) if frozen is None else frozen
            preview = frame.copy()

            if len(clicks.points) == 1:  # rubber band from first point to cursor
                start = clicks.points[0]
                length = measurer.distance_mm(start, clicks.cursor)
                draw_measurement(preview, start, clicks.cursor, length, YELLOW)
            elif len(clicks.points) == 2:
                length = measurer.distance_mm(*clicks.points)
                draw_measurement(preview, *clicks.points, length, GREEN)

            put_text(preview, "FROZEN" if frozen is not None else "LIVE")
            scale = measurer.mm_per_pixel(clicks.cursor)
            put_text(preview, f"{scale:.2f} mm per pixel at cursor", 1)
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                clicks.points = []
            if key == ord(" "):
                frozen = frame if frozen is None else None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
