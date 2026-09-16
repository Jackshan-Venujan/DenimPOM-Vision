# garmentiq/calibration/camera.py
"""Reading the DroidCam stream, and the small window helpers the live scripts share.

A network stream queues frames. If a script is slower than the camera, reading frames
in order makes the video fall further and further behind. `Camera` reads in a
background thread and keeps only the newest frame, so the view stays live.

Run on its own to check the stream and its resolution::

    python -m garmentiq.calibration.camera
"""
import threading
import time

import cv2

from garmentiq.calibration.config import CAMERA_URL, DISPLAY_SCALE, FRAME_SIZE

# Colours are BGR, because OpenCV draws in BGR.
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


class Camera:
    """The newest frame from a video stream.

    Use it as a context manager so the stream is always released::

        with Camera() as camera:
            frame = camera.read()
    """

    def __init__(self, url=CAMERA_URL):
        self.capture = cv2.VideoCapture(url)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera stream: {url}\n"
                "Check that DroidCam is streaming, the PC is on the same Wi-Fi, and "
                "no browser tab or other script is already connected to it."
            )

        self._frame = None
        self._frame_id = 0  # increases with every frame the thread receives
        self._last_read_id = 0  # the frame `read` returned last
        self._new_frame = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._keep_reading, daemon=True)
        self._thread.start()

    def _keep_reading(self):
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)  # stream hiccup: wait instead of spinning the CPU
                continue
            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._new_frame.notify_all()

    def read(self, timeout_s=5.0):
        """Returns the newest frame (BGR), waiting until one arrives that has not been
        returned before. So consecutive calls never return the same frame twice.

        Raises:
            RuntimeError: If no new frame arrives within `timeout_s`.
        """
        with self._new_frame:
            has_new = self._new_frame.wait_for(
                lambda: self._frame_id != self._last_read_id, timeout=timeout_s
            )
            if not has_new:
                raise RuntimeError(
                    f"No frame from the camera for {timeout_s} s. Is DroidCam running?"
                )
            self._last_read_id = self._frame_id
            return self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def check_frame_size(frame, expected_size=FRAME_SIZE, what="config.FRAME_SIZE"):
    """Raises ValueError if the frame is not `expected_size` (width, height).

    Every calibration is tied to one resolution, so a mismatch is always an error.
    """
    height, width = frame.shape[:2]
    if (width, height) != tuple(expected_size):
        raise ValueError(
            f"Frame is {width}x{height} but {what} is "
            f"{expected_size[0]}x{expected_size[1]}. Set the same resolution in the "
            "DroidCam app, or calibrate again at the new one."
        )


def open_window(name):
    """Creates a resizable window DISPLAY_SCALE times the frame size."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, FRAME_SIZE[0] * DISPLAY_SCALE, FRAME_SIZE[1] * DISPLAY_SCALE)


def put_text(image, text, line=0, color=GREEN):
    """Writes one line of text in the top-left corner. `line` 0 is the top line."""
    position = (10, 22 + 22 * line)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def main():
    window = "Camera check - q to quit"
    open_window(window)

    with Camera() as camera:
        frame = camera.read()
        height, width = frame.shape[:2]
        print(f"Stream resolution: {width}x{height}")
        size_ok = (width, height) == FRAME_SIZE
        if not size_ok:
            print(f"WARNING: config.FRAME_SIZE is {FRAME_SIZE[0]}x{FRAME_SIZE[1]}.")

        frame_count, start = 0, time.time()
        while True:
            frame = camera.read()
            frame_count += 1
            fps = frame_count / max(time.time() - start, 1e-6)

            put_text(frame, f"{width}x{height}  {fps:.1f} fps")
            if not size_ok:
                put_text(frame, "Resolution differs from config.FRAME_SIZE", 1, RED)
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
