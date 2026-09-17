import threading
import time
import typing as tp
from collections.abc import Callable
from dataclasses import dataclass

import cv2


@dataclass
class Frame:
    data: tp.Any
    timestamp: float
    frame_id: int


FrameCallback = Callable[[Frame], None]


class FrameGrabber:
    """
    Push-based grabber: mimics a camera SDK's event model.
    The registered callback is invoked directly from the
    acquisition thread.
    """

    def __init__(self, source, on_frame: FrameCallback, target_fps: float | None = None):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        self.on_frame = on_frame
        self.target_fps = target_fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._stop = threading.Event()
        self._frame_id = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        period = 1.0 / self.target_fps
        next_tick = time.monotonic()

        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break

            f = Frame(data=frame, timestamp=time.monotonic(), frame_id=self._frame_id)
            self._frame_id += 1

            self.on_frame(f)

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

        self._stop.set()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class AsyncDispatcher:
    def __init__(self, handler: FrameCallback, drop_policy="drop_oldest"):
        self.handler = handler
        self.drop_policy = drop_policy
        self._slot: Frame | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def __call__(self, frame: Frame):
        with self._lock:
            if self._slot is not None and self.drop_policy == "drop_newest":
                return
            self._slot = frame
        self._new_frame.set()

    def _loop(self):
        while not self._stop.is_set():
            if self._new_frame.wait(timeout=0.1):
                self._new_frame.clear()
                with self._lock:
                    frame, self._slot = self._slot, None
                if frame is not None:
                    self.handler(frame)

    def stop(self):
        self._stop.set()
        self._worker.join(timeout=2.0)

    def clear(self):
        with self._lock:
            self._slot = None
        self._new_frame.clear()
