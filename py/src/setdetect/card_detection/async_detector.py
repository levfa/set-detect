import multiprocessing as mp
import multiprocessing.process as mpp
import multiprocessing.shared_memory as mpsh
import multiprocessing.synchronize as mps
import queue
import time
import typing as tp
from dataclasses import dataclass

import numpy as np

import setdetect.card_detection.detector as det


@dataclass
class _InItemMeta:
    frame_id: int
    epoch: int
    shm_name: str
    shape: tuple[int, ...]
    dtype: str


def _unlink_shm(name: str) -> None:
    """Unlink a shared-memory block, ignoring ones that are already gone."""
    try:
        shm = mpsh.SharedMemory(name=name)
    except FileNotFoundError:
        return
    shm.close()
    shm.unlink()


@dataclass
class AsyncDetection:
    frame_id: int
    epoch: int
    detection: det.Detection


def _write_to_shm(shm: mpsh.SharedMemory, img: np.ndarray) -> None:
    buf = shm.buf
    assert buf is not None  # shm is open for the lifetime of the helper
    view = np.frombuffer(buf, dtype=img.dtype).reshape(img.shape)
    np.copyto(view, img)


def _detect_from_shm(
    shm: mpsh.SharedMemory,
    in_meta: _InItemMeta,
    card_detector: det.CardDetector,
) -> det.Detection:
    buf = shm.buf
    assert buf is not None  # shm is open for the lifetime of the helper
    img = np.frombuffer(buf, dtype=np.dtype(in_meta.dtype)).reshape(in_meta.shape)
    return card_detector.detect(img)


def _detection_worker(
    in_queue: "mp.Queue[_InItemMeta]",
    out_queue: "mp.Queue[AsyncDetection]",
    start_up_event: mps.Event,
    pause_event: mps.Event,
    terminate_event: mps.Event,
    detection_kwargs: dict[str, tp.Any],
) -> None:
    card_detector = det.CardDetector(**detection_kwargs)
    start_up_event.set()

    while not terminate_event.is_set():
        if pause_event.is_set():
            time.sleep(0.01)
            continue

        try:
            in_meta = in_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            shm = mpsh.SharedMemory(name=in_meta.shm_name)
        except FileNotFoundError:
            continue
        try:
            detection = _detect_from_shm(shm, in_meta, card_detector)
        finally:
            shm.close()
            shm.unlink()
        out_queue.put(AsyncDetection(frame_id=in_meta.frame_id, epoch=in_meta.epoch, detection=detection))


class AsyncCardDetector:
    """Runs the detection pipeline in a separate process.

    Lifecycle (all reversible, in this order):
      build_up()  -> spawn process, load ONNX sessions (blocks until ready)
      start()     -> resume listening on the input queue
      stop()      -> pause listening; sessions stay alive, start() may re-run
      tear_down() -> terminate process; build_up() may re-run
    """

    def __init__(self, detection_kwargs: dict[str, tp.Any], tear_down_timeout: float = 2.0) -> None:
        self._detection_kwargs = detection_kwargs
        self._tear_down_timeout = tear_down_timeout
        self._context = mp.get_context("spawn")
        self._in_queue = self._context.Queue(maxsize=1)
        self._out_queue = self._context.Queue(maxsize=1)
        self._start_up_event = self._context.Event()
        self._pause_event = self._context.Event()
        self._terminate_event = self._context.Event()

        self._process: mpp.BaseProcess | None = None
        self._worker = _detection_worker

    def build_up(self) -> None:
        proc = self._process
        if proc is not None and proc.is_alive():
            return
        self._start_up_event.clear()
        self._terminate_event.clear()
        self._pause_event.set()  # start paused; start() resumes
        proc = self._context.Process(
            target=self._worker,
            args=(
                self._in_queue,
                self._out_queue,
                self._start_up_event,
                self._pause_event,
                self._terminate_event,
                self._detection_kwargs,
            ),
            daemon=True,
        )
        proc.start()
        self._process = proc
        self._start_up_event.wait()

    def start(self) -> None:
        proc = self._process
        if proc is None or not proc.is_alive():
            raise RuntimeError("build_up() must be called before start()")
        self._pause_event.clear()

    def stop(self) -> None:
        self._pause_event.set()

    def tear_down(self) -> None:
        proc = self._process
        if proc is None or not proc.is_alive():
            return
        self._terminate_event.set()
        try:
            in_meta = self._in_queue.get_nowait()
            _unlink_shm(in_meta.shm_name)
        except queue.Empty:
            pass
        proc.join(timeout=self._tear_down_timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        self._terminate_event.clear()

    def open(self) -> None:
        self.build_up()
        self.start()

    def close(self) -> None:
        self.stop()
        self.tear_down()

    def submit(self, frame_id: int, epoch: int, img: np.ndarray) -> None:
        """Enqueue a frame for detection, dropping any unconsumed previous input."""
        img = np.ascontiguousarray(img)
        try:
            previous = self._in_queue.get_nowait()
            _unlink_shm(previous.shm_name)
        except queue.Empty:
            pass

        shm = mpsh.SharedMemory(create=True, size=img.nbytes)
        try:
            _write_to_shm(shm, img)
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        shm.close()

        meta = _InItemMeta(frame_id=frame_id, epoch=epoch, shm_name=shm.name, shape=img.shape, dtype=img.dtype.str)
        try:
            self._in_queue.put_nowait(meta)
        except queue.Full:
            _unlink_shm(meta.shm_name)

    def poll(self, epoch: int | None = None) -> AsyncDetection | None:
        """Return the latest completed detection, or ``None`` if none has arrived.

        The output queue holds at most one result (``maxsize=1``), so a single
        non-blocking read suffices. Results from an earlier ``epoch`` are
        discarded.
        """
        try:
            result = self._out_queue.get_nowait()
        except queue.Empty:
            return None
        if epoch is not None and result.epoch != epoch:
            return None
        return result

    def drain(self) -> None:
        """Discard everything currently queued on both input and output."""
        while True:
            try:
                in_meta = self._in_queue.get_nowait()
            except queue.Empty:
                break
            _unlink_shm(in_meta.shm_name)
        while True:
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> "AsyncCardDetector":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
