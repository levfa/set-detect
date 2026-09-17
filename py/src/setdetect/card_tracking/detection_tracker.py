import typing as tp

import cv2
import numpy as np

from setdetect.card_detection.async_detector import AsyncCardDetector
from setdetect.card_detection.detector import DetectedCard, Detection, _match_card
from setdetect.card_tracking.sparse_propagator import SparsePropagator


class DetectionTracker:
    """Combined async card detection and card corner propagation."""

    def __init__(
        self, detection_kwargs: dict[str, tp.Any] | None = None, propagation_kwargs: dict[str, tp.Any] | None = None
    ) -> None:
        # parameters
        if detection_kwargs is None:
            detection_kwargs = {}
        if propagation_kwargs is None:
            propagation_kwargs = {}

        # state
        self._epoch = 0
        self._detection_frame_id: int = -1
        self._detection: Detection | None = None

        # detector and propagator
        self._async_detector = AsyncCardDetector(detection_kwargs=detection_kwargs)
        self._propagator = SparsePropagator(**propagation_kwargs)

    def open(self) -> None:
        self._async_detector.open()

    def close(self) -> None:
        self._async_detector.close()

    def __enter__(self) -> "DetectionTracker":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def clear(self) -> None:
        self._epoch += 1
        self._detection_frame_id = -1
        self._detection = None
        self._async_detector.drain()
        self._propagator.clear()

    def _propagate_detection(self, detection: Detection, from_frame_id: int) -> Detection:
        cards = []
        for ddc in detection.detected_cards:
            dc_corners = self._propagator.propagate(from_frame_id, ddc.corners)
            if dc_corners is None:
                continue
            dc = DetectedCard(
                corners=dc_corners,
                corner_visibility=ddc.corner_visibility,
                count_probs=ddc.count_probs,
                color_probs=ddc.color_probs,
                shape_probs=ddc.shape_probs,
                fill_probs=ddc.fill_probs,
            )
            cards.append(dc)
        return Detection(
            detected_cards=cards,
            matches=[(dc, card) for dc in cards if (card := _match_card(dc)) is not None],
            matches_arrangement=detection.matches_arrangement,
        )

    def detect(self, img_rgb: np.ndarray) -> Detection | None:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        propagation_step = self._propagator.next_img(img_gray)
        self._detection_frame_id = propagation_step.frame_id

        self._async_detector.submit(propagation_step.frame_id, self._epoch, img_rgb)
        delayed_detection = self._async_detector.poll(self._epoch)

        if delayed_detection is not None:
            self._detection = self._propagate_detection(delayed_detection.detection, delayed_detection.frame_id)
        elif self._detection is not None:
            self._detection = self._propagate_detection(self._detection, self._detection_frame_id)
        else:
            self._detection = None
        return self._detection


class TimedDetectionTracker(DetectionTracker):
    def __init__(
        self, detection_kwargs: dict[str, tp.Any] | None = None, propagation_kwargs: dict[str, tp.Any] | None = None
    ) -> None:
        super().__init__(detection_kwargs, propagation_kwargs)
        self.times = {}

    def detect(self, img_rgb: np.ndarray) -> Detection | None:
        import time

        self.times = {}

        t0 = time.perf_counter()
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        self.times["cvtColor"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        propagation_step = self._propagator.next_img(img_gray)
        self._detection_frame_id = propagation_step.frame_id
        self.times["next_img"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._async_detector.submit(propagation_step.frame_id, self._epoch, img_rgb)
        self.times["submit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        delayed_detection = self._async_detector.poll(self._epoch)
        self.times["poll"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if delayed_detection is not None:
            self._detection = self._propagate_detection(delayed_detection.detection, delayed_detection.frame_id)
            self.times["propagate detection"] = time.perf_counter() - t0
        elif self._detection is not None:
            self._detection = self._propagate_detection(self._detection, self._detection_frame_id)
            self.times["propagate one frame"] = time.perf_counter() - t0
        else:
            self._detection = None
            self.times["propagate nothing"] = time.perf_counter() - t0
        return self._detection
