import cv2
import numpy as np
import pytest

from setdetect.card_detection.detector import DetectedCard


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _one_hot_probs(index: int) -> np.ndarray:
    probs = np.full(4, 0.01)
    probs[index] = 0.97
    return probs


@pytest.fixture
def make_board():
    """Factory for a small grid of non-overlapping, fully-visible ``DetectedCard``s."""

    def _make(n: int = 4, card_w: float = 100.0, card_h: float = 150.0, gap: float = 40.0) -> list[DetectedCard]:
        cols = max(1, int(np.ceil(np.sqrt(n))))
        cards = []
        for i in range(n):
            row, col = divmod(i, cols)
            x0 = col * (card_w + gap)
            y0 = row * (card_h + gap)
            corners = np.array(
                [[x0, y0], [x0 + card_w, y0], [x0 + card_w, y0 + card_h], [x0, y0 + card_h]],
                dtype=np.float64,
            )
            cards.append(
                DetectedCard(
                    corners=corners,
                    corner_visibility=np.ones(4, dtype=np.float32),
                    count_probs=_one_hot_probs(0),
                    color_probs=_one_hot_probs(0),
                    shape_probs=_one_hot_probs(0),
                    fill_probs=_one_hot_probs(0),
                )
            )
        return cards

    return _make


@pytest.fixture
def synthetic_frame_pair() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """A textured grayscale image and a copy shifted by a known integer translation."""
    rng = np.random.default_rng(0)
    h, w = 160, 160
    base = (rng.random((h, w)) * 255).astype(np.uint8)
    # coarse checkerboard-ish texture with enough gradient for goodFeaturesToTrack/optical flow
    base = cv2.GaussianBlur(base, (0, 0), 1.5)
    shift = (5, 3)  # (dx, dy)
    mat = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    shifted = cv2.warpAffine(base, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    return base, shifted, shift
