import numpy as np
import pytest

from setdetect.card_detection.detector import classify_cards, create_session, detect_card_corners
from setdetect.model import models_repo
from setdetect.model.card_classification import CARD_SIZE

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def real_model_paths() -> dict[str, str]:
    """Download (once, cached) the published ONNX models and return their local paths."""
    root = models_repo.default_root()
    try:
        paths = models_repo.download(root)
    except OSError as exc:
        pytest.skip(f"could not download models from {models_repo.REPO}: {exc}")
    return {name: str(path) for name, path in paths.items()}


@pytest.fixture(scope="session")
def corner_session(real_model_paths):
    return create_session(real_model_paths["card_corners"])


@pytest.fixture(scope="session")
def class_session(real_model_paths):
    return create_session(real_model_paths["card_classification"])


def test_corner_session_loads_and_runs(corner_session):
    img = np.zeros((640, 480, 3), dtype=np.uint8)
    detections = detect_card_corners(img, corner_session)
    assert isinstance(detections, list)
    for corners, visibility in detections:
        assert corners.shape == (4, 2)
        assert visibility.shape == (4,)


def test_classifier_session_loads_and_returns_valid_distributions(class_session):
    rng = np.random.default_rng(0)
    w, h = CARD_SIZE
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    (count_p, color_p, shape_p, fill_p) = classify_cards(img, [quad], class_session)[0]

    for probs in (count_p, color_p, shape_p, fill_p):
        assert probs.shape == (4,)
        assert np.all(probs >= 0.0)
        assert np.isclose(probs.sum(), 1.0)
