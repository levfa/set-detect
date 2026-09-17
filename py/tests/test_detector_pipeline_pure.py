import numpy as np

from setdetect.card_detection.detector import (
    COLOR_INV,
    COUNT_INV,
    FILL_INV,
    SHAPE_INV,
    DetectedCard,
    _letterbox,
    _match_card,
    _nms,
    _parse_corner_output,
    _softmax,
    _xywh_to_xyxy,
)


def test_xywh_to_xyxy():
    boxes = np.array([[10.0, 10.0, 4.0, 2.0]])
    out = _xywh_to_xyxy(boxes)
    np.testing.assert_allclose(out, [[8.0, 9.0, 12.0, 11.0]])


def test_nms_suppresses_lower_score_overlap():
    boxes_xyxy = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavily overlaps box 0, lower score -> suppressed
            [50.0, 50.0, 60.0, 60.0],  # disjoint
        ]
    )
    scores = np.array([0.9, 0.8, 0.95])
    keep = _nms(boxes_xyxy, scores, iou_threshold=0.5)
    assert keep == [2, 0]


def test_nms_keeps_disjoint_boxes():
    boxes_xyxy = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = np.array([0.6, 0.7])
    keep = _nms(boxes_xyxy, scores, iou_threshold=0.5)
    assert set(keep) == {0, 1}


def test_softmax_sums_to_one_and_preserves_order():
    x = np.array([1.0, 3.0, 2.0, -1.0])
    out = _softmax(x)
    assert np.isclose(out.sum(), 1.0)
    assert np.argsort(out).tolist() == np.argsort(x).tolist()


def _one_hot(index: int) -> np.ndarray:
    probs = np.zeros(4)
    probs[index] = 1.0
    return probs


def test_match_card_returns_card_when_all_heads_confident():
    dc = DetectedCard(
        corners=np.zeros((4, 2)),
        corner_visibility=np.ones(4),
        count_probs=_one_hot(0),
        color_probs=_one_hot(1),
        shape_probs=_one_hot(2),
        fill_probs=_one_hot(0),
    )
    card = _match_card(dc)
    assert card is not None
    assert card.count == COUNT_INV[0]
    assert card.color == COLOR_INV[1]
    assert card.shape == SHAPE_INV[2]
    assert card.fill == FILL_INV[0]


def test_match_card_returns_none_when_any_head_is_other():
    dc = DetectedCard(
        corners=np.zeros((4, 2)),
        corner_visibility=np.ones(4),
        count_probs=_one_hot(3),  # "other": reject
        color_probs=_one_hot(0),
        shape_probs=_one_hot(0),
        fill_probs=_one_hot(0),
    )
    assert _match_card(dc) is None


def test_letterbox_pads_to_target_shape():
    img = np.zeros((60, 100, 3), dtype=np.uint8)
    padded, ratio, pad_left, pad_top = _letterbox(img, target_h=128, target_w=128)
    assert padded.shape[:2] == (128, 128)
    assert padded.dtype == img.dtype
    assert np.isclose(ratio, 128 / 100, atol=1e-2)
    assert pad_left >= 0
    assert pad_top >= 0


def _make_raw_output(box_xywh, corners_xyv, score: float) -> np.ndarray:
    output = np.zeros((1, 17, 8400), dtype=np.float32)
    output[0, 0:4, 0] = box_xywh
    output[0, 4, 0] = score
    output[0, 5:17, 0] = np.asarray(corners_xyv, dtype=np.float32).ravel()
    return output


def test_parse_corner_output_no_detections_above_threshold():
    output = _make_raw_output([50, 50, 20, 20], [[0, 0, 1]] * 4, score=0.1)
    results = _parse_corner_output(
        output, conf_threshold=0.5, iou_threshold=0.45, img_h=100, img_w=100, ratio=1.0, pad_left=0.0, pad_top=0.0
    )
    assert results == []


def test_parse_corner_output_single_detection():
    corners_xyv = [[10, 10, 1.0], [90, 10, 1.0], [90, 90, 1.0], [10, 90, 1.0]]
    output = _make_raw_output([50, 50, 80, 80], corners_xyv, score=0.9)
    results = _parse_corner_output(
        output, conf_threshold=0.5, iou_threshold=0.45, img_h=100, img_w=100, ratio=1.0, pad_left=0.0, pad_top=0.0
    )
    assert len(results) == 1
    corners, conf = results[0]
    np.testing.assert_allclose(corners, [[10, 10], [90, 10], [90, 90], [10, 90]])
    np.testing.assert_allclose(conf, [1.0, 1.0, 1.0, 1.0])
