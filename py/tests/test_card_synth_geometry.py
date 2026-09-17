import numpy as np

from setdetect.synth_data.card_synth import (
    KIND_CARD,
    KIND_EDGE,
    KIND_ROT,
    BoardQuad,
    _cross2,
    _quad_iou,
    _valid_quad,
    add_false_positive_quads,
    jitter_quads,
)


def test_cross2():
    assert _cross2(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert _cross2(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == -1.0
    assert _cross2(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 0.0


def _square(x0=0.0, y0=0.0, size=10.0) -> np.ndarray:
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])


def test_valid_quad_accepts_convex_square():
    assert _valid_quad(_square())


def test_valid_quad_rejects_self_intersecting_bowtie():
    bowtie = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    assert not _valid_quad(bowtie)


def test_valid_quad_rejects_degenerate_zero_area():
    collinear = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    assert not _valid_quad(collinear)


def test_quad_iou_identical_is_one():
    q = _square()
    assert _quad_iou(q, q) == 1.0


def test_quad_iou_disjoint_is_zero():
    assert _quad_iou(_square(0, 0, 10), _square(100, 100, 10)) == 0.0


def test_quad_iou_partial_overlap():
    a = _square(0, 0, 10)  # [0,10]x[0,10]
    b = _square(5, 0, 10)  # [5,15]x[0,10]
    # intersection: [5,10]x[0,10] = 5*10=50; union = 100+100-50=150
    assert np.isclose(_quad_iou(a, b), 50 / 150)


def test_jitter_quads_noop_when_frac_zero():
    bq = BoardQuad(quad=_square(), visible=np.ones(4, dtype=bool), kind=KIND_CARD, label=None)
    original = bq.quad.copy()
    jitter_quads([bq], frac=0.0, res=(200, 200), rng=np.random.default_rng(0))
    np.testing.assert_array_equal(bq.quad, original)


def test_jitter_quads_keeps_result_valid():
    bq = BoardQuad(quad=_square(50, 50, 40), visible=np.ones(4, dtype=bool), kind=KIND_CARD, label=None)
    jitter_quads([bq], frac=0.06, res=(200, 200), rng=np.random.default_rng(0))
    assert _valid_quad(bq.quad)


def test_add_false_positive_quads_appends_expected_kinds():
    card = BoardQuad(quad=_square(100, 100, 40), visible=np.ones(4, dtype=bool), kind=KIND_CARD, label=None)
    quads = [card]
    add_false_positive_quads(quads, num_rot=2, num_edge=2, res=(300, 300), rng=np.random.default_rng(1))

    added = quads[1:]
    edge_added = [bq for bq in added if bq.kind == KIND_EDGE]
    rot_added = [bq for bq in added if bq.kind == KIND_ROT]
    assert len(edge_added) == 2
    assert len(rot_added) <= 2
    for bq in rot_added:
        assert _valid_quad(bq.quad)


def test_add_false_positive_quads_noop_with_no_real_cards():
    quads: list[BoardQuad] = []
    add_false_positive_quads(quads, num_rot=1, num_edge=1, res=(200, 200), rng=np.random.default_rng(0))
    assert quads == []
