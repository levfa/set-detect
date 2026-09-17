import math

import numpy as np

from setdetect.set_game.game import Card, Color, Count, Fill, Shape
from setdetect.synth_data.board_synth import CardQuad, _canonicalize_corner_order, _card_support, _relax_positions


def test_card_support_axis_aligned():
    half_w, half_h = 10.0, 4.0
    assert math.isclose(_card_support(0.0, half_w, half_h, np.array([1.0, 0.0])), half_w)
    assert math.isclose(_card_support(0.0, half_w, half_h, np.array([0.0, 1.0])), half_h)


def test_card_support_rotated_by_90_swaps_axes():
    half_w, half_h = 10.0, 4.0
    rot = math.pi / 2
    assert math.isclose(_card_support(rot, half_w, half_h, np.array([1.0, 0.0])), half_h, abs_tol=1e-9)
    assert math.isclose(_card_support(rot, half_w, half_h, np.array([0.0, 1.0])), half_w, abs_tol=1e-9)


def test_relax_positions_pushes_overlapping_cards_apart():
    xs = [0.0, 5.0]
    ys = [0.0, 0.0]
    rot_angles = [0.0, 0.0]
    initial_dist = abs(xs[1] - xs[0])

    _relax_positions(xs, ys, rot_angles, half_w=10.0, half_h=15.0, rel_min_separation=1.0, rng=np.random.default_rng(0))

    final_dist = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
    assert final_dist > initial_dist


def test_relax_positions_noop_below_two_cards():
    xs, ys = [0.0], [0.0]
    _relax_positions(xs, ys, [0.0], half_w=10.0, half_h=10.0, rel_min_separation=1.0, rng=np.random.default_rng(0))
    assert xs == [0.0] and ys == [0.0]


def test_relax_positions_noop_when_disabled():
    xs, ys = [0.0, 1.0], [0.0, 0.0]
    _relax_positions(xs, ys, [0.0, 0.0], half_w=10.0, half_h=10.0, rel_min_separation=0.0, rng=np.random.default_rng(0))
    assert xs == [0.0, 1.0] and ys == [0.0, 0.0]


def _card() -> Card:
    return Card(Count.ONE, Color.RED, Shape.DIAMOND, Fill.OPEN)


def test_canonicalize_corner_order_normalizes_spin_flip():
    canonical_quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]])
    flipped_quad = canonical_quad[[2, 3, 0, 1]].copy()
    cq = CardQuad(_card(), flipped_quad, visible=np.array([True, False, True, True]))

    _canonicalize_corner_order([cq])

    np.testing.assert_allclose(cq.quad, canonical_quad)
    np.testing.assert_array_equal(cq.visible, [True, True, True, False])


def test_canonicalize_corner_order_leaves_already_canonical_quad():
    canonical_quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]])
    cq = CardQuad(_card(), canonical_quad.copy())

    _canonicalize_corner_order([cq])

    np.testing.assert_allclose(cq.quad, canonical_quad)
