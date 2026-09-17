import numpy as np

from setdetect.card_detection.detector import (
    _CANON,
    CORNER_VISIBLE_THRESHOLD,
    DetectedCard,
    _apply_h,
    _fit_arrangement,
    _occlusion_order,
    cards_arrangement,
)


def test_apply_h_round_trip():
    mat = np.array([[1.5, 0.2, 30.0], [0.1, 1.2, -10.0], [0.0002, -0.0001, 1.0]])
    pts = np.array([[10.0, 20.0], [100.0, 5.0], [50.0, 80.0]])
    projected = _apply_h(mat, pts)
    recovered = _apply_h(np.linalg.inv(mat), projected)
    np.testing.assert_allclose(recovered, pts, atol=1e-6)


def test_fit_arrangement_single_card_recovers_canonical_pose():
    # An arbitrary (non-rectangular, perspective-distorted) quad in "image" space.
    quad = np.array([[100.0, 100.0], [230.0, 110.0], [215.0, 340.0], [90.0, 320.0]])
    world2img = _fit_arrangement(quad[np.newaxis, :, :])
    img2world = np.linalg.inv(world2img)
    recovered_world = _apply_h(img2world, quad)
    np.testing.assert_allclose(recovered_world, _CANON, atol=1e-3)


def test_cards_arrangement_empty_input():
    result = cards_arrangement([])
    assert result.card_poses == []
    assert result.card_z_orders == []
    assert result.card_width == 0.0
    assert result.card_height == 0.0


def test_cards_arrangement_grid(make_board):
    board = make_board(4)
    result = cards_arrangement(board)
    assert len(result.card_poses) == 4
    assert len(result.card_z_orders) == 4
    assert result.card_width > 0
    assert result.card_height > 0
    for pose in result.card_poses:
        assert len(pose) == 3
        assert all(np.isfinite(v) for v in pose)


def test_occlusion_order_ranks_covering_card_first():
    covered = DetectedCard(
        corners=np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]),
        corner_visibility=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),  # bottom-right hidden
        count_probs=np.zeros(4),
        color_probs=np.zeros(4),
        shape_probs=np.zeros(4),
        fill_probs=np.zeros(4),
    )
    coverer = DetectedCard(
        corners=np.array([[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]]),
        corner_visibility=np.ones(4, dtype=np.float32),
        count_probs=np.zeros(4),
        color_probs=np.zeros(4),
        shape_probs=np.zeros(4),
        fill_probs=np.zeros(4),
    )
    assert CORNER_VISIBLE_THRESHOLD == 0.5
    order = _occlusion_order([covered, coverer])
    assert order == [1, 0]


def test_occlusion_order_trivial_cases():
    assert _occlusion_order([]) == []
    single = DetectedCard(
        corners=np.zeros((4, 2)),
        corner_visibility=np.ones(4, dtype=np.float32),
        count_probs=np.zeros(4),
        color_probs=np.zeros(4),
        shape_probs=np.zeros(4),
        fill_probs=np.zeros(4),
    )
    assert _occlusion_order([single]) == [0]
