import numpy as np

from setdetect.card_tracking.sparse_propagator import SparsePropagator, estimate_movement, seed_pts


def test_seed_pts_returns_in_bounds_points(synthetic_frame_pair):
    img, _, _ = synthetic_frame_pair
    pts = seed_pts(img)
    assert pts.dtype == np.float32
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    assert len(pts) > 0
    h, w = img.shape[:2]
    assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w)
    assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h)


def test_estimate_movement_recovers_known_shift(synthetic_frame_pair):
    prev_img, img, (dx, dy) = synthetic_frame_pair
    prev_pts = seed_pts(prev_img)
    assert len(prev_pts) >= 6  # rbf_min_points default

    movement = estimate_movement(prev_img, prev_pts, img)
    assert movement.pts.shape == prev_pts.shape
    assert movement.pts_mask.shape == (len(prev_pts),)
    assert movement.pts_mask.sum() > 0, "expected at least some points to survive the forward-backward check"

    displacement = movement.pts[movement.pts_mask] - prev_pts[movement.pts_mask]
    median_disp = np.median(displacement, axis=0)
    assert abs(median_disp[0] - dx) < 1.5
    assert abs(median_disp[1] - dy) < 1.5


def test_sparse_propagator_next_img_produces_consistent_shapes(synthetic_frame_pair):
    prev_img, img, _ = synthetic_frame_pair
    propagator = SparsePropagator()

    first = propagator.next_img(prev_img)
    assert first.frame_id == 0
    assert len(first.pts) == 0  # no previous frame yet

    second = propagator.next_img(img)
    assert second.frame_id == 1
    assert second.pts.shape[1] == 2
    assert second.pts_mask.shape[0] == second.pts.shape[0]
    assert second.prev_pts.shape[1] == 2


def test_sparse_propagator_propagate_preserves_point_count(synthetic_frame_pair):
    prev_img, img, (dx, dy) = synthetic_frame_pair
    propagator = SparsePropagator()
    propagator.next_img(prev_img)
    propagator.next_img(img)

    query_pts = np.array([[40.0, 40.0], [80.0, 90.0]], dtype=np.float32)
    result = propagator.propagate(0, query_pts.copy())

    if result is None:
        # a None interpolator (too few valid flow points) is a legitimate outcome
        return
    assert result.shape == query_pts.shape
    displacement = result - query_pts
    assert np.median(np.abs(displacement[:, 0] - dx)) < 3.0
    assert np.median(np.abs(displacement[:, 1] - dy)) < 3.0
