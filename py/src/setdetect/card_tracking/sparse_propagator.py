import typing as tp
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.linalg import LinAlgError
from scipy.interpolate import RBFInterpolator


@dataclass
class PropagationStep:
    """Result of one frame consumed by the propagator."""

    frame_id: int  # assigned id for the current frame
    prev_pts: np.ndarray  # points tracked from, in the previous frame
    pts: np.ndarray  # tracked positions in the current frame
    pts_mask: np.ndarray  # which pts are valid
    interpolator: RBFInterpolator | None = None  # movement interpolator


@dataclass
class Movement:
    """Result of optical-flow tracking between two frames."""

    pts: np.ndarray
    pts_mask: np.ndarray
    interpolator: RBFInterpolator | None


def seed_pts(
    img_gray: np.ndarray,
    *,
    rows: int = 8,
    cols: int = 8,
    max_corners_per_cell: int = 8,
    quality_level: float = 0.01,
    min_distance: float = 10.0,
) -> np.ndarray:
    """Find feature points on a uniform grid."""
    pts: list[np.ndarray] = []
    h, w = img_gray.shape[:2]
    row_edges = np.linspace(0, h, rows + 1).astype(int)
    col_edges = np.linspace(0, w, cols + 1).astype(int)
    for ry in range(rows):
        for cx in range(cols):
            cell = img_gray[row_edges[ry] : row_edges[ry + 1], col_edges[cx] : col_edges[cx + 1]]
            corners = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=max_corners_per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
            )
            if corners is None:
                continue
            corners = corners.reshape(-1, 2)
            corners[:, 0] += col_edges[cx]
            corners[:, 1] += row_edges[ry]
            pts.append(corners)
    if pts:
        return np.concatenate(pts, axis=0).astype(np.float32)
    return np.empty((0, 2), dtype=np.float32)


def estimate_movement(
    prev_img: np.ndarray,
    prev_pts: np.ndarray,
    img: np.ndarray,
    *,
    lk_win_size: tuple[int, int] = (15, 15),
    lk_max_level: int = 3,
    lk_criteria: tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    fb_thresh: float = 1.0,
    rbf_min_points: int = 6,
    rbf_smoothing: float = 1.0,
    rbf_max_points: int = 100,
) -> Movement:
    """Estimate movement between two frames via optical flow."""
    prev_pts_col = prev_pts.reshape(-1, 1, 2)

    # forward flow
    fwd_pts, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
        prev_img,
        img,
        prev_pts_col,
        np.empty_like(prev_pts_col),
        winSize=lk_win_size,
        maxLevel=lk_max_level,
        criteria=lk_criteria,
    )

    # backward flow (forward-backward check)
    bwd_pts, bwd_status, _ = cv2.calcOpticalFlowPyrLK(
        img,
        prev_img,
        fwd_pts,
        np.empty_like(fwd_pts),
        winSize=lk_win_size,
        maxLevel=lk_max_level,
        criteria=lk_criteria,
    )

    # filter
    fb_error = np.linalg.norm(prev_pts - bwd_pts.reshape(-1, 2), axis=1)
    fb_valid = (fwd_status.flatten() == 1) & (bwd_status.flatten() == 1) & (fb_error < fb_thresh)
    pts = fwd_pts.reshape(-1, 2).astype(np.float32)

    # interpolate movement
    displacements = pts - prev_pts
    idx = np.where(fb_valid)[0]
    if rbf_max_points and idx.size > rbf_max_points:
        step = idx.size / rbf_max_points
        idx = idx[np.floor(np.arange(rbf_max_points) * step).astype(int)]
    if idx.size < rbf_min_points:
        interpolator = None
    else:
        try:
            interpolator = RBFInterpolator(
                pts[idx],
                displacements[idx],
                kernel="thin_plate_spline",
                smoothing=rbf_smoothing,
            )
        except LinAlgError:
            interpolator = None

    return Movement(interpolator=interpolator, pts=pts, pts_mask=fb_valid)


class _RingBuffer:
    def __init__(self, size: int) -> None:
        self._size = size
        empty_pts = np.empty((0, 2), dtype=np.float32)
        empty_mask = np.empty(0, dtype=bool)
        self._buf: list[PropagationStep] = [PropagationStep(-1, empty_pts, empty_mask, empty_pts, None)] * size

    @property
    def size(self) -> int:
        return self._size

    def put(self, propagation_step: PropagationStep) -> None:
        slot = propagation_step.frame_id % self._size
        self._buf[slot] = propagation_step

    def get(self, frame_id: int) -> PropagationStep | None:
        slot = frame_id % self._size
        ps = self._buf[slot]
        if ps.frame_id != frame_id:
            return None
        return ps


class SparsePropagator:
    """Establish a frame-to-frame transformation through optical flow."""

    def __init__(
        self,
        *,
        history_size: int = 60,
        scale: float = 0.5,
        seed_kwargs: dict[str, tp.Any] | None = None,
        est_movement_kwargs: dict[str, tp.Any] | None = None,
    ) -> None:
        # parameters
        if seed_kwargs is None:
            seed_kwargs = {}
        if est_movement_kwargs is None:
            est_movement_kwargs = {}
        if history_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {history_size}")
        if not 0 < scale <= 1:
            raise ValueError(f"scale must be in (0, 1], got {scale}")

        # settings
        self._scale = scale
        self._seed_kwargs = seed_kwargs
        self._est_movement_kwargs = est_movement_kwargs

        # state
        self._frame_id = -1
        self._prev_img_scaled: np.ndarray | None = None
        self._prev_pts_scaled = np.empty((0, 2), dtype=np.float32)
        self._history = _RingBuffer(history_size)

        # helper
        self._no_movement = Movement(
            interpolator=None, pts=np.empty((0, 2), dtype=np.float32), pts_mask=np.empty(0, dtype=bool)
        )

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def clear(self) -> None:
        self._frame_id = -1
        self._prev_img_scaled = None
        self._prev_pts_scaled = np.empty((0, 2), dtype=np.float32)
        self._history = _RingBuffer(self._history.size)

    def _to_scale(self, img_gray: np.ndarray) -> np.ndarray:
        if self._scale == 1.0:
            return img_gray
        return cv2.resize(img_gray, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)

    def _seed_pts_scaled(self, img_scaled: np.ndarray) -> np.ndarray:
        return seed_pts(img_scaled, **self._seed_kwargs).astype(np.float32)

    def next_img(self, img_gray: np.ndarray) -> PropagationStep:
        img_scaled = self._to_scale(img_gray)
        self._frame_id += 1

        prev_img_scaled = self._prev_img_scaled
        prev_pts_scaled = self._prev_pts_scaled

        if prev_img_scaled is None or len(prev_pts_scaled) == 0:
            movement = self._no_movement
        else:
            movement = estimate_movement(prev_img_scaled, prev_pts_scaled, img_scaled, **self._est_movement_kwargs)

        # reseed points for movement estimation
        self._prev_img_scaled = img_scaled
        self._prev_pts_scaled = self._seed_pts_scaled(img_scaled)

        ps = PropagationStep(
            frame_id=self._frame_id,
            pts=movement.pts / self._scale,
            pts_mask=movement.pts_mask,
            prev_pts=prev_pts_scaled / self._scale,
            interpolator=movement.interpolator,
        )
        self._history.put(ps)
        return ps

    def propagate(self, from_frame_id: int, pts: np.ndarray) -> np.ndarray | None:
        scale = self._scale
        for fid in range(from_frame_id + 1, self._frame_id + 1):
            ps = self._history.get(fid)
            if ps is None or ps.interpolator is None:
                return None
            try:
                pts += ps.interpolator(pts * scale).astype(np.float32) / scale
            except LinAlgError:
                return None
        return pts
