import json
import pathlib as pl
from dataclasses import dataclass

import cv2
import numpy as np

from setdetect.set_game.game import Card, Color, Count, Fill, Shape

LocationKey = tuple[str, str, str, str]
Locations = dict[LocationKey, list[tuple[int, int]]]

_EDGE_BAND = 40.0
_REFINE_BAND = 10.0
_EDGE_STEP = 3.0
_EDGE_TRIM = (0.18, 0.82)
_RANSAC_ITERS = 200
_RANSAC_THRESHOLD = 1.5


class Structure:
    def __init__(self) -> None:
        self.raw_dir = pl.Path("raw")
        self.masked_dir = pl.Path("masked")

    def raw(self, col: Color) -> pl.Path:
        return self.raw_dir / f"{col.value}.jpg"

    def rough_locations(self) -> pl.Path:
        return self.raw_dir / "rough_loc.jsonl"

    def exact_corners(self) -> pl.Path:
        return self.raw_dir / "corners.jsonl"

    def poses(self) -> pl.Path:
        return self.raw_dir / "poses.jsonl"

    def masked(self, key: LocationKey) -> pl.Path:
        return self.masked_dir / f"{key[0]}_{key[1]}_{key[2]}_{key[3]}.png"

    def masked_corners(self) -> pl.Path:
        return self.masked_dir / "corners.jsonl"


def load_locations(path: pl.Path) -> Locations:
    """Load the card-location jsonl (one record per card identity), empty if missing."""
    path = pl.Path(path)
    locations: Locations = {}
    if not path.is_file():
        return locations
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        data = json.loads(line)
        key = (data["count"], data["color"], data["shape"], data["fill"])
        locations[key] = [(x, y) for x, y in data["corners"]]
    return locations


def save_locations(path: pl.Path, locations: Locations) -> None:
    """Write the card-location jsonl, one record per card identity."""
    path = pl.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for (count, color, shape, fill), corners in locations.items():
            file.write(
                json.dumps(
                    {
                        "count": count,
                        "color": color,
                        "shape": shape,
                        "fill": fill,
                        "corners": corners,
                    }
                )
                + "\n"
            )


@dataclass
class CornerFit:
    corners: list[tuple[int, int]]
    edge_residuals: list[float]
    edge_inliers: list[int]


def find_exact_corners(
    img: np.ndarray,
    rough: list[tuple[int, int]],
) -> CornerFit | None:
    """Refine a card's rough corners to the intersections of its prolonged edges."""
    score = _cardness(img)
    corners = _order_corners(rough)
    centroid = np.mean(corners, axis=0)
    lines: list[tuple[float, float, float]] = []
    fits: list[tuple[float, int] | None] = [None] * 4
    band = _EDGE_BAND
    for _ in range(3):
        lines = []
        fits = []
        for i in range(4):
            p0, p1 = corners[i], corners[(i + 1) % 4]
            fit = _fit_edge(score, p0, p1, centroid, band)
            if fit is None:
                lines.append(_rough_line(p0, p1, centroid))
                fits.append(None)
            else:
                lines.append(fit.line)
                fits.append((fit.rms, fit.n_inliers))
        new_corners: list[np.ndarray] = []
        for i in range(4):
            inter = _line_intersection(lines[i], lines[(i - 1) % 4])
            new_corners.append(inter if inter is not None else corners[i])
        moved = max(np.linalg.norm(new_corners[i] - corners[i]) for i in range(4))
        corners = new_corners
        band = _REFINE_BAND
        if moved < 0.5:
            break

    height, width = img.shape[:2]
    clamped = [
        (
            min(max(int(round(float(p[0]))), 0), width - 1),
            min(max(int(round(float(p[1]))), 0), height - 1),
        )
        for p in corners
    ]
    residuals = [f[0] if f is not None else 0.0 for f in fits]
    inliers = [f[1] if f is not None else 0 for f in fits]
    return CornerFit(clamped, residuals, inliers)


def _cardness(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    score = lum - 0.25 * chroma
    return cv2.GaussianBlur(score, (0, 0), 1.5)


@dataclass
class EdgeFit:
    line: tuple[float, float, float]
    rms: float
    n_inliers: int


def _fit_edge(
    score: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    centroid: np.ndarray,
    band: float,
) -> EdgeFit | None:
    edge = p1 - p0
    length = np.linalg.norm(edge)
    if length < 1.0:
        return None
    direction = edge / length
    normal = np.array([-direction[1], direction[0]])
    if normal @ ((p0 + p1) / 2 - centroid) < 0:
        normal = -normal

    n_samples = int(length / _EDGE_STEP) + 1
    t = np.linspace(0.0, 1.0, n_samples)
    bases = p0[None] + np.outer(t, edge)
    half = int(band)
    s = np.arange(-half, half + 1, dtype=np.float64)
    coords = bases[:, None, :] + (s[None, :, None] * normal[None, None, :])
    vals = _sample(score, coords)

    grad = np.diff(vals, axis=1)
    abs_grad = np.abs(grad)
    gmax = abs_grad.max(axis=1, keepdims=True)
    threshold = np.maximum(0.25 * gmax, 10.0)
    genuine = abs_grad >= threshold

    jmin, jmax = 3, grad.shape[1] - 3
    starts = np.arange(jmin, jmax + 1)
    windows = np.lib.stride_tricks.sliding_window_view(vals, 3, axis=1)
    in_means = windows[:, starts - 3].mean(axis=-1)
    out_means = windows[:, starts + 1].mean(axis=-1)
    inward_brighter = in_means > out_means + 8.0
    candidates = genuine[:, starts] & inward_brighter

    positions = np.arange(grad.shape[1], dtype=np.float64)
    center_dist = np.abs((positions + 0.5) - half)
    cd = center_dist[starts]
    any_candidate = candidates.any(axis=1)
    k_idx = np.argmin(np.where(candidates, cd[None, :], 1e9), axis=1)
    fallback = genuine[:, starts].any(axis=1) & ~any_candidate
    k_idx = np.where(
        any_candidate,
        k_idx,
        np.argmin(np.where(genuine[:, starts], cd[None, :], 1e9), axis=1),
    )
    k = np.where(any_candidate | fallback, starts[k_idx], -1)
    ok = k >= 0
    k_safe = np.clip(k, 0, grad.shape[1] - 1)
    rows = np.arange(n_samples)
    g0 = grad[rows, np.maximum(k_safe - 1, 0)]
    g1 = grad[rows, k_safe]
    g2 = grad[rows, np.minimum(k_safe + 1, grad.shape[1] - 1)]
    denom = g0 - 2 * g1 + g2
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    offset = np.clip(0.5 * (g0 - g2) / denom, -0.5, 0.5)
    s_pos = k + offset + 0.5 - half
    points = bases + s_pos[:, None] * normal[None, :]

    keep = ok & (t >= _EDGE_TRIM[0]) & (t <= _EDGE_TRIM[1])
    pts = points[keep]
    if pts.shape[0] < 10:
        return None

    line, mask = _fit_line_robust(pts)
    if line is None or mask.sum() < 10:
        return None
    normal_l, c = line
    resid = np.abs(pts @ normal_l + c)
    rms = float(np.sqrt(np.mean(resid[mask] ** 2)))
    return EdgeFit((float(normal_l[0]), float(normal_l[1]), float(c)), rms, int(mask.sum()))


def _sample(score: np.ndarray, coords: np.ndarray) -> np.ndarray:
    map_x = coords[..., 0].T.astype(np.float32)
    map_y = coords[..., 1].T.astype(np.float32)
    return cv2.remap(score, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).T


def _fit_line_robust(pts: np.ndarray) -> tuple[tuple[np.ndarray, float] | None, np.ndarray]:
    n = pts.shape[0]
    rng = np.random.default_rng(0)
    best_inliers = 0
    best: tuple[np.ndarray, float, np.ndarray] | None = None
    for _ in range(_RANSAC_ITERS):
        idx = rng.integers(0, n, size=2)
        if idx[0] == idx[1]:
            continue
        a, b = pts[idx]
        v = b - a
        if np.linalg.norm(v) < 1e-6:
            continue
        n_ = np.array([-v[1], v[0]])
        n_ /= np.linalg.norm(n_)
        c = -(n_ @ a)
        dist = np.abs(pts @ n_ + c)
        inliers = int(np.sum(dist < _RANSAC_THRESHOLD))
        if inliers > best_inliers:
            best_inliers = inliers
            best = (n_, c, dist < _RANSAC_THRESHOLD)
    if best is None:
        return None, np.zeros(n, dtype=bool)
    n_, c, mask = best
    n_, c = _tls_line(pts[mask])
    resid = np.abs(pts @ n_ + c)
    sigma = np.median(resid[mask])
    keep = resid < max(3 * sigma, _RANSAC_THRESHOLD)
    if keep.sum() < 10:
        keep = resid < 3 * _RANSAC_THRESHOLD
    n_, c = _tls_line(pts[keep])
    return (n_, c), keep


def _tls_line(pts: np.ndarray) -> tuple[np.ndarray, float]:
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    u = vt[0]
    n_ = np.array([-u[1], u[0]])
    return n_, float(-(n_ @ center))


def _line_intersection(
    line1: tuple[float, float, float],
    line2: tuple[float, float, float],
) -> np.ndarray | None:
    a1, b1, d1 = line1
    a2, b2, d2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (b1 * d2 - b2 * d1) / det
    y = (a2 * d1 - a1 * d2) / det
    return np.array([x, y])


def _rough_line(
    p0: np.ndarray,
    p1: np.ndarray,
    centroid: np.ndarray,
) -> tuple[float, float, float]:
    v = p1 - p0
    length = np.linalg.norm(v)
    direction = v / length if length > 1e-9 else np.array([1.0, 0.0])
    n_ = np.array([-direction[1], direction[0]])
    if n_ @ ((p0 + p1) / 2 - centroid) < 0:
        n_ = -n_
    return (float(n_[0]), float(n_[1]), float(-(n_ @ p0)))


def _order_corners(points: list[tuple[int, int]]) -> list[np.ndarray]:
    pts = [np.asarray(p, float) for p in points]
    center = np.mean(pts, axis=0)
    angles = np.arctan2([p[1] - center[1] for p in pts], [p[0] - center[0] for p in pts]) % (2 * np.pi)
    ordered = [pts[i] for i in np.argsort(angles)]
    start = int(np.argmin([np.linalg.norm(p - pts[0]) for p in ordered]))
    return ordered[start:] + ordered[:start]


# Physical SET card is poker-size 2.5x3.5in -> portrait, width:height 5:7.
_CANON = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 7.0], [0.0, 7.0]])

_CARD_W = 400
_CARD_H = 560
_MASK_HUE = 25.0
_MASK_LUM_MIN = 20.0
_MASK_BAND = 16
_MASK_FEATHER_SIGMA = 1.0
_WB_TARGET = 230.0
_WB_SURF_LUM = 140.0
_EDGE_ZONE = 14
_EDGE_MIN_RATIO = 0.75
_MASK_TRIM = 8
_CARD_TARGET = np.array(
    [[0.0, 0.0], [_CARD_W, 0.0], [_CARD_W, _CARD_H], [0.0, _CARD_H]],
    np.float32,
)


@dataclass
class CardPose:
    theta_rad: float
    tx: float
    ty: float

    def world_corners(self) -> np.ndarray:
        c, s = np.cos(self.theta_rad), np.sin(self.theta_rad)
        rot = np.array([[c, -s], [s, c]])
        return _CANON @ rot.T + np.array([self.tx, self.ty])


@dataclass
class PlaneFit:
    A: np.ndarray
    anchor: LocationKey
    poses: dict[LocationKey, CardPose]
    residuals: dict[LocationKey, float]
    mean_residual_px: float
    max_residual_px: float


def _apply_h(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    out = mat @ np.hstack([pts, np.ones((len(pts), 1))]).T
    return (out[:2] / out[2:]).T


def _procrustes_pose(world: np.ndarray) -> CardPose:
    wc = world - world.mean(axis=0)
    cc = _CANON - _CANON.mean(axis=0)
    u, _, vt = np.linalg.svd(cc.T @ wc)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    t = world.mean(axis=0) - rot @ _CANON.mean(axis=0)
    return CardPose(float(np.arctan2(rot[1, 0], rot[0, 0])), float(t[0]), float(t[1]))


def fit_card_poses(corners: dict[LocationKey, np.ndarray]) -> PlaneFit:
    """Fit one shared plane homography A (world->image) plus a rigid pose per card."""

    keys = list(corners)
    obs = np.array([corners[k] for k in keys]).reshape(-1, 2)
    anchor = max(keys, key=lambda k: cv2.contourArea(corners[k].astype(np.int32)))
    img2world = np.array(
        cv2.getPerspectiveTransform(corners[anchor].astype(np.float32), _CANON.astype(np.float32)),
        np.float64,
    )
    A = np.linalg.inv(img2world)
    A /= A[2, 2]

    poses: dict[LocationKey, CardPose] = {k: _procrustes_pose(_apply_h(img2world, corners[k])) for k in keys}
    # Gauge: anchor pose at identity fixes rotation/translation, A[2,2]=1 fixes scale.
    poses[anchor] = CardPose(0.0, 0.0, 0.0)

    free = [k for k in keys if k != anchor]

    def pack() -> np.ndarray:
        out = list(A.ravel()[:8])
        for k in free:
            out += [poses[k].theta_rad, poses[k].tx, poses[k].ty]
        return np.array(out, np.float64)

    def unpack(p: np.ndarray) -> None:
        nonlocal A
        A = np.array(p[:8].tolist() + [1.0]).reshape(3, 3)
        for i, k in enumerate(free):
            th, tx, ty = p[8 + 3 * i : 11 + 3 * i]
            poses[k] = CardPose(float(th), float(tx), float(ty))

    def reprojection(p: np.ndarray) -> np.ndarray:
        unpack(p)
        out = np.empty((len(keys), 4, 2))
        for i, k in enumerate(keys):
            out[i] = _apply_h(A, poses[k].world_corners())
        return (out.reshape(-1, 2) - obs).ravel()

    p = pack()
    r = reprojection(p)
    for _ in range(25):
        n, m = len(p), len(r)
        jac = np.empty((m, n))
        for j in range(n):
            pp = p.copy()
            pp[j] += 1e-7
            jac[:, j] = (reprojection(pp) - r) / 1e-7
        step = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(n), -(jac.T @ r))
        pn = p + step
        rn = reprojection(pn)
        if np.mean(np.abs(rn)) < np.mean(np.abs(r)):
            p = pn
            r = rn

    unpack(p)
    residuals = {
        k: float(np.mean(np.linalg.norm(_apply_h(A, poses[k].world_corners()) - corners[k], axis=1))) for k in keys
    }
    errs = np.asarray(list(residuals.values()))
    return PlaneFit(A, anchor, poses, residuals, float(errs.mean()), float(errs.max()))


def fit_card_poses_per_file(corners: dict[LocationKey, np.ndarray]) -> dict[str, PlaneFit]:
    """Fit a separate plane homography per raw photo, grouped by card color.

    Each raw photo holds the cards of exactly one color, so the color is the file.
    """
    by_color: dict[str, dict[LocationKey, np.ndarray]] = {}
    for key, pts in corners.items():
        by_color.setdefault(key[1], {})[key] = pts
    return {color: fit_card_poses(pts) for color, pts in by_color.items()}


def save_poses(path: pl.Path, fits: dict[str, PlaneFit]) -> None:
    """Write per-color plane fits as a header line plus one record per card."""
    path = pl.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for color, fit in fits.items():
            file.write(
                json.dumps(
                    {
                        "color": color,
                        "homography": fit.A.ravel().tolist(),
                        "anchor": list(fit.anchor),
                    }
                )
                + "\n"
            )
            for key, pose in fit.poses.items():
                file.write(
                    json.dumps(
                        {
                            "count": key[0],
                            "color": key[1],
                            "shape": key[2],
                            "fill": key[3],
                            "theta_deg": float(np.degrees(pose.theta_rad)),
                            "tx": pose.tx,
                            "ty": pose.ty,
                            "world_corners": pose.world_corners().round(6).tolist(),
                            "residual_px": round(fit.residuals[key], 3),
                        }
                    )
                    + "\n"
                )


def load_poses(path: pl.Path) -> dict[str, PlaneFit]:
    """Read per-color plane fits written by ``save_poses`` (empty dict if missing)."""
    path = pl.Path(path)
    fits: dict[str, PlaneFit] = {}
    if not path.is_file():
        return fits

    color: str | None = None
    homography: np.ndarray | None = None
    anchor: LocationKey | None = None
    poses: dict[LocationKey, CardPose] = {}
    residuals: dict[LocationKey, float] = {}

    def flush() -> None:
        assert color is not None and homography is not None and anchor is not None
        errs = np.asarray(list(residuals.values()), np.float64)
        fits[color] = PlaneFit(
            A=homography,
            anchor=anchor,
            poses=poses,
            residuals=residuals,
            mean_residual_px=float(errs.mean()) if errs.size else 0.0,
            max_residual_px=float(errs.max()) if errs.size else 0.0,
        )

    for line in path.read_text().splitlines():
        data = json.loads(line)
        if "homography" in data:
            if color is not None:
                flush()
            color = data["color"]
            homography = np.asarray(data["homography"], np.float64).reshape(3, 3)
            anchor = tuple(data["anchor"])
            poses = {}
            residuals = {}
        elif color is not None:
            key = (data["count"], data["color"], data["shape"], data["fill"])
            poses[key] = CardPose(float(np.radians(data["theta_deg"])), float(data["tx"]), float(data["ty"]))
            residuals[key] = float(data["residual_px"])
    if color is not None:
        flush()
    return fits


def _card_class(bgr: np.ndarray) -> np.ndarray:
    """True where a pixel belongs to the card: surface, symbols, and shadowed edges.

    Card surface and symbols are strongly non-red (R - max(G,B) around -100) while
    the dark-red table is red-dominant (+30..+80). Shadows darken but preserve that
    hue, so curled/shadowed card edges stay classified as card.
    """
    f = bgr.astype(np.float32)
    r, g, b = f[..., 2], f[..., 1], f[..., 0]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (r - np.maximum(g, b) < _MASK_HUE) & (lum > _MASK_LUM_MIN)


def _surface_mask(bgr: np.ndarray) -> np.ndarray:
    m = cv2.erode(_card_class(bgr).astype(np.uint8), np.ones((9, 9), np.uint8))
    f = bgr.astype(np.float32)
    lum = 0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0]
    return (m > 0) & (lum > _WB_SURF_LUM)


def white_balance(bgr: np.ndarray) -> np.ndarray:
    """Map the card surface to neutral _WB_TARGET to remove the photo's blue cast.

    Two-slope LUT per channel: below the surface mean a von-Kries WB gain (neutral
    cast, keeps hue), above it a highlight-preserving ramp so bright pixels hit 255
    without clipping symbol colors.
    """
    f = bgr.astype(np.float32)
    surf = f[_surface_mask(bgr)].mean(axis=0)
    out = np.empty_like(f)
    for c in range(3):
        sc = float(surf[c])
        low = f[..., c] <= sc
        out[..., c] = np.where(
            low,
            _WB_TARGET * f[..., c] / max(sc, 1e-6),
            _WB_TARGET + (255.0 - _WB_TARGET) * (f[..., c] - sc) / max(255.0 - sc, 1e-6),
        )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _edge_neutralize(wb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remap the shadowed card-edge band to the surface color so no dark rim remains."""
    h, w = wb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    zone = (dist <= _EDGE_ZONE) & (alpha > 0)
    f = wb.astype(np.float32)
    surf = f[_surface_mask(wb)].mean(axis=0)
    lum_surf = 0.299 * surf[2] + 0.587 * surf[1] + 0.114 * surf[0]
    lum_px = 0.299 * f[..., 2] + 0.587 * f[..., 1] + 0.114 * f[..., 0]
    ratio = np.clip(lum_px / lum_surf, _EDGE_MIN_RATIO, 1.0)
    out = np.where(zone[..., None], surf[None, None, :] * ratio[..., None], f)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def mask_card_image(bgr: np.ndarray) -> np.ndarray:
    """Feathered alpha mask of the card incl. symbols, robust to edge shadows."""
    cls = _card_class(bgr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cls, connectivity=8)
    if n >= 2:
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        body = np.where(labels == largest, np.uint8(1), np.uint8(0))
    else:
        body = cls
    flooded = body.copy()
    cv2.floodFill(flooded, None, (0, 0), 1)
    filled = ((flooded == 0) | (body > 0)).astype(np.uint8)
    closed = np.asarray(
        cv2.morphologyEx(
            filled,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
    )
    band = np.zeros_like(closed)
    band[:_MASK_BAND, :] = 1
    band[-_MASK_BAND:, :] = 1
    band[:, :_MASK_BAND] = 1
    band[:, -_MASK_BAND:] = 1
    # Only near the frame (rounded-corner table wedge / border slivers) re-drop
    # pixels the closing bridged; the interior keeps full symbol coverage.
    closed = np.where(band, np.bitwise_and(np.asarray(closed, dtype=np.uint8), cls), closed)
    hard = cv2.erode(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), borderValue=0)
    yy, xx = np.mgrid[0 : hard.shape[0], 0 : hard.shape[1]]
    dist = np.minimum(np.minimum(xx, hard.shape[1] - 1 - xx), np.minimum(yy, hard.shape[0] - 1 - yy))
    hard[dist <= _MASK_TRIM] = 0
    hard_f = hard.astype(np.float32)
    feathered = cv2.GaussianBlur(hard_f, (0, 0), _MASK_FEATHER_SIGMA) * hard_f
    return np.clip(np.round(feathered * 255.0), 0, 255).astype(np.uint8)


def extract_cards(
    img: np.ndarray,
    homography: np.ndarray,
    world_corners: dict[LocationKey, np.ndarray],
) -> dict[LocationKey, np.ndarray]:
    """Warp each card upright, white-balance it, neutralize the edge band, and return BGRA."""
    out: dict[LocationKey, np.ndarray] = {}
    for key, world in world_corners.items():
        img_corners = _apply_h(homography, world).astype(np.float32)
        mat = cv2.getPerspectiveTransform(img_corners, _CARD_TARGET)
        warped = cv2.warpPerspective(img, mat, (_CARD_W, _CARD_H))
        alpha = mask_card_image(warped)
        bgra = cv2.cvtColor(_edge_neutralize(white_balance(warped), alpha), cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = alpha
        out[key] = bgra
    return out


def mask_inset_corners() -> list[tuple[int, int]]:
    """Corners of the visible card region within every masked PNG (the trim inset)."""
    t = int(_MASK_TRIM)
    return [
        (t, t),
        (_CARD_W - t, t),
        (_CARD_W - t, _CARD_H - t),
        (t, _CARD_H - t),
    ]


@dataclass(frozen=True)
class MaskedCard(Card):
    image: np.ndarray
    corners: np.ndarray


class MaskedCardDataset:
    """Cards from the extracted masked dataset: BGRA image plus image-space corners."""

    def __init__(self, root: pl.Path) -> None:
        self.root = pl.Path(root)
        structure = Structure()
        self._cards = self._scan(
            self.root / structure.masked_dir,
            self.root / structure.masked_corners(),
        )
        self._by_identity = {card.attributes: card for card in self._cards}
        self._identities = list(self._by_identity)

    @staticmethod
    def _scan(masked_dir: pl.Path, corners_path: pl.Path) -> list[MaskedCard]:
        corners = {key: np.asarray(pts, np.float64) for key, pts in load_locations(corners_path).items()}
        cards: list[MaskedCard] = []
        for image_path in sorted(masked_dir.glob("*.png")):
            count, color, shape, fill = image_path.stem.split("_")
            key = (count, color, shape, fill)
            if key not in corners:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            cards.append(
                MaskedCard(
                    count=Count(count),
                    color=Color(color),
                    shape=Shape(shape),
                    fill=Fill(fill),
                    image=image,
                    corners=corners[key],
                )
            )
        return cards

    def get(self, card: Card) -> MaskedCard:
        try:
            return self._by_identity[card.attributes]
        except KeyError:
            raise KeyError(f"no masked card for {card.label}") from None

    def random_cards(self, num_cards: int, rng: np.random.Generator) -> list[MaskedCard]:
        if num_cards < 0:
            raise ValueError(f"num_cards must be non-negative, got {num_cards}")
        if num_cards > len(self._identities):
            raise RuntimeError(
                f"requested {num_cards} distinct cards but dataset has only "
                f"{len(self._identities)} distinct card identities"
            )
        chosen = rng.choice(len(self._identities), size=num_cards, replace=False)
        return [self._by_identity[self._identities[int(i)]] for i in chosen]
