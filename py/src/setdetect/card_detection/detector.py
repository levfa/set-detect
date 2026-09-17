import pathlib as pl
import typing as tp
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import scipy.optimize as so

from setdetect.model.card_classification import CARD_SIZE, MEAN, STD
from setdetect.set_game.game import Card, Color, Count, Fill, Shape


def create_session(model_path: str | pl.Path, device_id: int = 0) -> ort.InferenceSession:
    providers = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(model_path), providers=providers)


_ONNX_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def _dummy_input(inp: ort.NodeArg) -> np.ndarray:
    shape = [1 if not isinstance(dim, int) else dim for dim in (inp.shape or (1,))]
    return np.zeros(tuple(shape), dtype=_ONNX_TO_NP.get(inp.type, np.float32))


def _warm_up_session(session: ort.InferenceSession) -> None:
    """Run one dummy inference per input, paying ORT's one-time plan build + allocation."""
    feed = {inp.name: _dummy_input(inp) for inp in session.get_inputs()}
    session.run(None, feed)


# Maps classifier head index to game enum (index 3 = "other", unused).
COUNT_INV = {0: Count.ONE, 1: Count.TWO, 2: Count.THREE}
COLOR_INV = {0: Color.RED, 1: Color.GREEN, 2: Color.PURPLE}
SHAPE_INV = {0: Shape.DIAMOND, 1: Shape.OVAL, 2: Shape.SQUIGGLE}
FILL_INV = {0: Fill.OPEN, 1: Fill.SOLID, 2: Fill.STRIPED}


@dataclass
class DetectedCard:
    corners: np.ndarray  # (4, 2) float - corner pixel positions in original image
    corner_visibility: np.ndarray  # (4,) float - per-corner visibility
    count_probs: np.ndarray  # (4,) float
    color_probs: np.ndarray  # (4,) float
    shape_probs: np.ndarray  # (4,) float
    fill_probs: np.ndarray  # (4,) float


def _match_card(dc: DetectedCard) -> Card | None:
    indices = [
        dc.count_probs.argmax(),
        dc.color_probs.argmax(),
        dc.shape_probs.argmax(),
        dc.fill_probs.argmax(),
    ]
    if all(i != 3 for i in indices):
        return Card(
            count=COUNT_INV[int(indices[0])],
            color=COLOR_INV[int(indices[1])],
            shape=SHAPE_INV[int(indices[2])],
            fill=FILL_INV[int(indices[3])],
        )
    return None


RotatedRect = tuple[float, float, float]  # (center_x, center_y, angle_radians)

_CARD_AR = CARD_SIZE[0] / CARD_SIZE[1]  # width:height of the flat card
_CANON = np.array([[0.0, 0.0], [_CARD_AR, 0.0], [_CARD_AR, 1.0], [0.0, 1.0]])


@dataclass
class CardsArrangement:
    card_poses: list[RotatedRect]
    card_z_orders: list[int]  # per-card rank into card_poses: 0 = deepest, larger = higher
    card_width: float
    card_height: float


def _apply_h(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    out = mat @ np.hstack([pts, np.ones((len(pts), 1))]).T
    return (out[:2] / out[2:]).T


def _procrustes_pose(world: np.ndarray) -> tuple[float, float, float]:
    wc = world - world.mean(axis=0)
    cc = _CANON - _CANON.mean(axis=0)
    u, _, vt = np.linalg.svd(cc.T @ wc)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    t = world.mean(axis=0) - rot @ _CANON.mean(axis=0)
    return (float(np.arctan2(rot[1, 0], rot[0, 0])), float(t[0]), float(t[1]))


def _anchor_homography(quad: np.ndarray) -> np.ndarray:
    """Quad -> canonical homography, with a rectified-box fallback for degenerate quads.

    ``cv2.getPerspectiveTransform`` raises on some degenerate quads and silently
    returns a numerically broken homography on others (e.g. collinear corners), so
    the result is validated for finiteness and invertibility before use.
    """

    def _try_build(pts: np.ndarray) -> np.ndarray | None:
        try:
            return np.array(cv2.getPerspectiveTransform(pts.astype(np.float32), _CANON.astype(np.float32)), np.float64)
        except cv2.error:
            return None

    def usable(h: np.ndarray | None) -> bool:
        return h is not None and bool(np.all(np.isfinite(h))) and abs(np.linalg.det(h)) > 1e-9

    img2world = _try_build(quad)
    if not usable(img2world):
        box = np.array(
            [
                [quad[:, 0].min() - 4, quad[:, 1].min() - 4],
                [quad[:, 0].max() + 4, quad[:, 1].min() - 4],
                [quad[:, 0].max() + 4, quad[:, 1].max() + 4],
                [quad[:, 0].min() - 4, quad[:, 1].max() + 4],
            ],
            np.float32,
        )
        img2world = _try_build(box)
    if not usable(img2world):
        raise RuntimeError("could not build an invertible anchor homography")
    assert img2world is not None
    return img2world


def _fit_arrangement(corners: np.ndarray) -> np.ndarray:
    """Jointly fit the shared plane homography (world->image) and per-card rigid poses.

    The arrangement space is the world plane of the cards, where every card is a
    rectangle of known aspect ratio. Returns the world->image homography.
    """
    n = len(corners)
    obs = corners.reshape(-1, 2)
    anchor = max(range(n), key=lambda i: float(cv2.contourArea(corners[i].astype(np.int32))))

    img2world = _anchor_homography(corners[anchor])
    world2img = np.linalg.inv(img2world)
    world2img /= world2img[2, 2]

    poses: list[tuple[float, float, float]] = [_procrustes_pose(_apply_h(img2world, corners[i])) for i in range(n)]
    poses[anchor] = (0.0, 0.0, 0.0)
    free = [i for i in range(n) if i != anchor]

    def pack() -> np.ndarray:
        out = list(world2img.ravel()[:8])
        for i in free:
            out += [poses[i][0], poses[i][1], poses[i][2]]
        return np.array(out, np.float64)

    def unpack(p: np.ndarray) -> None:
        nonlocal world2img
        world2img = np.array(p[:8].tolist() + [1.0]).reshape(3, 3)
        for j, i in enumerate(free):
            th, tx, ty = p[8 + 3 * j : 11 + 3 * j]
            poses[i] = (float(th), float(tx), float(ty))

    def residual_and_jac(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Residual and analytic Jacobian of the reprojection over all corners."""
        unpack(p)
        th = np.array([poses[i][0] for i in free])
        tx = np.array([poses[i][1] for i in free])
        ty = np.array([poses[i][2] for i in free])
        cos_t, sin_t = np.cos(th), np.sin(th)

        rot = np.empty((len(free), 2, 2))
        rot[:, 0, 0] = cos_t
        rot[:, 0, 1] = -sin_t
        rot[:, 1, 0] = sin_t
        rot[:, 1, 1] = cos_t

        # world corners of every card (the anchor card sits at the canonical pose)
        world = np.empty((n, 4, 2))
        world[free] = (rot @ _CANON.T).transpose(0, 2, 1) + np.stack([tx, ty], axis=1)[:, None, :]
        world[anchor] = _CANON

        pts = world.reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        denom = world2img[2, 0] * x + world2img[2, 1] * y + 1.0
        m = world2img[0, 0] * x + world2img[0, 1] * y + world2img[0, 2]
        n_sign = world2img[1, 0] * x + world2img[1, 1] * y + world2img[1, 2]
        pred = np.stack([m / denom, n_sign / denom], axis=1)
        res = (pred - obs).ravel()

        n_pts = len(x)
        denom2 = denom * denom
        n_params = 8 + 3 * len(free)
        jac = np.zeros((n_pts, 2, n_params))
        jac[:, 0, 0] = x / denom
        jac[:, 0, 1] = y / denom
        jac[:, 0, 2] = 1.0 / denom
        jac[:, 1, 3] = x / denom
        jac[:, 1, 4] = y / denom
        jac[:, 1, 5] = 1.0 / denom
        jac[:, 0, 6] = -m * x / denom2
        jac[:, 0, 7] = -m * y / denom2
        jac[:, 1, 6] = -n_sign * x / denom2
        jac[:, 1, 7] = -n_sign * y / denom2

        # d pred / d world point (projection of a plane point under the homography)
        d_world = np.empty((n_pts, 2, 2))
        d_world[:, 0, 0] = (world2img[0, 0] * denom - m * world2img[2, 0]) / denom2
        d_world[:, 0, 1] = (world2img[0, 1] * denom - m * world2img[2, 1]) / denom2
        d_world[:, 1, 0] = (world2img[1, 0] * denom - n_sign * world2img[2, 0]) / denom2
        d_world[:, 1, 1] = (world2img[1, 1] * denom - n_sign * world2img[2, 1]) / denom2

        for j, i in enumerate(free):
            rot_p = np.array([[-sin_t[j], -cos_t[j]], [cos_t[j], -sin_t[j]]])  # d rot / d theta
            inner = np.empty((4, 2, 3))
            inner[:, :, 0] = (rot_p @ _CANON.T).T
            inner[:, :, 1] = np.array([1.0, 0.0])
            inner[:, :, 2] = np.array([0.0, 1.0])
            cols = slice(8 + 3 * j, 11 + 3 * j)
            jac[i * 4 : i * 4 + 4, :, cols] = np.einsum("nxy,nyb->nxb", d_world[i * 4 : i * 4 + 4], inner)
        return res, jac.reshape(len(res), n_params)

    def residual(p: np.ndarray) -> np.ndarray:
        return residual_and_jac(p)[0]

    def jacobian(p: np.ndarray) -> np.ndarray:
        return residual_and_jac(p)[1]

    p0 = pack()
    try:
        # the lm overload's stub types ``jac`` as str, so pass it via **
        result = so.least_squares(residual, p0, **{"method": "lm", "jac": jacobian})
    except Exception:
        result = so.least_squares(residual, p0)
    if result.cost > 0.5 * np.sum(residual(p0) ** 2):
        unpack(p0)
    else:
        unpack(result.x)
    return world2img


CORNER_VISIBLE_THRESHOLD = 0.5


def _occlusion_order(detected_cards: list[DetectedCard]) -> list[int]:
    """Return card indices from topmost to deepest, inferred from per-corner occlusion.

    A corner of card a hidden below card b (contained in b's quad with low
    corner visibility) is evidence that b is above a. The partial order from
    such pairwise evidence is topologically sorted; ambiguous/cyclic cases
    resolve deterministically by occlusion depth (hidden-corner count).
    """
    n = len(detected_cards)
    if n < 2:
        return list(range(n))

    def hidden_in(a: int, b: int) -> int:
        """Number of card a's hidden corners that lie inside card b's quad."""
        poly = detected_cards[b].corners.astype(np.float32)
        count = 0
        for (x, y), vis in zip(detected_cards[a].corners, detected_cards[a].corner_visibility):
            if vis >= CORNER_VISIBLE_THRESHOLD:
                continue
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                count += 1
        return count

    above: list[set[int]] = [set() for _ in range(n)]  # above[i] = cards covering card i
    for a in range(n):
        for b in range(a + 1, n):
            a_in_b = hidden_in(a, b)
            b_in_a = hidden_in(b, a)
            if a_in_b == b_in_a:
                continue
            if a_in_b > b_in_a:
                above[a].add(b)
            else:
                above[b].add(a)

    depth = [int(np.sum(conf < CORNER_VISIBLE_THRESHOLD)) for conf in (dc.corner_visibility for dc in detected_cards)]

    down: list[set[int]] = [set() for _ in range(n)]  # down[j] = cards that j covers
    for i in range(n):
        for j in above[i]:
            down[j].add(i)
    covers = [len(s) for s in above]
    remaining = set(range(n))
    frontier = {i for i in range(n) if covers[i] == 0}
    order: list[int] = []
    while remaining:
        if frontier:
            pick = min(frontier, key=lambda i: (depth[i], i))
        else:
            # cycle among the rest: break it deterministically
            pick = min(remaining, key=lambda i: (depth[i], i))
        order.append(pick)
        remaining.discard(pick)
        frontier.discard(pick)
        for j in down[pick]:
            if j in remaining:
                covers[j] -= 1
                if covers[j] == 0:
                    frontier.add(j)
    return order


def cards_arrangement(detected_cards: list[DetectedCard], inset: float = 0.02) -> CardsArrangement:
    if not detected_cards:
        return CardsArrangement(card_poses=[], card_z_orders=[], card_width=0.0, card_height=0.0)

    corners = np.array([dc.corners.astype(np.float64) for dc in detected_cards])
    n = len(detected_cards)

    world2img = _fit_arrangement(corners)
    img2world = np.linalg.inv(world2img)

    world_corners = np.array([_apply_h(img2world, corners[i]) for i in range(n)])

    align = np.arctan2(world2img[1, 0], world2img[0, 0])
    cos_a, sin_a = np.cos(align), np.sin(align)
    rot_align = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    world_corners = world_corners @ rot_align.T

    d_top = world_corners[:, 1] - world_corners[:, 0]
    rotations = np.arctan2(d_top[:, 1], d_top[:, 0])
    widths = np.linalg.norm(world_corners[:, 1] - world_corners[:, 0], axis=1)
    heights = np.linalg.norm(world_corners[:, 2] - world_corners[:, 1], axis=1)

    avg_w = float(np.mean(widths))
    avg_h = float(np.mean(heights))
    common_h = float(np.sqrt(avg_w * avg_h / _CARD_AR))
    common_w = common_h * _CARD_AR

    centers = world_corners.mean(axis=1)

    half_x = common_w / 2
    half_y = common_h / 2
    half = np.array([[half_x, half_y], [half_x, -half_y], [-half_x, -half_y], [-half_x, half_y]])
    quads = np.empty((n, 4, 2))
    cos_r, sin_r = np.cos(rotations), np.sin(rotations)
    for i in range(n):
        rot_card = np.array([[cos_r[i], -sin_r[i]], [sin_r[i], cos_r[i]]])
        quads[i] = centers[i] + half @ rot_card.T

    inset = min(max(inset, 0.0), 0.49)
    inner = 1.0 - 2.0 * inset
    x_min = float(quads[:, :, 0].min())
    x_max = float(quads[:, :, 0].max())
    y_min = float(quads[:, :, 1].min())
    y_max = float(quads[:, :, 1].max())
    range_x = x_max - x_min
    range_y = y_max - y_min
    if range_x < 1e-9 or range_y < 1e-9:
        range_x = max(range_x, 1.0)
        range_y = max(range_y, 1.0)

    scale = inner * min(1.0 / range_x, 1.0 / range_y)
    offset_x = (inner - range_x * scale) / 2 - x_min * scale
    offset_y = (inner - range_y * scale) / 2 - y_min * scale

    card_poses: list[RotatedRect] = [
        (
            float(centers[i, 0] * scale + offset_x),
            float(centers[i, 1] * scale + offset_y),
            float(rotations[i]),
        )
        for i in range(n)
    ]

    z_orders = [0] * n
    for rank, i in enumerate(_occlusion_order(detected_cards)):
        z_orders[i] = n - 1 - rank

    return CardsArrangement(
        card_poses=card_poses,
        card_z_orders=z_orders,
        card_width=common_w * scale,
        card_height=common_h * scale,
    )


@dataclass
class Detection:
    detected_cards: list[DetectedCard]
    matches: list[tuple[DetectedCard, Card]]
    matches_arrangement: CardsArrangement


def _letterbox(
    img: np.ndarray,
    target_h: int = 640,
    target_w: int = 640,
) -> tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    r = min(target_h / h, target_w / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = target_w - new_unpad[0]
    dh = target_h - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, left, top


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_j = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = rest[inds]
    return keep


def _parse_corner_output(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    img_h: int,
    img_w: int,
    ratio: float,
    pad_left: float,
    pad_top: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse raw corner-keypoint output (1, 17, 8400) into per-card corners."""
    raw = output[0].T  # (8400, 17)

    boxes_xywh = raw[:, :4]
    scores = raw[:, 4]
    kpts_raw = raw[:, 5:]  # (8400, 12)

    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    kpts_raw = kpts_raw[mask]

    if boxes_xywh.size == 0:
        return []

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    keep = _nms(boxes_xyxy, scores, iou_threshold)

    results: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in keep:
        kpt = kpts_raw[idx].reshape(4, 3)  # (x, y, vis) in letterboxed coords
        kpt[:, 0] = (kpt[:, 0] - pad_left) / ratio
        kpt[:, 1] = (kpt[:, 1] - pad_top) / ratio
        kpt[:, 0] = np.clip(kpt[:, 0], 0, img_w - 1)
        kpt[:, 1] = np.clip(kpt[:, 1], 0, img_h - 1)
        corners = kpt[:, :2].astype(np.float32)
        conf = kpt[:, 2].astype(np.float32)
        results.append((corners, conf))

    return results


def detect_card_corners(
    img_rgb: np.ndarray,
    session: ort.InferenceSession,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> list[tuple[np.ndarray, np.ndarray]]:
    letterboxed, ratio, pad_left, pad_top = _letterbox(img_rgb, imgsz, imgsz)

    tensor = letterboxed.astype(np.float32)
    tensor *= 1.0 / 255.0
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), 0)  # BCHW

    output = tp.cast(np.ndarray, session.run(None, {"images": tensor})[0])  # (1, 17, 8400)
    h, w = img_rgb.shape[:2]
    return _parse_corner_output(output, conf, iou, h, w, ratio, pad_left, pad_top)


def _warp_card(
    img: np.ndarray,
    quad: np.ndarray,
    size: tuple[int, int] = CARD_SIZE,
    crop_frac: float = 0.06,
    warp_pad: int = 20,
) -> np.ndarray:
    w, h = size

    # Compute bounding box of the quad in the original image
    x_min = int(np.min(quad[:, 0]))
    x_max = int(np.max(quad[:, 0]))
    y_min = int(np.min(quad[:, 1]))
    y_max = int(np.max(quad[:, 1]))

    # Add padding around the quad, clamp to image borders
    pad = max(warp_pad, 0)
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(img.shape[1], x_max + pad)
    y_max = min(img.shape[0], y_max + pad)

    # Crop the image to the padded bounding box
    img_cropped = img[y_min:y_max, x_min:x_max]

    # Adjust quad coordinates to the cropped image, ensure float32
    quad_adj = (quad - [x_min, y_min]).astype(np.float32)

    # Perspective transform from adjusted quad to rectangle
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    persp_t = cv2.getPerspectiveTransform(quad_adj, dst)
    warped = cv2.warpPerspective(img_cropped, persp_t, (w, h))

    cw = max(int(round(crop_frac * w)), 0)
    ch = max(int(round(crop_frac * h)), 0)
    cropped = warped[ch : h - ch, cw : w - cw]
    if cropped.size == 0:
        cropped = warped
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


_NORM_SCALE = (1.0 / 255.0) / np.asarray(STD, dtype=np.float32)
_NORM_SHIFT = -np.asarray(MEAN, dtype=np.float32) / np.asarray(STD, dtype=np.float32)


def _preprocess_card(img_rgb: np.ndarray) -> np.ndarray:
    tensor = img_rgb.astype(np.float32)
    np.multiply(tensor, _NORM_SCALE, out=tensor)
    tensor += _NORM_SHIFT
    return tensor.transpose(2, 0, 1)[np.newaxis]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def classify_cards(
    img_rgb: np.ndarray,
    corners_list: list[np.ndarray],
    session: ort.InferenceSession,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not corners_list:
        return []

    batch = np.concatenate(
        [_preprocess_card(_warp_card(img_rgb, c)) for c in corners_list],
        axis=0,
    )

    outputs = tp.cast(list[np.ndarray], session.run(None, {"input": batch}))
    # outputs: [count_logits, shape_logits, fill_logits, color_logits] each (N, 4)
    results = []
    n = batch.shape[0]
    for i in range(n):
        count_p = _softmax(outputs[0][i])
        shape_p = _softmax(outputs[1][i])
        fill_p = _softmax(outputs[2][i])
        color_p = _softmax(outputs[3][i])
        results.append((count_p, color_p, shape_p, fill_p))

    return results


def detect_cards(
    img_rgb: np.ndarray,
    corner_session: ort.InferenceSession,
    class_session: ort.InferenceSession,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> Detection:
    """Run the full detection + classification pipeline on a single image."""
    detections = detect_card_corners(img_rgb, corner_session, conf, iou, imgsz)
    corners_list = [c for c, _ in detections]
    classifications = classify_cards(img_rgb, corners_list, class_session)

    detected_cards = []
    for (corners, corner_vis), (count_p, color_p, shape_p, fill_p) in zip(detections, classifications):
        detected_cards.append(
            DetectedCard(
                corners=corners,
                corner_visibility=corner_vis,
                count_probs=count_p,
                color_probs=color_p,
                shape_probs=shape_p,
                fill_probs=fill_p,
            )
        )

    matches = [(dc, card) for dc in detected_cards if (card := _match_card(dc)) is not None]
    matched_cards = [dc for dc, _ in matches]
    return Detection(
        detected_cards=detected_cards, matches=matches, matches_arrangement=cards_arrangement(matched_cards)
    )


class CardDetector:
    """Detection pipeline."""

    def __init__(
        self,
        corner_weights: pl.Path,
        class_weights: pl.Path,
        *,
        device_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        warm_up: bool = True,
    ) -> None:
        self.corner_session = create_session(corner_weights, device_id)
        self.class_session = create_session(class_weights, device_id)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        if warm_up:
            _warm_up_session(self.corner_session)
            _warm_up_session(self.class_session)

    def detect(self, img_rgb: np.ndarray) -> Detection:
        return detect_cards(img_rgb, self.corner_session, self.class_session, self.conf, self.iou, self.imgsz)
