import typing as tp
from dataclasses import dataclass

import cv2
import numpy as np

from setdetect.data.card_cutouts import MaskedCard, MaskedCardDataset, mask_inset_corners
from setdetect.set_game.game import Color, Count, Fill, Shape
from setdetect.synth_data.board_synth import Img, Quad, make_image
from setdetect.synth_data.effects import Effects

KIND_CARD = "card"
KIND_FAKE = "fake"
KIND_ROT = "rot"
KIND_EDGE = "edge"

_FAKE_W = 400
_FAKE_H = 560
# hues kept away from SET's red/green/purple symbols so fakes read as non-SET cards
_FAKE_HUES = [10, 25, 95, 125, 145, 165]
_FAKE_LETTERS = "ABCDHKMNPRTUVWXYZ"
_FAKE_WORDS = ["ACE", "VII", "X", "KING", "JOKER", "LUCKY"]


@dataclass(frozen=True)
class FakeCard(MaskedCard):
    """Card-like false positive: duck-types as a ``MaskedCard`` for ``make_image``."""


@dataclass
class BoardQuad:
    quad: Quad
    visible: np.ndarray
    kind: str
    label: tuple[str, str, str, str] | None


@dataclass(frozen=True)
class ExtractedCard:
    image: np.ndarray
    kind: str
    label: tuple[str, str, str, str] | None


def _accent_color(rng: np.random.Generator) -> tuple[int, int, int]:
    hue = int(rng.choice(_FAKE_HUES))
    sat = int(rng.integers(120, 255))
    val = int(rng.integers(90, 220))
    bgr = cv2.cvtColor(np.array([[[hue, sat, val]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _pastel_color(rng: np.random.Generator) -> tuple[int, int, int]:
    hue = int(rng.choice(_FAKE_HUES))
    sat = int(rng.integers(60, 130))
    val = int(rng.integers(225, 245))
    bgr = cv2.cvtColor(np.array([[[hue, sat, val]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _fake_face(rng: np.random.Generator) -> np.ndarray:
    """White, pastel, or two-tone card face on opaque BGRA."""
    style = rng.choice(["white", "pastel", "split"], p=[0.45, 0.3, 0.25])
    bg = np.full((_FAKE_H, _FAKE_W, 4), 255, np.uint8)
    if style == "white":
        tint = rng.uniform(0.92, 1.0, size=3)
        bg[..., :3] = np.rint(bg[..., :3] * tint[None, None, :]).astype(np.uint8)
        ramp = np.linspace(0.93, 1.07, _FAKE_H)[:, None, None]
        bg[..., :3] = np.clip(bg[..., :3] * ramp, 0, 255).astype(np.uint8)
    elif style == "pastel":
        bg[..., :3] = np.array(_pastel_color(rng), np.uint8)
    else:
        mid = _FAKE_H // 2
        bg[:mid, :, :3] = np.array(_pastel_color(rng), np.uint8)
        bg[mid:, :, :3] = np.array(_pastel_color(rng), np.uint8)
    return bg


def _draw_doodle(img: np.ndarray, color: tuple[int, int, int], rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    cx = int(rng.integers(int(0.15 * w), int(0.85 * w)))
    cy = int(rng.integers(int(0.15 * h), int(0.85 * h)))
    kind = rng.choice(["rect", "tri", "cross", "circle", "diamond", "letter"])
    rgba = (*color, 255)
    if kind == "letter":
        letter = rng.choice(list(_FAKE_LETTERS))
        scale = float(rng.uniform(1.5, 3.5))
        thickness = int(rng.integers(2, 4))
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(
            img,
            letter,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            rgba,
            thickness,
            cv2.LINE_AA,
        )
        img[..., 3] = 255
    elif kind == "circle":
        radius = int(rng.integers(15, 40))
        cv2.circle(img, (cx, cy), radius, rgba, thickness=2)
    elif kind == "diamond":
        r = int(rng.integers(20, 45))
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=rgba, thickness=2)
    else:
        span = int(rng.integers(25, 70))
        if kind == "rect":
            cv2.rectangle(
                img,
                (max(cx - span, 0), max(cy - span, 0)),
                (min(cx + span, w - 1), min(cy + span, h - 1)),
                color,
                thickness=2,
            )
        elif kind == "tri":
            pts = np.array(
                [
                    (cx, cy),
                    (min(cx + span, w - 1), max(cy - span, 0)),
                    (max(cx - span, 0), min(cy + span, h - 1)),
                ],
                np.int32,
            )
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        else:
            cv2.line(img, (cx, cy), (min(cx + span, w - 1), min(cy + span, h - 1)), color, thickness=3)
            cv2.line(img, (cx, cy), (max(cx - span, 0), min(cy + span, h - 1)), color, thickness=3)


def _draw_border(img: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    rgba = (*color, 255)
    cv2.rectangle(img, (12, 12), (w - 13, h - 13), rgba, thickness=2)
    cv2.rectangle(img, (18, 18), (w - 19, h - 19), rgba, thickness=1)


def _star_points(cx: int, cy: int, r_outer: float, r_inner: float) -> np.ndarray:
    pts = []
    for i in range(10):
        angle = -np.pi / 2 + i * np.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))
    return np.array(pts, np.int32)


def _draw_tarot(img: np.ndarray, color: tuple[int, int, int], rng: np.random.Generator) -> None:
    """Big centered glyph (letter or filled star) plus a short word band."""
    h, w = img.shape[:2]
    rgba = (*color, 255)
    if rng.random() < 0.5:
        letter = rng.choice(list(_FAKE_LETTERS))
        scale = float(rng.uniform(5.5, 7.5))
        thickness = int(rng.integers(4, 6))
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(
            img,
            letter,
            (w // 2 - tw // 2, h // 2 - th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            rgba,
            thickness,
            cv2.LINE_AA,
        )
        img[..., 3] = 255
    else:
        r = 0.22 * min(w, h)
        pts = _star_points(w // 2, h // 2 - 30, r, 0.45 * r)
        cv2.fillPoly(img, [pts], rgba)
    word = rng.choice(_FAKE_WORDS)
    scale = 1.1
    thickness = 2
    (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        img,
        word,
        (w // 2 - tw // 2, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        rgba,
        thickness,
        cv2.LINE_AA,
    )
    img[..., 3] = 255


def make_fake_cards(n: int, rng: np.random.Generator) -> list[FakeCard]:
    """Generate ``n`` card-like false positives: varied faces, doodles, letters,
    border frames, and tarot-style big-glyph cards."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    inset = np.asarray(mask_inset_corners(), np.float64)
    cards: list[FakeCard] = []
    for _ in range(n):
        bg = _fake_face(rng)
        if rng.random() < 0.25:
            _draw_border(bg, _accent_color(rng))
        if rng.random() < 0.2:
            _draw_tarot(bg, _accent_color(rng), rng)
        else:
            for _ in range(int(rng.integers(0, 4))):
                _draw_doodle(bg, _accent_color(rng), rng)
        cards.append(FakeCard(Count.ONE, Color.RED, Shape.OVAL, Fill.OPEN, image=bg, corners=inset))
    return cards


def _inside_fraction(q: np.ndarray, res: tuple[int, int]) -> float:
    """Fraction of quad ``q`` area that lies inside the ``res`` frame."""
    img_h, img_w = res
    x = np.clip(q[:, 0], 0, img_w)
    y = np.clip(q[:, 1], 0, img_h)
    clipped = abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0
    full = abs((q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]).sum()) / 2.0
    return float(clipped / max(full, 1e-9))


def _on_frame(q: np.ndarray, res: tuple[int, int], min_vis: float) -> bool:
    """True if quad ``q`` is not severely cut by the frame edge."""
    img_h, img_w = res
    cx, cy = q.mean(axis=0)
    if not (0 <= cx <= img_w and 0 <= cy <= img_h):
        return False
    return _inside_fraction(q, res) >= min_vis


def make_board(
    card_dataset: MaskedCardDataset,
    texture_provider: tp.Callable[[np.random.Generator], np.ndarray],
    num_cards: int = 8,
    num_fake: int = 4,
    res: tuple[int, int] = (640, 640),
    grid_size: tuple[int, int] = (3, 4),
    rot_range: tuple[float, float] = (-45, 45),
    rot_outlier_prob: float = 0.15,
    rot_outlier_range: tuple[float, float] = (60, 180),
    rel_min_separation: float = 0.85,
    persp: float = 1.25,
    effects: Effects | None = None,
    min_vis: float = 0.7,
    rng: np.random.Generator | None = None,
) -> tuple[Img, list[BoardQuad]]:
    """Composite a board of real cards plus fake false-positive cards.

    ``make_image`` appends quads in provider order, so quads map index-wise to
    the real-then-fake provider list; cards with less than ``min_vis`` of their
    area in the frame (or a centroid outside it) are dropped. ``effects``
    applies per-card photometric augmentation and specular before placement.
    """
    if num_cards < 0 or num_fake < 0:
        raise ValueError(f"num_cards and num_fake must be non-negative, got {num_cards}, {num_fake}")
    rng = rng if rng is not None else np.random.default_rng()
    real = card_dataset.random_cards(num_cards, rng)
    fake = make_fake_cards(num_fake, rng)
    providers: list[MaskedCard] = [*real, *fake]

    img, card_quads = make_image(
        lambda n, _rng: providers,
        texture_provider,
        num_cards=len(providers),
        res=res,
        grid_size=grid_size,
        rot_range=rot_range,
        rot_outlier_prob=rot_outlier_prob,
        rot_outlier_range=rot_outlier_range,
        rel_min_separation=rel_min_separation,
        persp=persp,
        card_highlighter=effects.specular_highlighter if effects is not None else None,
        card_shader=effects.shader if effects is not None else None,
        shadow_drawer=effects.shadow_drawer if effects is not None else None,
        hard_shadow_drawer=effects.hard_shadow_drawer if effects is not None else None,
        card_blurrer=effects.card_blurrer if effects is not None else None,
        bg_blurrer=effects.bg_blurrer if effects is not None else None,
        defocus_blurrer=effects.defocus_blurrer if effects is not None else None,
        augmenter=effects.augmenter if effects is not None else None,
        card_coverer=effects.card_coverer if effects is not None else None,
        rng=rng,
    )
    quads: list[BoardQuad] = []
    for k, card_quad in enumerate(card_quads):
        kind, label = (KIND_CARD, real[k].label) if k < num_cards else (KIND_FAKE, None)
        if not _on_frame(card_quad.quad, res, min_vis):
            continue
        quads.append(
            BoardQuad(
                quad=card_quad.quad,
                visible=card_quad.visible.copy(),
                kind=kind,
                label=label,
            )
        )
    return img, quads


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product (z of the 3D cross), a scalar per pair."""
    return float(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])


def _valid_quad(q: np.ndarray) -> bool:
    """True if ``q`` is a simple convex quad with a sane area."""
    crosses = [_cross2(q[i] - q[(i - 1) % 4], q[(i + 1) % 4] - q[i]) for i in range(4)]
    area = 0.5 * abs(sum(_cross2(q[i], q[(i + 1) % 4]) for i in range(4)))
    return len({np.sign(c) for c in crosses}) == 1 and area > 1.0


def jitter_quads(
    quads: list[BoardQuad],
    frac: float = 0.06,
    res: tuple[int, int] = (640, 640),
    tail_mult: float = 3.0,
    tail_prob: float = 0.25,
    rng: np.random.Generator | None = None,
) -> None:
    """Perturb every in-bounds corner by up to ``frac`` of the quad's own edge length.

    Offsets are a two-component mixture: ``1 - tail_prob`` of corners sample
    uniformly within ``amp``, the rest draw from a ``tail_mult``-wider uniform
    (heavy tails). Quads already poking outside the board (partial-visible
    cards) are left alone so jitter cannot distort the warp that
    ``extract_cards`` applies.
    """
    if frac <= 0:
        return
    rng = rng if rng is not None else np.random.default_rng()
    img_h, img_w = res
    for bq in quads:
        q = bq.quad
        if q.min() < 0:
            continue
        mean_edge = float(np.mean([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]))
        amp = frac * mean_edge
        for _ in range(24):
            mult = np.where(rng.random(q.shape) < tail_prob, tail_mult, 1.0)
            jittered = q + rng.uniform(-amp, amp, size=q.shape) * mult
            if jittered.min() < 0 or jittered[:, 0].max() > img_w or jittered[:, 1].max() > img_h:
                continue
            if _valid_quad(jittered):
                bq.quad = jittered
                break


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    bx0, by0, bx1, by1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union)


def add_false_positive_quads(
    quads: list[BoardQuad],
    num_rot: int = 3,
    num_edge: int = 3,
    res: tuple[int, int] = (640, 640),
    rng: np.random.Generator | None = None,
) -> None:
    """Append detector-style false-positive quads: rotated card quads and edge bands."""
    if num_rot < 0 or num_edge < 0:
        raise ValueError(f"num_rot and num_edge must be non-negative, got {num_rot}, {num_edge}")
    rng = rng if rng is not None else np.random.default_rng()
    cards = [bq for bq in quads if bq.kind == KIND_CARD]
    if not cards:
        return
    img_h, img_w = res

    # a real card quad rotated ~90deg (card aspect flips to landscape), scaled and
    # moved to a spot that matches no real card: a plausible wrong-orientation FP
    added, attempts = 0, 0
    while added < num_rot and attempts < 400:
        attempts += 1
        card = cards[int(rng.choice(len(cards)))]
        q = card.quad
        center = q.mean(axis=0)
        angle = np.deg2rad(90 + rng.uniform(-8, 8))
        scale = rng.uniform(1.05, 1.3)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        theta = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.1, 0.35) * max(img_w, img_h)
        rot_quad = center + (q - center) @ rot.T * scale
        rot_quad = rot_quad + np.array([np.cos(theta), np.sin(theta)]) * radius
        rot_quad = np.rint(rot_quad).astype(np.float32)
        if not _valid_quad(rot_quad):
            continue
        if (
            rot_quad[:, 0].min() < 0
            or rot_quad[:, 1].min() < 0
            or rot_quad[:, 0].max() > img_w
            or rot_quad[:, 1].max() > img_h
        ):
            continue
        if any(_quad_iou(rot_quad, c.quad) > 0.5 for c in cards):
            continue
        quads.append(BoardQuad(rot_quad, np.ones(4, dtype=bool), KIND_ROT, None))
        added += 1

    for _ in range(num_edge):
        card = cards[int(rng.choice(len(cards)))]
        q = card.quad
        frac = float(rng.uniform(0.3, 0.5))
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            quad = np.array([q[0], q[1], q[1] + (q[2] - q[1]) * frac, q[0] + (q[3] - q[0]) * frac])
        elif side == "bottom":
            quad = np.array([q[3] + (q[0] - q[3]) * frac, q[2] + (q[1] - q[2]) * frac, q[2], q[3]])
        elif side == "left":
            quad = np.array([q[0], q[0] + (q[1] - q[0]) * frac, q[3] + (q[2] - q[3]) * frac, q[3]])
        else:
            quad = np.array([q[1] + (q[0] - q[1]) * frac, q[1], q[2], q[2] + (q[3] - q[2]) * frac])
        quads.append(BoardQuad(quad.astype(np.float32), np.ones(4, dtype=bool), KIND_EDGE, None))


def extract_cards(
    img: np.ndarray,
    quads: list[BoardQuad],
    out_size: tuple[int, int] = (224, 320),
    crop_frac: float = 0.06,
) -> list[ExtractedCard]:
    """Perspective-unwrap each quad into an upright card crop of ``out_size``."""
    out_w, out_h = out_size
    target = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32)
    cw = max(int(round(crop_frac * out_w)), 0)
    ch = max(int(round(crop_frac * out_h)), 0)
    extracted: list[ExtractedCard] = []
    for bq in quads:
        mat = cv2.getPerspectiveTransform(bq.quad.astype(np.float32), target)
        warped = cv2.warpPerspective(img, mat, (out_w, out_h))
        cropped = warped[ch : out_h - ch, cw : out_w - cw]
        if cropped.size == 0:
            cropped = warped
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        extracted.append(ExtractedCard(resized, bq.kind, bq.label))
    return extracted


@dataclass(frozen=True)
class CardAugConfig:
    """Per-card photometric augmentation. Zero disables a transform."""

    hue_deg: float = 15.0
    sat: float = 0.35
    wb: float = 0.15
    bright: float = 0.25
    contrast: float = 0.30


def make_card_augmenter(
    cfg: CardAugConfig,
) -> tp.Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Build the per-card photometric fold applied by ``make_image`` before placement.

    Directional darkening lives on the shared ``effects.make_card_shading`` hook instead of
    here, so a board's cards share one coherent light direction rather than each drawing an
    independent random one. Specular and card blur similarly stay on the
    ``specular_highlighter``/``card_blurrer`` hooks. All random parameters come from the
    passed per-image ``rng`` so generation stays deterministic under
    ``np.random.default_rng(seed)``.
    """

    def _augment(card_img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rgb = card_img[..., :3].astype(np.float32)

        if cfg.hue_deg > 0 or cfg.sat > 0:
            hsv = cv2.cvtColor(card_img[..., :3], cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue_deg, cfg.hue_deg)) % 180
            hsv[..., 1] = hsv[..., 1] * rng.uniform(1.0 - cfg.sat, 1.0 + cfg.sat)
            rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        if cfg.wb > 0:
            rgb *= (1.0 + rng.uniform(-cfg.wb, cfg.wb, size=3))[None, None, :]

        if cfg.bright > 0 or cfg.contrast > 0:
            contrast = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
            bright = 1.0 + rng.uniform(-cfg.bright, cfg.bright)
            rgb = (rgb - 128.0) * contrast + 128.0
            rgb *= bright

        out = card_img.copy()
        out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        return out

    return _augment
