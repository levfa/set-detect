import math
import typing as tp
from dataclasses import dataclass

import cv2
import numpy as np


def make_specular_highlight(
    card_img: np.ndarray,
    rot_agl: float,
    prob: float = 0.8,
    intensity: float = 0.5,
    size: float = 0.30,
    streak: float = 2.5,
    direction: float | None = None,
    align: str = "global",
    rng: np.random.Generator | None = None,
    res_scale: float = 0.125,
) -> np.ndarray | None:
    """Card-local additive specular highlight layer at ``res_scale`` resolution.

    Returns ``None`` when this card gets no highlight. Layer values are a 0..1
    brightness fraction (multiply by 255 to add to a uint8 image). Blobs are soft
    2-D Gaussians computed on a smaller ``res_scale``-sized grid for speed; the
    caller resizes the returned layer onto the full card. Randomness comes from
    ``rng`` (a fresh Generator when None) so generation stays deterministic under
    ``np.random.default_rng(seed)``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if rng.random() >= prob:
        return None
    h, w = card_img.shape[:2]
    lh, lw = max(round(h * res_scale), 1), max(round(w * res_scale), 1)
    if direction is None:
        direction = float(rng.uniform(0, 180))
    ang = np.deg2rad(direction)
    if align == "global":
        ang -= rot_agl
    layer = np.zeros((lh, lw), np.float32)
    yy, xx = np.mgrid[0:lh, 0:lw]
    n_blobs = 2 if rng.random() < 0.2 else 1
    for _ in range(n_blobs):
        u = rng.uniform(-0.3, 0.3) * lw
        v = rng.uniform(-0.3, 0.3) * lh
        length = size * lw * rng.uniform(0.8, 1.2)
        a = length / 2
        b = max(a / streak, 1.0)
        c, s = np.cos(ang), np.sin(ang)
        dx = xx - (lw / 2 + u)
        dy = yy - (lh / 2 + v)
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        layer += intensity * rng.uniform(0.7, 1.0) * np.exp(-((xr / a) ** 2 + (yr / b) ** 2) / 2)
    return layer


@dataclass(frozen=True)
class SpecConfig:
    prob: float = 0.8
    intensity: float = 0.5
    size: float = 0.30
    streak: float = 2.5
    direction: float | None = None
    align: str = "global"


def make_specular_highlighter(
    spec: SpecConfig,
) -> tp.Callable[[np.ndarray, float, np.random.Generator], np.ndarray | None]:
    def _hl(card_img: np.ndarray, rot_agl: float, rng: np.random.Generator) -> np.ndarray | None:
        if spec.prob <= 0:
            return None
        return make_specular_highlight(
            card_img,
            rot_agl,
            prob=spec.prob,
            intensity=spec.intensity,
            size=spec.size,
            streak=spec.streak,
            direction=spec.direction,
            align=spec.align,
            rng=rng,
        )

    return _hl


@dataclass(frozen=True)
class ShadeConfig:
    prob: float = 0.8
    amp: float = 0.18
    type: str = "mixed"
    direction: str = "global"


@dataclass(frozen=True)
class ShadowConfig:
    prob: float = 0.5
    strength: float = 0.3
    offset_px: tuple[float, float] = (0.5, 1.5)
    blur_px: float = 0.35


@dataclass(frozen=True)
class BlurConfig:
    card_prob: float = 0.7
    card_sigma: float = 0.02
    bg_prob: float = 0.8
    bg_sigma: float = 2.5
    defocus_prob: float = 0.10
    defocus_sigma: float = 4.5


def make_card_blurrer(
    cfg: BlurConfig,
) -> tp.Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def _blur(card_img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if cfg.card_prob <= 0 or rng.random() >= cfg.card_prob:
            return card_img
        sigma = cfg.card_sigma * rng.uniform(0.0, 1.0) * max(card_img.shape[1], 1)
        if sigma < 0.5:
            return card_img
        k = max(int(math.ceil(sigma * 3)) | 1, 3)
        blurred = cv2.GaussianBlur(card_img, (k, k), sigma)
        blurred[..., 3] = card_img[..., 3]
        return blurred

    return _blur


def make_bg_blurrer(
    cfg: BlurConfig,
) -> tp.Callable[[np.ndarray, np.random.Generator], None]:
    def _blur(img: np.ndarray, rng: np.random.Generator) -> None:
        if cfg.bg_prob <= 0 or rng.random() >= cfg.bg_prob:
            return
        sigma = cfg.bg_sigma * rng.uniform(0.0, 1.0)
        if sigma < 0.5:
            return
        k = max(int(math.ceil(sigma * 3)) | 1, 3)
        img[...] = cv2.GaussianBlur(img, (k, k), sigma)

    return _blur


def make_defocus_blurrer(
    cfg: BlurConfig,
) -> tp.Callable[[np.ndarray, np.random.Generator], None]:
    def _blur(img: np.ndarray, rng: np.random.Generator) -> None:
        if cfg.defocus_prob <= 0 or rng.random() >= cfg.defocus_prob:
            return
        # floor at half the cap so triggered boards are out of focus
        sigma = cfg.defocus_sigma * rng.uniform(0.5, 1.0)
        k = max(int(math.ceil(sigma * 3)) | 1, 3)
        img[...] = cv2.GaussianBlur(img, (k, k), sigma)

    return _blur


_SHADOW_COLOR = np.array([15.0, 15.0, 18.0], dtype=np.float32)


def make_card_shading(
    card_img: np.ndarray,
    rot_agl: float,
    light_dir: float,
    prob: float,
    amp: float,
    type_: str,
    direction: str,
    rng: np.random.Generator,
    res_scale: float = 0.125,
) -> np.ndarray | None:
    """Low-res monochromatic multiplicative darkening: a soft shadow across a card.

    Returns ``None`` when this card gets no shading. The returned factor is a single
    per-pixel value in ``[max(1 - amp, 0), 1.0]`` broadcast identically across all three
    channels (never brightens, never tints one channel differently from another: a real
    shadow/occlusion effect, not a lighting-color effect), so ``factor * 255`` always fits a
    uint8 without wraparound. Falls off smoothly (via
    a power curve, not a hard linear cutoff) along the board's light axis rotated into the
    card's own frame (``directional``, when ``direction`` is "global") or radially from the
    card's center (``vignette``, e.g. corners tucked under a neighbor). Randomness comes from
    ``rng``.
    """
    if rng.random() >= prob:
        return None
    h, w = card_img.shape[:2]
    lh, lw = max(round(h * res_scale), 1), max(round(w * res_scale), 1)
    yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    if type_ == "mixed":
        type_ = "directional" if rng.random() < 0.5 else "vignette"
    amp = min(amp * rng.uniform(0.6, 1.4), 1.0)
    u = (xx - (lw - 1) / 2) / max(lw / 2, 1)
    v = (yy - (lh - 1) / 2) / max(lh / 2, 1)
    if type_ == "vignette":
        shadow = np.clip(u * u + v * v, 0.0, 1.0)
    else:
        ang = np.deg2rad(light_dir)
        if direction == "global":
            ang -= np.deg2rad(rot_agl)
        proj = np.cos(ang) * u + np.sin(ang) * v  # +1 toward the light, -1 away from it
        shadow = np.clip((1.0 - proj) / 2.0, 0.0, 1.0)
    factor = 1.0 - amp * shadow**1.3  # power curve: soft falloff, not a hard linear ramp
    return np.repeat(factor[..., None], 3, axis=-1).astype(np.float32)


def make_shader(
    shade: ShadeConfig,
) -> tp.Callable[[np.ndarray, float, float, np.random.Generator], np.ndarray | None]:
    def _shade(
        card_img: np.ndarray,
        rot_agl: float,
        light_dir: float,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        if shade.prob <= 0:
            return None
        return make_card_shading(
            card_img,
            rot_agl,
            light_dir,
            prob=shade.prob,
            amp=shade.amp,
            type_=shade.type,
            direction=shade.direction,
            rng=rng,
        )

    return _shade


def make_shadow_drawer(
    cfg: ShadowConfig,
) -> tp.Callable[[np.ndarray, np.ndarray, tuple[int, int], float, np.random.Generator], None]:
    def _shadow(
        img: np.ndarray,
        alpha: np.ndarray,
        origin: tuple[int, int],
        light_dir: float,
        rng: np.random.Generator,
    ) -> None:
        """Hairline shadow sliver where a not-quite-flat card lifts off the table.

        ``alpha`` is the card's own just-warped alpha channel (canvas-space, top-left at
        ``origin``). Shifting ``alpha`` by a tiny offset and keeping only where that makes
        it *more* opaque than the original gives exactly the region a slightly tilted card
        would newly cover: a sliver along at most two adjacent edges (whichever face the
        offset direction), tapering along each rather than running the full edge length,
        and exactly zero elsewhere because the identical trim/feather on both copies
        cancels out in the subtraction.
        ``offset_px``/``blur_px`` are absolute canvas pixels, not scaled by the card's own
        size, so a card lifted a fraction of a mm stays a pixel or so wide regardless of how
        big that card renders on the board. ``blur_px`` is capped to the offset magnitude so
        it can't spread the sliver past its own width.
        """
        if cfg.prob <= 0 or rng.random() >= cfg.prob:
            return
        canvas_h, canvas_w = img.shape[:2]
        h, w = alpha.shape
        x0, y0 = origin
        ang = np.deg2rad(light_dir)
        offset_mag = float(rng.uniform(*cfg.offset_px))
        dx, dy = np.cos(ang) * offset_mag, np.sin(ang) * offset_mag
        shift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(alpha, shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        reveal = cv2.subtract(shifted, alpha)  # only pixels newly covered by the shift
        blur_r = max(min(cfg.blur_px, offset_mag), 0.15)
        reveal = cv2.GaussianBlur(reveal, (0, 0), blur_r)
        a = (reveal.astype(np.float32) / 255.0) * cfg.strength * rng.uniform(0.7, 1.3)

        yy0, xx0 = max(y0, 0), max(x0, 0)
        yy1, xx1 = min(y0 + h, canvas_h), min(x0 + w, canvas_w)
        if yy1 <= yy0 or xx1 <= xx0:
            return
        a = a[yy0 - y0 : yy1 - y0, xx0 - x0 : xx1 - x0]
        patch = img[yy0:yy1, xx0:xx1].astype(np.float32)
        patch = patch * (1 - a[..., None]) + _SHADOW_COLOR * a[..., None]
        img[yy0:yy1, xx0:xx1] = patch.astype(np.uint8)

    return _shadow


def _make_blob_mask(
    h: int,
    w: int,
    coverage: float,
    rng: np.random.Generator,
    grid: int = 14,
    sigma_factor: float = 0.5,
) -> np.ndarray:
    """Soft 0..1 mask with an irregular (non-polygonal) blob covering ~``coverage`` of the area.

    Thresholding smoothed low-res noise gives a naturally curved boundary; the grid is coarse
    and heavily smoothed relative to its own size (``sigma_factor``) so a level set of it is
    usually one or two large coherent blobs rather than many small speckled ones. ``grid``
    sizes the noise field relative to the *smaller* image dimension, so this behaves the same
    way whether called at whole-canvas or single-card resolution.
    """
    scale = grid / max(min(h, w), 1)
    lh, lw = max(round(h * scale), 12), max(round(w * scale), 12)
    noise = rng.random((lh, lw)).astype(np.float32)
    sigma = max(lh, lw) * sigma_factor
    k = max(int(math.ceil(sigma * 3)) | 1, 3)
    noise = cv2.GaussianBlur(noise, (k, k), sigma)
    thresh = np.quantile(noise, 1.0 - float(np.clip(coverage, 0.02, 0.95)))
    mask = (noise > thresh).astype(np.float32)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


@dataclass(frozen=True)
class HardShadowConfig:
    prob: float = 0.25
    strength: float = 0.45
    coverage: float = 0.2
    edge_softness: float = 0.006


def make_hard_shadow(
    img: np.ndarray,
    prob: float,
    strength: float,
    coverage: float,
    edge_softness: float,
    rng: np.random.Generator,
) -> None:
    """Whole-board hard shadow with an irregular (non-polygonal) edge, in place.

    Mimics an off-scene occluder (a hand, a nearby object) between the light and the
    table: darkens everything underneath it indiscriminately, cards included.
    ``edge_softness`` only blurs a thin transition band, so the irregularity comes
    entirely from ``_make_blob_mask``'s boundary shape, not from a wide blur.
    """
    if rng.random() >= prob:
        return
    canvas_h, canvas_w = img.shape[:2]
    target = float(np.clip(coverage * rng.uniform(0.6, 1.4), 0.02, 0.9))
    mask = _make_blob_mask(canvas_h, canvas_w, target, rng)
    edge_px = max(edge_softness * max(canvas_w, canvas_h), 1.0)
    ek = max(int(math.ceil(edge_px * 3)) | 1, 3)
    mask = cv2.GaussianBlur(mask, (ek, ek), edge_px)
    amount = strength * rng.uniform(0.75, 1.25)
    factor = 1.0 - mask * amount
    img[...] = np.clip(img.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)


def make_hard_shadow_drawer(
    cfg: HardShadowConfig,
) -> tp.Callable[[np.ndarray, np.random.Generator], None]:
    def _draw(img: np.ndarray, rng: np.random.Generator) -> None:
        if cfg.prob <= 0:
            return
        make_hard_shadow(
            img,
            prob=cfg.prob,
            strength=cfg.strength,
            coverage=cfg.coverage,
            edge_softness=cfg.edge_softness,
            rng=rng,
        )

    return _draw


@dataclass(frozen=True)
class CardCoverConfig:
    prob: float = 0.3
    coverage: float = 0.25
    color_range: tuple[int, int] = (20, 235)


def make_card_cover(
    card_img: np.ndarray,
    prob: float,
    coverage: float,
    color_range: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Cover part of a card's printed face with a solid, irregularly-shaped patch.

    Deliberately not aiming for photographic realism: the corner detector must not rely on
    a card's printed content being fully legible to find its corners, and real photos have
    glare/dirt/fingers partially covering a card's face while its corners stay fully visible,
    a case the subtle, realistic card shading doesn't reliably produce on its own. Reuses
    ``_make_blob_mask`` for a cheap, non-polygonal patch shape. Only ever touches RGB, scaled
    by the card's own alpha so the patch can't spill past the card's true cutout silhouette;
    alpha itself, and therefore corner geometry/visibility labels, is untouched.
    """
    if rng.random() >= prob:
        return card_img
    h, w = card_img.shape[:2]
    mask = _make_blob_mask(h, w, coverage * rng.uniform(0.6, 1.4), rng)
    alpha = mask * (card_img[..., 3].astype(np.float32) / 255.0)
    color = float(rng.integers(*color_range))
    out = card_img.copy()
    out[..., :3] = (out[..., :3].astype(np.float32) * (1 - alpha[..., None]) + color * alpha[..., None]).astype(
        np.uint8
    )
    return out


def make_card_coverer(
    cfg: CardCoverConfig,
) -> tp.Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def _cover(card_img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if cfg.prob <= 0:
            return card_img
        return make_card_cover(card_img, cfg.prob, cfg.coverage, cfg.color_range, rng)

    return _cover


@dataclass(frozen=True)
class Effects:
    """The composable effect callables ``make_image`` applies to a board."""

    specular_highlighter: tp.Callable[[np.ndarray, float, np.random.Generator], np.ndarray | None] | None = None
    shader: tp.Callable[[np.ndarray, float, float, np.random.Generator], np.ndarray | None] | None = None
    shadow_drawer: tp.Callable[[np.ndarray, np.ndarray, tuple[int, int], float, np.random.Generator], None] | None = (
        None
    )
    hard_shadow_drawer: tp.Callable[[np.ndarray, np.random.Generator], None] | None = None
    card_blurrer: tp.Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
    bg_blurrer: tp.Callable[[np.ndarray, np.random.Generator], None] | None = None
    defocus_blurrer: tp.Callable[[np.ndarray, np.random.Generator], None] | None = None
    augmenter: tp.Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
    card_coverer: tp.Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None


def make_effects(
    spec: SpecConfig | None = None,
    shade: ShadeConfig | None = None,
    shadow: ShadowConfig | None = None,
    blur: BlurConfig | None = None,
    hard_shadow: HardShadowConfig | None = None,
    card_cover: CardCoverConfig | None = None,
) -> Effects:
    """Build the full effect stack from configs, resolving ``None`` to defaults.

    Callers that only want a subset pass the ``Effects`` fields they need into
    ``make_image`` and ignore the rest.
    """
    spec = spec if spec is not None else SpecConfig()
    shade = shade if shade is not None else ShadeConfig()
    shadow = shadow if shadow is not None else ShadowConfig()
    blur = blur if blur is not None else BlurConfig()
    hard_shadow = hard_shadow if hard_shadow is not None else HardShadowConfig()
    card_cover = card_cover if card_cover is not None else CardCoverConfig()
    return Effects(
        specular_highlighter=make_specular_highlighter(spec),
        shader=make_shader(shade),
        shadow_drawer=make_shadow_drawer(shadow),
        hard_shadow_drawer=make_hard_shadow_drawer(hard_shadow),
        card_blurrer=make_card_blurrer(blur),
        bg_blurrer=make_bg_blurrer(blur),
        defocus_blurrer=make_defocus_blurrer(blur),
        card_coverer=make_card_coverer(card_cover),
    )
