import math
import typing as tp

import cv2
import jaxtyping as jt
import numpy as np

from setdetect.data.card_cutouts import MaskedCard
from setdetect.set_game.game import Card

Quad = jt.Float[np.ndarray, "4 2"]
Img = jt.Float[np.ndarray, "H W 3"]


class CardQuad(Card):
    def __init__(self, card: Card, quad: Quad, visible: np.ndarray | None = None) -> None:
        super().__init__(card.count, card.color, card.shape, card.fill)
        self.quad = quad
        self.visible = visible if visible is not None else np.ones(4, dtype=bool)


def _cover(image: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    scale = max(w / iw, h / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))
    resized = cv2.resize(image, (nw, nh))
    x0 = (nw - w) // 2
    y0 = (nh - h) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def _card_support(rot_agl: float, half_w: float, half_h: float, direction: np.ndarray) -> float:
    """Half-length of a card's projection onto ``direction`` (a unit vector).

    The card is an axis-aligned ``half_w``x``half_h`` rectangle rotated by ``rot_agl``;
    projecting its two rotated edge axes onto ``direction`` gives the exact support
    (max extent) of the rectangle's projection onto that direction -- used by
    ``_relax_positions`` to run the separating-axis test between two cards.
    """
    ux = np.array([math.cos(rot_agl), math.sin(rot_agl)])
    uy = np.array([-math.sin(rot_agl), math.cos(rot_agl)])
    return half_w * abs(direction @ ux) + half_h * abs(direction @ uy)


def _relax_positions(
    xs: list[float],
    ys: list[float],
    rot_angles: list[float],
    half_w: float,
    half_h: float,
    rel_min_separation: float,
    rng: np.random.Generator,
    iterations: int = 4,
) -> None:
    """Push apart any pair of cards that overlap more than ``rel_min_separation`` allows, in place.

    Two rotated rectangles are separated iff their centers' projection onto *some* axis
    exceeds the two rectangles' combined support on that axis, and by the separating axis
    theorem the only candidate axes that can possibly witness that are each rectangle's own
    two edge normals (four candidates per pair here, since both cards share ``half_w``/
    ``half_h``) -- the line connecting the two centers is *not* generally a valid separating
    axis, so it can't be used as a shortcut. For each pair, if every candidate axis's
    ``rel_min_separation``-scaled combined support still exceeds the actual center-projection
    on that axis, the pair is pushed apart along whichever axis needs the smallest
    correction (the standard minimum-translation-vector choice). ``rel_min_separation`` of
    ``1.0`` means no overlap at all; a smaller value still allows the mild edge overlap
    ``rel_card_overscale`` intentionally introduces between grid neighbors. Repeated for a
    small fixed number of passes: O(n^2) per pass, trivial for board-sized n (8-12). The
    iteration count is fixed so this can never loop unboundedly, even though full convergence
    isn't guaranteed for a pathologically over-packed board.
    """
    n = len(xs)
    if n < 2 or rel_min_separation <= 0:
        return
    # each rectangle's own two edge-normal directions are the only axes SAT needs to check
    axes = [(np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])) for a in rot_angles]
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = xs[j] - xs[i], ys[j] - ys[i]
                if dx * dx + dy * dy < 1e-12:
                    theta = rng.uniform(0, 2 * math.pi)
                    dx, dy = math.cos(theta) * 1e-3, math.sin(theta) * 1e-3
                d = np.array([dx, dy])

                best_shortfall = None
                best_axis: np.ndarray | None = None
                for axis in (*axes[i], *axes[j]):
                    support = _card_support(rot_angles[i], half_w, half_h, axis) + _card_support(
                        rot_angles[j], half_w, half_h, axis
                    )
                    shortfall = rel_min_separation * support - abs(d @ axis)
                    if shortfall <= 0:
                        best_shortfall = None
                        break
                    if best_shortfall is None or shortfall < best_shortfall:
                        best_shortfall = shortfall
                        best_axis = axis
                if best_shortfall is None or best_axis is None:
                    continue

                moved = True
                sign = 1.0 if (d @ best_axis) >= 0 else -1.0
                push = best_axis * (sign * best_shortfall / 2.0)
                xs[i] -= push[0]
                ys[i] -= push[1]
                xs[j] += push[0]
                ys[j] += push[1]
        if not moved:
            break


def make_image(
    card_provider: tp.Callable[[int, np.random.Generator], list[MaskedCard]],
    texture_provider: tp.Callable[[np.random.Generator], np.ndarray],
    num_cards: int = 12,
    res: tuple[int, int] = (640, 640),
    grid_size: tuple[int, int] = (3, 4),
    p_landscape: float = 0.5,
    rel_img_pad_range: tuple[float, float] = (0.05, 0.15),
    rel_cell_pad_range: tuple[float, float] = (0.02, 0.06),
    rel_cell_jitter: float = 0.15,
    rel_card_overscale: tuple[float, float] = (0.95, 1.05),
    rel_min_separation: float = 0.85,
    rot_range: tuple[float, float] | np.ndarray = (-45, 45),
    rot_outlier_prob: float = 0.15,
    rot_outlier_range: tuple[float, float] = (60, 180),
    rel_crop_jitter: float = 0.05,
    persp: float = 1.25,
    card_highlighter: tp.Callable[[np.ndarray, float, np.random.Generator], np.ndarray | None] | None = None,
    card_shader: tp.Callable[[np.ndarray, float, float, np.random.Generator], np.ndarray | None] | None = None,
    shadow_drawer: tp.Callable[[np.ndarray, np.ndarray, tuple[int, int], float, np.random.Generator], None]
    | None = None,
    hard_shadow_drawer: tp.Callable[[np.ndarray, np.random.Generator], None] | None = None,
    card_blurrer: tp.Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    bg_blurrer: tp.Callable[[np.ndarray, np.random.Generator], None] | None = None,
    defocus_blurrer: tp.Callable[[np.ndarray, np.random.Generator], None] | None = None,
    augmenter: tp.Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    card_coverer: tp.Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Img, list[CardQuad]]:
    """Composite a synthetic board: perspective-warped cards over a texture.

    ``rot_range`` is ``(min, max)`` degrees around the card orientation (default ``(-45, 45)``)
    and ``res`` is ``(height, width)``. Effects are composable: pass only the ones you want
    (each defaults to disabled).

    Most cards draw their rotation from ``rot_range``, but each card independently has a
    ``rot_outlier_prob`` chance of instead becoming a rotation outlier: a card rotated by a
    guaranteed-large ``rot_outlier_range`` magnitude (either direction). This mimics a card
    dropped at a much different angle than the rest of the board (e.g. one card
    near-perpendicular to 11 roughly-aligned others), a pattern real photos have but a single
    flat rotation range never produces. ``rel_min_separation`` then caps how much any two cards
    (however jitter and/or rotation happened to land them) are allowed to overlap, via a
    separating-axis check between each pair of (rotated) card rectangles: ``1.0`` means no
    overlap at all; the default leaves room for the mild edge overlap ``rel_card_overscale``
    intentionally introduces between grid neighbors. See ``_relax_positions``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    img_h, img_w = res
    rot_range = np.deg2rad(np.asarray(rot_range, dtype=float))
    rot_outlier_range_rad = np.deg2rad(np.asarray(rot_outlier_range, dtype=float))

    # per-image RNG for effects, spawned from the shared stream so effect draws
    # never perturb the scene geometry
    hl_rng = rng.spawn(1)[0]
    light_dir: float = 0.0
    if card_shader is not None or shadow_drawer is not None:
        light_dir = float(hl_rng.uniform(0, 360))

    # random global perspective transform, centered on the padded canvas. persp is
    # the projective cap (1/(px*W+py*H) scale); a floor keeps some perspective on
    # every board instead of letting it vanish on most of them
    hi = min(1.2e-3, persp / (img_w + img_h))
    px, py = rng.choice([-1.0, 1.0], size=2) * rng.uniform(0.2 * hi, hi)

    # coverage margin so the perspective warp never exposes the canvas edge.
    # the projective denominator couples px and py: u = |px|*W/2 + |py|*H/2,
    # and the compressed window corner needs source out to (W/2)/(1-u).
    u = abs(px) * (img_w / 2) + abs(py) * (img_h / 2)
    m_x = math.ceil((img_w / 2) * u / (1 - u)) + 1
    m_y = math.ceil((img_h / 2) * u / (1 - u)) + 1
    canvas_w, canvas_h = img_w + 2 * m_x, img_h + 2 * m_y

    cx, cy = canvas_w / 2, canvas_h / 2
    hom = (
        np.array(
            [
                [1, 0, cx],
                [0, 1, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        @ np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [px, py, 1],
            ],
            dtype=np.float32,
        )
        @ np.array(
            [
                [1, 0, -cx],
                [0, 1, -cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    )

    # Get texture from provider and cover the padded canvas
    texture = texture_provider(rng)
    img = _cover(texture, canvas_w, canvas_h)
    if bg_blurrer is not None:
        bg_blurrer(img, hl_rng)

    # load cards
    cards = card_provider(num_cards, rng)
    card_imgs: list[np.ndarray] = []
    for card in cards:
        card_imgs.append(card.image)

    # whole-board landscape arrangement: rotate every card by ±90 deg and lay the
    # grid out wide instead of tall. corners are copies; card.corners is shared
    # per identity across calls, so it must not be mutated.
    landscape = rng.random() < p_landscape
    if landscape:
        spin = int(rng.choice([1, 3]))
        orig_h, orig_w = card_imgs[0].shape[:2]
        card_corners: list[np.ndarray] = []
        for i, card in enumerate(cards):
            card_imgs[i] = np.ascontiguousarray(np.rot90(card_imgs[i], spin))
            corners = card.corners
            if spin == 1:
                corners = np.stack([corners[:, 1], orig_w - 1 - corners[:, 0]], axis=1)
            else:
                corners = np.stack([orig_h - 1 - corners[:, 1], corners[:, 0]], axis=1)
            card_corners.append(corners)
    else:
        card_corners = [card.corners for card in cards]

    # positioning grid
    grid_cols, grid_rows = grid_size[::-1] if landscape else grid_size
    img_pad = rng.uniform(*rel_img_pad_range) * min(res)
    # random per-gap spacing so boards don't look machine-made
    col_gaps = rng.uniform(*rel_cell_pad_range, size=grid_cols - 1) * min(res)
    row_gaps = rng.uniform(*rel_cell_pad_range, size=grid_rows - 1) * min(res)
    cell_w = (img_w - 2 * img_pad - col_gaps.sum()) / grid_cols
    cell_h = (img_h - 2 * img_pad - row_gaps.sum()) / grid_rows

    # keep cells at the card's aspect ratio so cards fit without strong overlap
    card_h, card_w = card_imgs[0].shape[:2]
    aspect = card_h / card_w
    cell_h = min(cell_h, cell_w * aspect)
    cell_w = cell_h / aspect

    # positioning order
    positions = [(row, col) for row in range(grid_rows) for col in range(grid_cols)]
    rng.shuffle(positions)

    # scale: card fills its cell, slight overscale adds natural overlap
    scale_fac = (cell_w / card_w) * rng.uniform(*rel_card_overscale)

    n_place = min(len(cards), len(positions))

    # Pass 1: sample every placed card's pose (position + rotation) with no image
    # work yet, so overlap relaxation below can see every card's final center
    # before anything is composited.
    xs: list[float] = []
    ys: list[float] = []
    rot_angles: list[float] = []
    for k in range(n_place):
        row, col = positions[k]
        # cell center (the transform below pivots each card's rotation/scale about
        # its own center, so this is also where that center ends up)
        x = m_x + img_pad + col * cell_w + (col_gaps[:col].sum() if col else 0.0) + cell_w / 2
        y = m_y + img_pad + row * cell_h + (row_gaps[:row].sum() if row else 0.0) + cell_h / 2
        x += rng.uniform(-rel_cell_jitter, rel_cell_jitter) * cell_w
        y += rng.uniform(-rel_cell_jitter, rel_cell_jitter) * cell_h

        # rotation outliers: a minority of cards get a guaranteed-large rotation
        # (mimicking a carelessly-placed card), instead of every card drawing from
        # the same narrow, uniformly "neat" range
        if rng.random() < rot_outlier_prob:
            rot_agl = rng.choice([-1.0, 1.0]) * rng.uniform(*rot_outlier_range_rad)
        else:
            rot_agl = rng.uniform(*rot_range)

        xs.append(x)
        ys.append(y)
        rot_angles.append(rot_agl)

    # Bound overlap directionally: every card shares scale_fac, so half_w/half_h are
    # the same for every card regardless of its own rotation. rel_min_separation < 1
    # still allows the mild neighbor overlap rel_card_overscale intends.
    half_w, half_h = 0.5 * scale_fac * card_w, 0.5 * scale_fac * card_h
    _relax_positions(xs, ys, rot_angles, half_w, half_h, rel_min_separation, rng)

    # Pass 2: composite each card using the pose finalized above.
    card_quads: list[CardQuad] = []
    owner = np.full((canvas_h, canvas_w), -1, dtype=np.int32)
    for k in range(n_place):
        card, card_img = cards[k], card_imgs[k]
        x, y, rot_agl = xs[k], ys[k], rot_angles[k]

        if augmenter is not None:
            card_img = augmenter(card_img, hl_rng)

        # per-card lighting folded into the card before warping so it is clipped
        # by the card alpha and respects compositing order: a broad brightness
        # gradient (multiplicative) then a specular highlight (additive)
        factor = card_shader(card_img, rot_agl, light_dir, hl_rng) if card_shader is not None else None
        hl = card_highlighter(card_img, rot_agl, hl_rng) if card_highlighter is not None else None
        if factor is not None or hl is not None:
            hh, ww = card_img.shape[:2]
            bgrab = card_img.copy()
            if factor is not None:
                fac8 = cv2.resize(
                    np.rint(factor * 255.0).astype(np.uint8),
                    (ww, hh),
                    interpolation=cv2.INTER_LINEAR,
                )
                fac4 = np.empty((hh, ww, 4), np.uint8)
                fac4[..., :3] = fac8
                fac4[..., 3] = 255
                cv2.multiply(bgrab, fac4, dst=bgrab, scale=1.0 / 255.0)
            if hl is not None:
                hl8 = cv2.resize(
                    np.rint(hl * 255.0).astype(np.uint8),
                    (ww, hh),
                    interpolation=cv2.INTER_LINEAR,
                )
                hl4 = np.empty((hh, ww, 4), np.uint8)
                hl4[..., :3] = hl8[..., None]
                hl4[..., 3] = 0
                cv2.add(bgrab, hl4, dst=bgrab)
            card_img = bgrab

        # per-card defocus blur after the lighting fold (so highlight/shade blur
        # with the card) and before warping; alpha restored so the cutout stays sharp
        if card_blurrer is not None:
            card_img = card_blurrer(card_img, hl_rng)

        # deliberate content-hiding patch, last so it survives regardless of what the
        # other (randomly-triggered) effects above did; only ever touches RGB, so it
        # can't affect corner geometry/visibility
        if card_coverer is not None:
            card_img = card_coverer(card_img, hl_rng)

        # compose transformation: rotate and scale about the card's own center (so x,y
        # is where that center lands, independent of rot_agl), then translate there.
        # For an unrotated corner-pivot transform ``dst = scale*R@src + (x,y)``, pivoting
        # about the center instead means subtracting the center's own transformed offset
        # from the translation: ``dst = scale*R@(src - center) + (x,y)``.
        h, w = card_img.shape[:2]
        cos_a, sin_a = np.cos(rot_agl), np.sin(rot_agl)
        center_x, center_y = w / 2.0, h / 2.0
        tx = x - scale_fac * (cos_a * center_x - sin_a * center_y)
        ty = y - scale_fac * (sin_a * center_x + cos_a * center_y)
        transform = np.array(
            [
                [scale_fac * cos_a, -scale_fac * sin_a, tx],
                [scale_fac * sin_a, scale_fac * cos_a, ty],
            ],
            dtype=np.float32,
        )

        # destination bbox of the card on the padded canvas; blend only there instead
        # of round-tripping the whole canvas through float32 per card (the measured
        # generation bottleneck)
        box_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pts = cv2.transform(box_corners[None], transform)[0]
        x0 = int(np.floor(pts[:, 0].min())) - 1
        y0 = int(np.floor(pts[:, 1].min())) - 1
        x1 = int(np.ceil(pts[:, 0].max())) + 1
        y1 = int(np.ceil(pts[:, 1].max())) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, canvas_w), min(y1, canvas_h)

        # place image with alpha blending so the transparent background vanishes
        if x1 > x0 and y1 > y0:
            sub = transform.copy()
            sub[0, 2] -= x0
            sub[1, 2] -= y0
            warped = cv2.warpAffine(
                card_img,
                sub,
                (x1 - x0, y1 - y0),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            # hairline shadow where a not-quite-flat card lifts off the table, drawn
            # before the card so it layers behind it. Uses the card's own just-warped
            # alpha (computed above) rather than its bounding box, so the trimmed,
            # feathered silhouette cancels out in the shift-vs-original comparison
            # inside shadow_drawer instead of always showing along the padding border.
            if shadow_drawer is not None:
                shadow_drawer(img, warped[..., 3], (x0, y0), light_dir, hl_rng)

            alpha = warped[..., 3:4].astype(np.float32) / 255.0
            patch = img[y0:y1, x0:x1].astype(np.float32)
            patch = patch * (1 - alpha) + warped[..., :3].astype(np.float32) * alpha
            img[y0:y1, x0:x1] = patch.astype(np.uint8)
            # record which card is topmost per pixel (draw order = compositing order)
            owner[y0:y1, x0:x1][alpha[..., 0] > 0] = k

        # compute quad (padded canvas coordinates) from the card's image-space corners
        corners = card_corners[k].astype(np.float32)
        quad = cv2.transform(corners[None], transform)[0]
        card_quads.append(CardQuad(card, quad))

    # per-corner occlusion: a corner is hidden if a later-drawn card covers any pixel
    # in a small window around it (owner > k). Occlusion survives the global warp below.
    for k, card_quad in enumerate(card_quads):
        visible = np.ones(4, dtype=bool)
        for i, (cx, cy) in enumerate(card_quad.quad):
            x0 = max(int(cx) - 3, 0)
            y0 = max(int(cy) - 3, 0)
            x1 = min(int(cx) + 4, canvas_w)
            y1 = min(int(cy) + 4, canvas_h)
            visible[i] = not np.any(owner[y0:y1, x0:x1] > k)
        card_quad.visible = visible

    # whole-board hard shadow (an off-scene occluder, not any one card's own shape):
    # drawn after every card so it darkens cards too, and before the warp so it gets
    # carried through the same perspective transform as the rest of the scene
    if hard_shadow_drawer is not None:
        hard_shadow_drawer(img, hl_rng)

    img = cv2.warpPerspective(img, hom, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # warp the card quads into canvas coords first; the nonlinear projective warp
    # is not symmetric, so recentering the crop on the warped card bbox (instead of
    # the canvas center) keeps the board filling the frame instead of drifting
    # toward one side
    for card_quad in card_quads:
        card_quad.quad = cv2.perspectiveTransform(card_quad.quad.reshape(4, 1, 2), hom).reshape(4, 2)

    if card_quads:
        # center the crop on the area-weighted content centroid: a projective warp
        # skews the corner-bbox center away from where the pixels actually land
        num = np.zeros(2)
        den = 0.0
        for card_quad in card_quads:
            x, y = card_quad.quad[:, 0], card_quad.quad[:, 1]
            shoe = x * np.roll(y, -1) - np.roll(x, -1) * y
            area2 = np.sum(shoe)
            num += np.stack(
                [
                    np.sum((np.roll(x, -1) + x) * shoe),
                    np.sum((np.roll(y, -1) + y) * shoe),
                ]
            )
            den += area2
        content_x, content_y = num / (3.0 * den)
        jitter = rng.uniform(-rel_crop_jitter, rel_crop_jitter, size=2) * min(res)
        crop_x = int(np.clip(content_x - img_w / 2 + jitter[0], 0, canvas_w - img_w))
        crop_y = int(np.clip(content_y - img_h / 2 + jitter[1], 0, canvas_h - img_h))
    else:
        crop_x, crop_y = m_x, m_y
    img = img[crop_y : crop_y + img_h, crop_x : crop_x + img_w]

    for card_quad in card_quads:
        card_quad.quad = card_quad.quad - np.array([crop_x, crop_y], dtype=np.float32)

    _canonicalize_corner_order(card_quads)

    # whole-frame defocus after the crop: labels/geometry are computed above, and
    # the draw comes from the net-zero hl_rng, so defocus boards stay deterministic
    if defocus_blurrer is not None:
        defocus_blurrer(img, hl_rng)

    return img, card_quads


_UP = np.array([0.0, -1.0])
_RIGHT = np.array([1.0, 0.0])


def _canonicalize_corner_order(card_quads: list[CardQuad]) -> None:
    """Resolve the 180deg label ambiguity: SET cards are rotationally symmetric,
    so a spin-flipped placement looks identical but would swap TL<->BR and
    TR<->BL. Force a convention: portrait cards get TR pointing up, landscape
    cards get TR pointing left, so identical appearances always carry identical
    labels."""
    for card_quad in card_quads:
        q = card_quad.quad
        v = (q[0] + q[1] - q[2] - q[3]) / 2.0
        portrait = abs(v @ _UP) >= abs(v @ _RIGHT)
        flip = (v @ _UP < 0) if portrait else (v @ _RIGHT > 0)
        if flip:
            card_quad.quad = q[[2, 3, 0, 1]]
            card_quad.visible = card_quad.visible[[2, 3, 0, 1]]
