import argparse
import pathlib as pl
import typing as tp

import cv2
import numpy as np

from setdetect.data.card_cutouts import MaskedCard, MaskedCardDataset
from setdetect.data.texturecan import DEFAULT_MANIFEST_NAME, TextureDataset
from setdetect.synth_data.board_synth import (
    CardQuad,
    make_image,
)
from setdetect.synth_data.effects import (
    make_card_coverer,
    make_effects,
    make_hard_shadow_drawer,
    make_shader,
    make_shadow_drawer,
    make_specular_highlight,
)
from setdetect.synth_data.scenarios import SCENARIOS
from setdetect.ui.synth_args import (
    add_blur_args,
    add_card_cover_args,
    add_dataset_args,
    add_hard_shadow_args,
    add_persp_arg,
    add_rot_arg,
    add_shading_args,
    add_shadow_args,
    add_spec_args,
    blur_from_args,
    card_cover_from_args,
    hard_shadow_from_args,
    shading_from_args,
    shadow_from_args,
    spec_from_args,
)

_KEYPOINT_COLORS = [
    (255, 0, 255),  # TL
    (255, 255, 0),  # TR
    (0, 165, 255),  # BR
    (0, 200, 0),  # BL
]
_KEYPOINT_TAGS = ["TL", "TR", "BR", "BL"]


def _draw_keypoints(img: np.ndarray, quads: list[CardQuad]) -> None:
    """Draw the four card corner keypoints, color-coded by corner identity and
    filled vs hollow to mark visibility (matching the label-file flags)."""
    h, w = img.shape[:2]
    for card_quad in quads:
        quad = card_quad.quad
        centroid = quad.mean(axis=0)
        for i, (cx, cy) in enumerate(quad):
            visible = bool(card_quad.visible[i]) and 0 <= cx < w and 0 <= cy < h
            x = int(np.clip(cx, 0, w - 1))
            y = int(np.clip(cy, 0, h - 1))
            color = _KEYPOINT_COLORS[i]
            if visible:
                cv2.circle(img, (x, y), 5, color, thickness=-1)
            else:
                cv2.circle(img, (x, y), 5, color, thickness=2)
                cv2.line(img, (x - 4, y - 4), (x + 4, y + 4), color, 1)
                cv2.line(img, (x - 4, y + 4), (x + 4, y - 4), color, 1)
            offset = np.array([cx, cy]) - centroid
            norm = float(np.linalg.norm(offset))
            if norm > 1e-6:
                offset = offset / norm * 14.0
            else:
                offset = np.array([-14.0, -14.0])
            tx = int(np.clip(x + offset[0], 8, w - 8))
            ty = int(np.clip(y + offset[1], 8, h - 8))
            text_color = color if visible else tuple(int(c * 0.5) for c in color)
            cv2.putText(
                img,
                _KEYPOINT_TAGS[i],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )


def _load_datasets(args: argparse.Namespace) -> tuple[MaskedCardDataset, TextureDataset]:
    manifest_path = args.textures_manifest or (pl.Path(args.textures_root) / DEFAULT_MANIFEST_NAME)
    return MaskedCardDataset(args.cards_root), TextureDataset(args.textures_root, manifest_path)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    parser.add_argument(
        "--num-cards",
        type=int,
        default=12,
        help="Number of cards to place (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--out",
        type=pl.Path,
        default=None,
        help="Save the image to this path instead of showing it",
    )


def _render_scene(
    args: argparse.Namespace,
    card_provider: tp.Callable[[int, np.random.Generator], list[MaskedCard]],
    texture_dataset: TextureDataset,
) -> None:
    rng = np.random.default_rng(args.seed)

    if args.show_labels:
        print("legend: TL=magenta TR=cyan BR=orange BL=green; filled=visible, hollow+cross=occluded/off-frame")

    while True:
        effects = make_effects(
            spec=spec_from_args(args),
            shade=shading_from_args(args),
            shadow=shadow_from_args(args),
            blur=blur_from_args(args),
            hard_shadow=hard_shadow_from_args(args),
            card_cover=card_cover_from_args(args),
        )
        img, quads = make_image(
            card_provider,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            rot_range=(-args.rot_range, args.rot_range),
            rot_outlier_prob=args.rot_outlier_prob,
            rot_outlier_range=(args.rot_outlier_min, args.rot_outlier_max),
            rel_min_separation=args.rel_min_separation,
            persp=args.persp,
            card_highlighter=effects.specular_highlighter,
            card_shader=effects.shader,
            shadow_drawer=effects.shadow_drawer,
            hard_shadow_drawer=effects.hard_shadow_drawer,
            card_blurrer=effects.card_blurrer,
            bg_blurrer=effects.bg_blurrer,
            defocus_blurrer=effects.defocus_blurrer,
            card_coverer=effects.card_coverer,
            rng=rng,
        )
        if args.out is not None or args.show_labels:
            for card_quad in quads:
                cv2.polylines(img, [card_quad.quad.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            if args.show_labels:
                _draw_keypoints(img, quads)
        if args.out is not None:
            cv2.imwrite(str(args.out), img)
            print(f"saved {args.out} with {len(quads)} card quads")
            return
        cv2.imshow("Synthetic Image", img)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break


def _show_board(args: argparse.Namespace) -> None:
    card_dataset, texture_dataset = _load_datasets(args)
    _render_scene(args, card_dataset.random_cards, texture_dataset)


def _show_scenario_board(args: argparse.Namespace) -> None:
    card_dataset, texture_dataset = _load_datasets(args)
    card_provider = SCENARIOS[args.scenario](card_dataset)
    _render_scene(args, card_provider, texture_dataset)


def _show_specular(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    card_dataset, texture_dataset = _load_datasets(args)
    spec = spec_from_args(args)

    while True:
        # one coherent light direction per board so the reflected scene is reproducible
        direction = args.spec_direction
        if direction is None:
            direction = float(rng.uniform(0, 180))

        def specular_highlighter(
            card_img: np.ndarray, rot_agl: float, hl_rng: np.random.Generator
        ) -> np.ndarray | None:
            return make_specular_highlight(
                card_img,
                rot_agl,
                rng=hl_rng,
                prob=spec.prob,
                intensity=spec.intensity,
                size=spec.size,
                streak=spec.streak,
                direction=direction,
                align=spec.align,
            )

        # same scene twice (independent specular-highlighter RNG keeps the shared
        # stream aligned), once plain and once with per-card specular highlights
        state = rng.bit_generator.state
        plain, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            rng=rng,
        )
        rng.bit_generator.state = state
        reflected, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            card_highlighter=specular_highlighter,
            rng=rng,
        )

        side_by_side = np.hstack([plain, reflected])
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.out), side_by_side):
                raise SystemExit(f"failed to write {args.out}")
            print(
                f"saved {args.out} (left: plain, right: specular); "
                f"params: prob={args.spec_prob} intensity={args.spec_intensity} size={args.spec_size} "
                f"streak={args.spec_streak} direction={direction:.1f} align={args.spec_align}"
            )
            return
        cv2.imshow("Specular Highlight (left plain, right specular)", side_by_side)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break


def _show_shading(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    card_dataset, texture_dataset = _load_datasets(args)
    shade = shading_from_args(args)
    shadow = shadow_from_args(args)

    while True:
        # same scene twice (net-zero highlight/shade RNG keeps the shared stream
        # aligned), once plain and once with per-card gradients + drop shadows
        state = rng.bit_generator.state
        plain, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            rng=rng,
        )
        rng.bit_generator.state = state
        shaded, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            card_shader=make_shader(shade),
            shadow_drawer=make_shadow_drawer(shadow),
            rng=rng,
        )

        side_by_side = np.hstack([plain, shaded])
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.out), side_by_side):
                raise SystemExit(f"failed to write {args.out}")
            print(
                f"saved {args.out} (left: plain, right: gradients + shadows); "
                f"params: shading_prob={args.shading_prob} shading_amp={args.shading_amp} "
                f"shading_type={args.shading_type} "
                f"shadow_prob={args.shadow_prob} shadow_strength={args.shadow_strength} "
                f"shadow_offset={args.shadow_offset_min}-{args.shadow_offset_max} "
                f"shadow_blur={args.shadow_blur}"
            )
            return
        cv2.imshow("Card Gradients + Shadows (left plain, right shaded)", side_by_side)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break


def _show_hard_shadow(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    card_dataset, texture_dataset = _load_datasets(args)
    cfg = hard_shadow_from_args(args)

    while True:
        # same scene twice (net-zero effects RNG keeps the shared stream aligned),
        # once plain and once with the whole-board hard shadow
        state = rng.bit_generator.state
        plain, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            rng=rng,
        )
        rng.bit_generator.state = state
        shadowed, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            hard_shadow_drawer=make_hard_shadow_drawer(cfg),
            rng=rng,
        )

        side_by_side = np.hstack([plain, shadowed])
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.out), side_by_side):
                raise SystemExit(f"failed to write {args.out}")
            print(
                f"saved {args.out} (left: plain, right: hard shadow); "
                f"params: hard_shadow_prob={args.hard_shadow_prob} "
                f"hard_shadow_strength={args.hard_shadow_strength} "
                f"hard_shadow_coverage={args.hard_shadow_coverage} "
                f"hard_shadow_edge_softness={args.hard_shadow_edge_softness}"
            )
            return
        cv2.imshow("Hard Shadow (left plain, right shadowed)", side_by_side)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break


def _show_card_cover(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    card_dataset, texture_dataset = _load_datasets(args)
    cfg = card_cover_from_args(args)

    while True:
        # same scene twice (net-zero effects RNG keeps the shared stream aligned),
        # once plain and once with per-card content-covering patches
        state = rng.bit_generator.state
        plain, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            rng=rng,
        )
        rng.bit_generator.state = state
        covered, _ = make_image(
            card_dataset.random_cards,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            card_coverer=make_card_coverer(cfg),
            rng=rng,
        )

        side_by_side = np.hstack([plain, covered])
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.out), side_by_side):
                raise SystemExit(f"failed to write {args.out}")
            print(
                f"saved {args.out} (left: plain, right: card cover); "
                f"params: card_cover_prob={args.card_cover_prob} "
                f"card_cover_coverage={args.card_cover_coverage}"
            )
            return
        cv2.imshow("Card Cover (left plain, right covered)", side_by_side)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-board-synth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_p = subparsers.add_parser(
        "show-board",
        help="Show a random synthetic board image with card corner annotations",
    )
    _add_common_args(show_p)
    add_rot_arg(show_p)
    add_persp_arg(show_p)
    add_spec_args(show_p)
    add_shading_args(show_p)
    add_shadow_args(show_p)
    add_hard_shadow_args(show_p)
    add_card_cover_args(show_p)
    add_blur_args(show_p)
    show_p.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw the four card corner keypoints (TL/TR/BR/BL) color-coded by "
        "corner, filled when visible and hollow with a cross when occluded",
    )

    scenario_p = subparsers.add_parser(
        "show-scenario-board",
        help="Show a crafted example board illustrating a specific scenario (e.g. a duplicate card)",
    )
    _add_common_args(scenario_p)
    scenario_p.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        required=True,
        help="Which crafted scenario to render",
    )
    add_rot_arg(scenario_p)
    add_persp_arg(scenario_p)
    add_spec_args(scenario_p)
    add_shading_args(scenario_p)
    add_shadow_args(scenario_p)
    add_hard_shadow_args(scenario_p)
    add_card_cover_args(scenario_p)
    add_blur_args(scenario_p)
    scenario_p.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw the four card corner keypoints (TL/TR/BR/BL) color-coded by "
        "corner, filled when visible and hollow with a cross when occluded",
    )

    spec_p = subparsers.add_parser(
        "show-specular",
        help="Show a random board with per-card specular highlights (plain vs. specular)",
    )
    _add_common_args(spec_p)
    add_spec_args(spec_p)

    shading_p = subparsers.add_parser(
        "show-shading",
        help="Show a random board with per-card brightness gradients and dropped shadows (plain vs. shaded)",
    )
    _add_common_args(shading_p)
    add_shading_args(shading_p)
    add_shadow_args(shading_p)

    hard_shadow_p = subparsers.add_parser(
        "show-hard-shadow",
        help="Show a random board with a whole-board irregular hard shadow (plain vs. shadowed)",
    )
    _add_common_args(hard_shadow_p)
    add_hard_shadow_args(hard_shadow_p)

    card_cover_p = subparsers.add_parser(
        "show-card-cover",
        help="Show a random board with per-card content-covering patches (plain vs. covered)",
    )
    _add_common_args(card_cover_p)
    add_card_cover_args(card_cover_p)

    args = parser.parse_args(argv)
    if args.command == "show-board":
        _show_board(args)
    elif args.command == "show-scenario-board":
        _show_scenario_board(args)
    elif args.command == "show-specular":
        _show_specular(args)
    elif args.command == "show-shading":
        _show_shading(args)
    elif args.command == "show-hard-shadow":
        _show_hard_shadow(args)
    elif args.command == "show-card-cover":
        _show_card_cover(args)


if __name__ == "__main__":
    main()
