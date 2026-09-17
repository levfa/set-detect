import argparse
import dataclasses
import pathlib as pl

import cv2
import numpy as np

from setdetect.data.card_cutouts import MaskedCardDataset
from setdetect.data.texturecan import DEFAULT_MANIFEST_NAME, TextureDataset
from setdetect.synth_data.card_synth import (
    KIND_CARD,
    KIND_EDGE,
    KIND_FAKE,
    KIND_ROT,
    BoardQuad,
    add_false_positive_quads,
    extract_cards,
    jitter_quads,
    make_board,
    make_card_augmenter,
)
from setdetect.synth_data.effects import Effects, make_effects
from setdetect.ui.synth_args import (
    add_blur_args,
    add_card_aug_args,
    add_card_cover_args,
    add_dataset_args,
    add_hard_shadow_args,
    add_shading_args,
    add_shadow_args,
    add_spec_args,
    blur_from_args,
    card_aug_from_args,
    card_cover_from_args,
    hard_shadow_from_args,
    shading_from_args,
    shadow_from_args,
    spec_from_args,
)

_KIND_COLORS = {
    KIND_CARD: (0, 200, 0),
    KIND_FAKE: (0, 0, 255),
    KIND_ROT: (0, 255, 255),
    KIND_EDGE: (255, 0, 255),
}


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.split(",")
    return int(w), int(h)


def _effects_from_args(args: argparse.Namespace) -> Effects:
    if args.no_aug:
        return Effects()
    effects = make_effects(
        spec=spec_from_args(args),
        shade=shading_from_args(args),
        shadow=shadow_from_args(args),
        blur=blur_from_args(args),
        hard_shadow=hard_shadow_from_args(args),
        card_cover=card_cover_from_args(args),
    )
    return dataclasses.replace(effects, augmenter=make_card_augmenter(card_aug_from_args(args)))


def _status_line(crop_index: int, total: int, quads: list[BoardQuad]) -> str:
    bq = quads[crop_index]
    label_text = " ".join(bq.label) if bq.label is not None else "N/A"
    return f"card {crop_index + 1}/{total} kind={bq.kind} label={label_text}"


def _show_board(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    manifest_path = args.textures_manifest or (pl.Path(args.textures_root) / DEFAULT_MANIFEST_NAME)
    card_dataset = MaskedCardDataset(args.cards_root)
    texture_dataset = TextureDataset(args.textures_root, manifest_path)
    out_w, out_h = args.crop_size

    while True:
        img, quads = make_board(
            card_dataset,
            texture_dataset.random_texture,
            num_cards=args.num_cards,
            num_fake=args.num_fake,
            effects=_effects_from_args(args),
            min_vis=args.min_vis,
            rng=rng,
        )
        add_false_positive_quads(quads, args.num_rot, args.num_edge, rng=rng)
        jitter_quads(quads, args.jitter, rng=rng)
        crops = extract_cards(img, quads, (out_w, out_h))

        annotated = img.copy()
        for bq in quads:
            cv2.polylines(
                annotated,
                [bq.quad.astype(np.int32)],
                isClosed=True,
                color=_KIND_COLORS[bq.kind],
                thickness=2,
            )

        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.out), annotated):
                raise SystemExit(f"failed to write {args.out}")
            counts = {kind: sum(1 for bq in quads if bq.kind == kind) for kind in _KIND_COLORS}
            print(f"saved {args.out}: {counts}")
            return

        cv2.imshow("Synthetic Board", annotated)
        idx = 0
        while True:
            print(_status_line(idx, len(crops), quads))
            display = crops[idx].image.copy()
            cv2.putText(
                display,
                _status_line(idx, len(crops), quads),
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Extracted Card", display)
            key = cv2.waitKey(-1)
            if key in (ord(" "), ord("n")):
                if idx == len(crops) - 1:
                    break
                idx += 1
            elif key == ord("p"):
                idx = (idx - 1) % len(crops)
            elif key == ord("r"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-synth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser(
        "show-board",
        help="Show a synthetic board of real + fake cards and step through extracted card crops",
    )
    add_dataset_args(show_parser)
    show_parser.add_argument(
        "--num-cards",
        type=int,
        default=8,
        help="Number of real cards (default: %(default)s)",
    )
    show_parser.add_argument(
        "--num-fake",
        type=int,
        default=4,
        help="Number of fake card-like false positives (default: %(default)s)",
    )
    show_parser.add_argument(
        "--jitter",
        type=float,
        default=0.06,
        help="Max per-corner quad fluctuation as a fraction of the quad's own edge length; "
        "25%% of corners draw up to 3x this (default: %(default)s)",
    )
    show_parser.add_argument(
        "--min-vis",
        type=float,
        default=0.7,
        help="Drop cards with less of their area inside the frame than this (default: %(default)s)",
    )
    show_parser.add_argument(
        "--num-rot",
        type=int,
        default=3,
        help="Number of rotated-card false-positive quads (default: %(default)s)",
    )
    show_parser.add_argument(
        "--num-edge",
        type=int,
        default=3,
        help="Number of card-edge-band false-positive quads (default: %(default)s)",
    )
    show_parser.add_argument(
        "--crop-size",
        type=_parse_size,
        dest="crop_size",
        default=(224, 320),
        metavar="W,H",
        help="Extracted crop size in px (default: %(default)s)",
    )
    show_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    add_spec_args(show_parser)
    add_shading_args(show_parser)
    add_shadow_args(show_parser)
    add_hard_shadow_args(show_parser)
    add_card_cover_args(show_parser)
    add_card_aug_args(show_parser)
    add_blur_args(show_parser)
    show_parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable all per-card and whole-board augmentation (photometric, shading, "
        "shadows, specular, blur, card-cover)",
    )
    show_parser.add_argument(
        "--out",
        type=pl.Path,
        default=None,
        help="Save the annotated board to this path instead of showing it",
    )

    args = parser.parse_args(argv)
    if args.command == "show-board":
        _show_board(args)


if __name__ == "__main__":
    main()
