import argparse
import pathlib as pl

from setdetect.synth_data.card_synth import CardAugConfig
from setdetect.synth_data.effects import (
    BlurConfig,
    CardCoverConfig,
    HardShadowConfig,
    ShadeConfig,
    ShadowConfig,
    SpecConfig,
)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cards-root",
        type=pl.Path,
        required=True,
        help="Directory containing the masked card-cutout dataset (raw/ + masked/).",
    )
    parser.add_argument(
        "--textures-root",
        type=pl.Path,
        required=True,
        help="Directory containing the texture dataset (e.g. your local huggingface/nyuuzyou/texturecan mirror)",
    )
    parser.add_argument(
        "--textures-manifest",
        type=pl.Path,
        default=None,
        help="Path to the background manifest (default: <textures-root>/.backgrounds.jsonl)",
    )


def add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spec-prob",
        type=float,
        default=0.8,
        help="Probability a card catches a specular highlight; 0 disables highlights (default: %(default)s)",
    )
    parser.add_argument(
        "--spec-intensity",
        type=float,
        default=0.5,
        help="Peak added brightness, 0..1 (default: %(default)s)",
    )
    parser.add_argument(
        "--spec-size",
        type=float,
        default=0.30,
        help="Specular highlight length as fraction of card width (default: %(default)s)",
    )
    parser.add_argument(
        "--spec-streak",
        type=float,
        default=2.5,
        help="Specular highlight length/width elongation; 1.0 is a round glow (default: %(default)s)",
    )
    parser.add_argument(
        "--spec-direction",
        type=float,
        default=None,
        help="Specular highlight streak direction in degrees; default random per card",
    )
    parser.add_argument(
        "--spec-align",
        type=str,
        choices=["global", "card"],
        default="global",
        help="global: same world angle on every card; card: rotates with each card (default: %(default)s)",
    )


def add_shading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shading-prob",
        type=float,
        default=0.8,
        help="Probability a card gets a soft shadow gradient; 0 disables shading (default: %(default)s)",
    )
    parser.add_argument(
        "--shading-amp",
        type=float,
        default=0.18,
        help="Peak shadow depth (relative darkening, never brightens), 0..~0.4 (default: %(default)s)",
    )
    parser.add_argument(
        "--shading-type",
        type=str,
        choices=["directional", "vignette", "mixed"],
        default="mixed",
        help="directional: shadow along the light axis; vignette: darker corners; mixed: "
        "random per card (default: %(default)s)",
    )


def add_shadow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shadow-prob",
        type=float,
        default=0.5,
        help="Probability a card casts a drop shadow; 0 disables shadows (default: %(default)s)",
    )
    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=0.3,
        help="Shadow opacity, 0..1 (default: %(default)s)",
    )
    parser.add_argument(
        "--shadow-offset-min",
        type=float,
        default=0.5,
        help="Minimum shadow sliver offset, in absolute canvas pixels -- not scaled by "
        "card size, so it stays hairline-thin regardless of how big a card renders "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--shadow-offset-max",
        type=float,
        default=1.5,
        help="Maximum shadow sliver offset, in absolute canvas pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--shadow-blur",
        type=float,
        default=0.35,
        help="Shadow softness (Gaussian sigma) in absolute canvas pixels, capped to the "
        "offset itself so it can't bleed past the sliver (default: %(default)s)",
    )


def add_hard_shadow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hard-shadow-prob",
        type=float,
        default=0.25,
        help="Probability the whole board gets an irregular hard shadow patch; 0 disables it (default: %(default)s)",
    )
    parser.add_argument(
        "--hard-shadow-strength",
        type=float,
        default=0.45,
        help="Hard shadow opacity, 0..1 (default: %(default)s)",
    )
    parser.add_argument(
        "--hard-shadow-coverage",
        type=float,
        default=0.2,
        help="Approximate fraction of the board area the shadow covers, 0..1 (default: %(default)s)",
    )
    parser.add_argument(
        "--hard-shadow-edge-softness",
        type=float,
        default=0.006,
        help="Shadow edge softness (Gaussian sigma) as a fraction of canvas size -- small, "
        "for a hard edge (default: %(default)s)",
    )


def add_card_cover_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--card-cover-prob",
        type=float,
        default=0.3,
        help="Probability a given card gets a solid covering patch over part of its printed "
        "face; 0 disables it (default: %(default)s)",
    )
    parser.add_argument(
        "--card-cover-coverage",
        type=float,
        default=0.25,
        help="Approximate fraction of the card's own area the patch covers, 0..1 (default: %(default)s)",
    )


def add_card_aug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hue",
        type=float,
        default=15.0,
        help="Max hue shift in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--sat",
        type=float,
        default=0.35,
        help="Max relative saturation factor change (default: %(default)s)",
    )
    parser.add_argument(
        "--wb",
        type=float,
        default=0.15,
        help="Max relative per-channel white-balance gain change (default: %(default)s)",
    )
    parser.add_argument(
        "--bright",
        type=float,
        default=0.25,
        help="Max relative brightness change (default: %(default)s)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.30,
        help="Max relative contrast change (default: %(default)s)",
    )


def add_blur_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--card-blur-prob",
        type=float,
        default=0.7,
        help="Probability a card gets a per-card Gaussian blur; 0 disables card blur (default: %(default)s)",
    )
    parser.add_argument(
        "--card-blur",
        type=float,
        default=0.02,
        help="Maximum card blur (Gaussian sigma) as a fraction of card width (default: %(default)s)",
    )
    parser.add_argument(
        "--bg-blur-prob",
        type=float,
        default=0.8,
        help="Probability the background gets a Gaussian blur; 0 disables background blur (default: %(default)s)",
    )
    parser.add_argument(
        "--bg-blur",
        type=float,
        default=2.5,
        help="Maximum background blur (Gaussian sigma) in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--defocus-prob",
        type=float,
        default=0.10,
        help="Probability a board is strongly out of focus (whole-frame blur); "
        "0 disables defocus boards (default: %(default)s)",
    )
    parser.add_argument(
        "--defocus-blur",
        type=float,
        default=4.5,
        help="Maximum whole-frame defocus blur (Gaussian sigma) in pixels (default: %(default)s)",
    )


def add_rot_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rot-range",
        type=float,
        default=45,
        help="Card rotation augmentation as +/- degrees around the card orientation (default: %(default)s)",
    )
    parser.add_argument(
        "--rot-outlier-prob",
        type=float,
        default=0.15,
        help="Probability a given card becomes a rotation outlier: a card rotated much more than "
        "the rest of the board, mimicking a carelessly-placed card (default: %(default)s)",
    )
    parser.add_argument(
        "--rot-outlier-min",
        type=float,
        default=60,
        help="Minimum rotation magnitude in degrees for outlier cards (default: %(default)s)",
    )
    parser.add_argument(
        "--rot-outlier-max",
        type=float,
        default=180,
        help="Maximum rotation magnitude in degrees for outlier cards (default: %(default)s)",
    )
    parser.add_argument(
        "--rel-min-separation",
        type=float,
        default=0.85,
        help="Minimum card separation as a fraction of full non-overlap distance along the "
        "line between two card centers (1.0 = no overlap at all); pairs closer than this are "
        "pushed apart to cap worst-case overlap from jitter/rotation. 0 disables "
        "(default: %(default)s)",
    )


def add_persp_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--persp",
        type=float,
        default=1.25,
        help="Perspective warp strength: the projective cap coefficient (default: %(default)s)",
    )


def spec_from_args(args: argparse.Namespace) -> SpecConfig:
    return SpecConfig(
        prob=args.spec_prob,
        intensity=args.spec_intensity,
        size=args.spec_size,
        streak=args.spec_streak,
        direction=args.spec_direction,
        align=args.spec_align,
    )


def shading_from_args(args: argparse.Namespace) -> ShadeConfig:
    return ShadeConfig(
        prob=args.shading_prob,
        amp=args.shading_amp,
        type=args.shading_type,
    )


def shadow_from_args(args: argparse.Namespace) -> ShadowConfig:
    return ShadowConfig(
        prob=args.shadow_prob,
        strength=args.shadow_strength,
        offset_px=(args.shadow_offset_min, args.shadow_offset_max),
        blur_px=args.shadow_blur,
    )


def hard_shadow_from_args(args: argparse.Namespace) -> HardShadowConfig:
    return HardShadowConfig(
        prob=args.hard_shadow_prob,
        strength=args.hard_shadow_strength,
        coverage=args.hard_shadow_coverage,
        edge_softness=args.hard_shadow_edge_softness,
    )


def card_cover_from_args(args: argparse.Namespace) -> CardCoverConfig:
    return CardCoverConfig(prob=args.card_cover_prob, coverage=args.card_cover_coverage)


def card_aug_from_args(args: argparse.Namespace) -> CardAugConfig:
    return CardAugConfig(
        hue_deg=args.hue,
        sat=args.sat,
        wb=args.wb,
        bright=args.bright,
        contrast=args.contrast,
    )


def blur_from_args(args: argparse.Namespace) -> BlurConfig:
    return BlurConfig(
        card_prob=args.card_blur_prob,
        card_sigma=args.card_blur,
        bg_prob=args.bg_blur_prob,
        bg_sigma=args.bg_blur,
        defocus_prob=args.defocus_prob,
        defocus_sigma=args.defocus_blur,
    )
