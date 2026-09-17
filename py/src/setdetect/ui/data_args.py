import argparse
import pathlib as pl


def add_img_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--data-root",
        type=pl.Path,
        required=True,
        help="Root directory of a real board-photo dataset (raw/ + optional labels/).",
    )


def add_video_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--data-root",
        type=pl.Path,
        required=True,
        help="Root directory of a video dataset.",
    )
