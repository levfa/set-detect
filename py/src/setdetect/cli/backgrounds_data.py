import argparse
import pathlib as pl
from collections import Counter

import cv2
import numpy as np

from setdetect.data.texturecan import (
    DEFAULT_MANIFEST_NAME,
    BackgroundEntry,
    build_backgrounds,
    download,
    load_backgrounds,
    render_contact_sheet,
)


def _add_local_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--local-dir",
        type=pl.Path,
        required=True,
        help="Directory containing the dataset (e.g. your local huggingface/nyuuzyou/texturecan mirror)",
    )


def _add_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=pl.Path,
        default=None,
        help=f"Manifest path (default: <local-dir>/{DEFAULT_MANIFEST_NAME})",
    )


def _build_manifest(args: argparse.Namespace) -> None:
    root = args.local_dir
    manifest = args.manifest or root / DEFAULT_MANIFEST_NAME
    entries = build_backgrounds(
        root,
        out=manifest,
        max_mean_lum=args.max_mean_lum,
        min_dim=args.min_dim,
        cap_per_category=args.cap_per_category,
    )
    counts = Counter(entry.category for entry in entries)
    print(f"wrote {manifest} ({len(entries)} images)")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")


def _qc(args: argparse.Namespace) -> None:
    root = args.local_dir
    manifest = args.manifest or root / DEFAULT_MANIFEST_NAME
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest} (run `texturecan build-manifest` first)")
    entries = load_backgrounds(manifest)
    if args.list:
        for index, entry in enumerate(entries):
            print(f"{index}\t{entry.category}\t{entry.mean_lum:.0f}\t{entry.rel_path}")
    by_category: dict[str, list[tuple[int, BackgroundEntry]]] = {}
    for index, entry in enumerate(entries):
        by_category.setdefault(entry.category, []).append((index, entry))
    out_dir = args.out_dir or root / ".qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for category in sorted(by_category):
        sheet = render_contact_sheet(
            by_category[category],
            root,
            title=f"{category} ({len(by_category[category])})",
            cols=args.cols,
            thumb=args.thumb,
        )
        path = out_dir / f"{category}.png"
        sheet.save(path)
        print(f"saved {path}")
        if not args.no_show:
            cv2.imshow(category, cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            cv2.destroyWindow(category)
            if key == ord("q"):
                break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="texturecan")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download the nyuuzyou/texturecan dataset")
    _add_local_dir_arg(download_parser)
    download_parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision to download (default: %(default)s)",
    )

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Build the background manifest (brightness filter + per-category quota)",
    )
    _add_local_dir_arg(manifest_parser)
    _add_manifest_arg(manifest_parser)
    manifest_parser.add_argument(
        "--max-mean-lum",
        type=float,
        default=215.0,
        help="Drop textures with mean luminance above this (default: %(default)s)",
    )
    manifest_parser.add_argument(
        "--min-dim",
        type=int,
        default=512,
        help="Drop textures smaller than this in either dimension (default: %(default)s)",
    )
    manifest_parser.add_argument(
        "--cap-per-category",
        type=int,
        default=100,
        help="Keep at most this many textures per material category (default: %(default)s)",
    )

    qc_parser = subparsers.add_parser(
        "qc",
        help="Render labeled contact sheets of the background manifest",
    )
    _add_local_dir_arg(qc_parser)
    _add_manifest_arg(qc_parser)
    qc_parser.add_argument(
        "--cols",
        type=int,
        default=10,
        help="Thumbnails per row (default: %(default)s)",
    )
    qc_parser.add_argument(
        "--thumb",
        type=int,
        default=128,
        help="Thumbnail size in pixels (default: %(default)s)",
    )
    qc_parser.add_argument(
        "--out-dir",
        type=pl.Path,
        default=None,
        help="Directory to save sheets into (default: <local-dir>/.qc)",
    )
    qc_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save the sheets, do not display them",
    )
    qc_parser.add_argument(
        "--list",
        action="store_true",
        help="Also print the indexed manifest (index, category, mean_lum, path)",
    )

    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.local_dir, args.revision)
    elif args.command == "build-manifest":
        _build_manifest(args)
    elif args.command == "qc":
        _qc(args)


if __name__ == "__main__":
    main()
