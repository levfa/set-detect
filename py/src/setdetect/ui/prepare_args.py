import argparse
import pathlib as pl


def add_prepare_args(parser: argparse.ArgumentParser, default_out_dir: pl.Path | None = None) -> None:
    """Add ``--version`` and ``--out-dir``; the latter is required unless ``default_out_dir`` is given."""
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version under --runs-root to use (default: the promoted 'current' version)",
    )
    parser.add_argument(
        "--out-dir",
        type=pl.Path,
        default=default_out_dir,
        required=default_out_dir is None,
        help="Directory to copy the files into" + ("" if default_out_dir is None else " (default: %(default)s)"),
    )
