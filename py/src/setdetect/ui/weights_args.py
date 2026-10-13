import argparse
import pathlib as pl

from setdetect.model import versioning


def add_weights_args(parser: argparse.ArgumentParser, runs_root_flag: str = "--runs-root") -> None:
    """Add ``--weights`` and ``--checkpoint``, which together select a trained checkpoint file."""
    parser.add_argument(
        "--weights",
        type=pl.Path,
        default=None,
        help=f"Checkpoint file (default: the --checkpoint of the promoted 'current' run under {runs_root_flag})",
    )
    parser.add_argument(
        "--checkpoint",
        choices=versioning.CHECKPOINTS,
        default=versioning.CHECKPOINTS[0],
        help="Which checkpoint of the promoted run to use when --weights is omitted: the final epoch, "
        "or the one that scored best on the validation set (default: %(default)s)",
    )


def weights_from_args(args: argparse.Namespace, runs_root: pl.Path, weights_subdir: str = "") -> pl.Path:
    """The explicit ``--weights`` file, else the promoted run's ``--checkpoint``.

    ``weights_subdir`` is where a run keeps its checkpoints, relative to the run directory.
    """
    if args.weights is not None:
        return pl.Path(args.weights).expanduser()
    return versioning.promoted_checkpoint(pl.Path(runs_root), args.checkpoint, weights_subdir)
