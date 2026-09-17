import argparse
import pathlib as pl
import typing as tp


def resolve_corner_weights(corner_weights: pl.Path) -> pl.Path:
    if not corner_weights.exists():
        raise SystemExit(
            f"no corner weights found at {corner_weights}; pass --det-corner-weights explicitly, "
            "or run 'set-card-corners promote <version>' to select a trained run"
        )
    return corner_weights


def resolve_class_weights(class_weights: pl.Path) -> pl.Path:
    if not class_weights.exists():
        raise SystemExit(
            f"no classifier weights found at {class_weights}; pass --det-class-weights explicitly, "
            "or run 'set-card-classification promote <version>' to select a trained run"
        )
    return class_weights


def add_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--det-corner-weights",
        type=pl.Path,
        required=True,
        help="Corner-keypoint ONNX model (YOLOv8-pose)",
    )
    parser.add_argument(
        "--det-class-weights",
        type=pl.Path,
        required=True,
        help="Classifier ONNX model",
    )
    parser.add_argument(
        "--det-device",
        type=int,
        default=0,
        help="Corner detection and card classification GPU device (default: 0, -1 for CPU)",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="Corner detection confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--det-imgsz",
        type=int,
        default=640,
        help="Corner model input size (default: %(default)s)",
    )


def detection_kwargs_from_args(args: argparse.Namespace) -> dict[str, tp.Any]:
    return {
        "corner_weights": resolve_corner_weights(args.det_corner_weights),
        "class_weights": resolve_class_weights(args.det_class_weights),
        "device_id": args.det_device,
        "conf": args.det_conf,
        "imgsz": args.det_imgsz,
    }
