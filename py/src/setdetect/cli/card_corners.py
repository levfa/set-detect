import argparse
import pathlib as pl

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setdetect.data.texturecan import DEFAULT_MANIFEST_NAME, IMAGE_SUFFIXES
from setdetect.model import versioning
from setdetect.model.card_corners import SynthConfig, generate_dataset, predict, train
from setdetect.ui.gui import CORNER_COLORS
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


def _resolve_model_path(model: str | pl.Path, models_root: pl.Path) -> pl.Path:
    """Resolve a model argument to an absolute path; bare pretrained names are
    downloaded into `models_root` by ultralytics instead of the CWD."""
    p = pl.Path(model).expanduser()
    if p.exists() or p.parent != pl.Path("."):
        return p
    models_root.mkdir(parents=True, exist_ok=True)
    return models_root / p


def _promoted_weights(runs_root: pl.Path) -> pl.Path:
    """Weights (best.pt, falling back to last.pt) from the promoted 'current' run."""
    weights_dir = versioning.resolve_current(runs_root) / "weights"
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    last = weights_dir / "last.pt"
    if last.exists():
        return last
    raise SystemExit(f"no weights found under {weights_dir}")


def _newest_results_csv(runs_root: pl.Path) -> pl.Path | None:
    results = sorted(
        runs_root.glob("*/results.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return results[-1] if results else None


def _monitor_train(args: argparse.Namespace) -> None:
    results = args.results if args.results is not None else _newest_results_csv(pl.Path(args.runs_root))
    if results is None or not results.is_file():
        raise SystemExit(
            f"no results.csv found under {args.runs_root}; pass a results.csv path or start a training run first"
        )
    df = pd.read_csv(results)
    df.columns = df.columns.str.strip()
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["metrics/mAP50(P)"], label="Corner mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(P)"], label="Corner mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Card Corner Performance ({results.parent.name})")
    plt.legend()
    plt.grid()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out)
        print(f"saved {args.out}")
    else:
        plt.show()


def _add_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=pl.Path,
        required=True,
        help="Directory of the corner dataset (val split + data.yaml) written by "
        "'synth-gen' and read by 'synth-train'.",
    )


def _add_runs_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root",
        type=pl.Path,
        required=True,
        help="Directory holding this model's versioned training runs (vNN-YYYYMMDD "
        "subdirectories, plus a 'current' symlink once one is promoted)",
    )


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-val",
        type=int,
        default=200,
        help="Number of validation images to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for generation (default: %(default)s)",
    )
    parser.add_argument(
        "--num-cards",
        type=int,
        default=12,
        help="Number of cards placed per board (default: %(default)s)",
    )


def _synth_config_from_args(args: argparse.Namespace) -> SynthConfig:
    manifest = args.textures_manifest or (pl.Path(args.textures_root) / DEFAULT_MANIFEST_NAME)
    return SynthConfig(
        cards_root=args.cards_root,
        textures_root=args.textures_root,
        manifest=manifest,
        samples_per_epoch=getattr(args, "samples_per_epoch", SynthConfig.samples_per_epoch),
        num_cards=args.num_cards,
        spec=spec_from_args(args),
        shade=shading_from_args(args),
        shadow=shadow_from_args(args),
        blur=blur_from_args(args),
        hard_shadow=hard_shadow_from_args(args),
        card_cover=card_cover_from_args(args),
        rot_range=(-args.rot_range, args.rot_range) if args.rot_range is not None else None,
        rot_outlier_prob=args.rot_outlier_prob,
        rot_outlier_range=(args.rot_outlier_min, args.rot_outlier_max),
        rel_min_separation=args.rel_min_separation,
        persp=args.persp,
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="YOLO corner-keypoint model to fine-tune; bare pretrained names are downloaded "
        "into --models-root (default: %(default)s)",
    )
    parser.add_argument(
        "--models-root",
        type=pl.Path,
        required=True,
        help="Directory for downloaded pretrained base weights (used when --model is a bare "
        "pretrained name rather than a path)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for training, e.g. 0 or cpu (default: auto)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run version name (default: auto 'vNN-YYYYMMDD', next after existing runs under --runs-root)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training (default: %(default)s)",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=2000,
        help="Synthetic boards drawn per training epoch (default: %(default)s)",
    )
    parser.add_argument(
        "--num-cards",
        type=int,
        default=12,
        help="Number of cards placed per board (default: %(default)s)",
    )


def _draw_cards(img: np.ndarray, keypoints: np.ndarray, visibility: np.ndarray) -> None:
    for kp, vis in zip(keypoints, visibility):
        quad = np.asarray(kp, np.float32)
        cv2.polylines(img, [quad.astype(np.int32)], True, (0, 0, 255), 3)
        for i, (x, y) in enumerate(quad):
            if vis[i] < 0.5:
                # occluded corner: small gray marker instead of a colored dot
                cv2.circle(img, (int(x), int(y)), 3, (200, 200, 200), -1)
            else:
                cv2.circle(img, (int(x), int(y)), 5, CORNER_COLORS[i], -1)


def _show_boards(args: argparse.Namespace) -> None:
    raw_root = pl.Path(args.raw_root)
    image_paths = [
        path for path in sorted(raw_root.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not image_paths:
        raise SystemExit(f"no images found in {raw_root}")
    weights = (
        pl.Path(args.weights).expanduser() if args.weights is not None else _promoted_weights(pl.Path(args.runs_root))
    )

    print(f"inferring {len(image_paths)} images with {weights} ...")
    predictions = predict(
        weights=weights,
        image_paths=image_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )

    win_name = "Inference"
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for path, (keypoints, visibility) in zip(image_paths, predictions):
        img = cv2.imread(str(path))
        if img is None:
            print(f"warning: could not read {path}, skipping")
            continue
        _draw_cards(img, keypoints, visibility)
        cv2.putText(
            img,
            f"{path.name}: {len(keypoints)} cards",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        if args.out_dir is not None:
            cv2.imwrite(str(args.out_dir / path.name), img)
            continue
        cv2.imshow(win_name, img)
        if cv2.waitKey(0) == ord("q"):
            break
    if args.out_dir is not None:
        print(f"saved annotated images to {args.out_dir}")
    else:
        cv2.destroyWindow(win_name)


def _synth_gen(args: argparse.Namespace) -> None:
    print("generating the synthetic validation split ...")
    val_stats = generate_dataset(
        cfg=_synth_config_from_args(args),
        out_root=args.data_root,
        num_val=args.num_val,
        seed=args.seed,
    )
    print(f"val:   {val_stats.images} images, {val_stats.labels} labeled cards")
    print(f"dataset written to {args.data_root} ({args.data_root / 'data.yaml'})")


def _write_run_metadata(run_dir: pl.Path, args: argparse.Namespace, model: pl.Path, name: str) -> None:
    metrics: dict = {}
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        metrics = {col: float(last[col]) for col in df.columns if col != "epoch"}
    versioning.write_run_metadata(
        run_dir,
        version=name,
        family="set-card-corners",
        base_model=str(model),
        hyperparams={
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "num_cards": args.num_cards,
            "samples_per_epoch": args.samples_per_epoch,
        },
        metrics=metrics,
    )


def _synth_train(args: argparse.Namespace) -> None:
    data_yaml = pl.Path(args.data_root) / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(
            f"missing {data_yaml}; run 'set-card-corners synth-gen' first to create the val split + data.yaml"
        )
    model = _resolve_model_path(args.model, pl.Path(args.models_root))
    name = args.name or versioning.next_version_name(pl.Path(args.runs_root))
    best = train(
        cfg=_synth_config_from_args(args),
        data_yaml=data_yaml,
        model=model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.runs_root,
        name=name,
        seed=args.seed,
    )
    run_dir = best.parent.parent
    _write_run_metadata(run_dir, args, model, name)
    print(f"training finished, best weights at {best}")
    print(f"run 'set-card-corners promote {name}' to make this the default")


def _export(args: argparse.Namespace) -> None:
    from setdetect.model.card_corners import export_onnx

    weights = (
        pl.Path(args.weights).expanduser() if args.weights is not None else _promoted_weights(pl.Path(args.runs_root))
    )
    out_path = weights.parent.parent / "onnx" / "card_corners.onnx"
    out = export_onnx(weights, out_path=out_path, imgsz=args.imgsz)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"exported to {out}  ({size_mb:.1f} MB)")


def _corner_model_card_body() -> str:
    return "\n".join(
        [
            "# set-card-corners",
            "",
            "YOLO11-pose ONNX model that detects the four corners of SET® playing cards in an image.",
            "",
            "## Input / output",
            "",
            "- Input: RGB image, resized to 640x640.",
            "- Output: per detected card, 4 keypoints (x, y, visibility) ordered "
            "clockwise from the card's top-left corner: top-left, top-right, "
            "bottom-right, bottom-left.",
            "",
            "## License",
            "",
            "AGPL-3.0.",
            "",
            "## Disclaimer",
            "",
            "Unofficial, fan-made project; not affiliated with, endorsed by, or sponsored by SET Enterprises, Inc.",
            "",
        ]
    )


def _prepare_hf(args: argparse.Namespace) -> None:
    from setdetect.model import hf_export

    runs_root = pl.Path(args.runs_root)
    version = args.version or versioning.current_version(runs_root)
    if version is None:
        raise SystemExit(f"no promoted version under {runs_root}; pass --version explicitly or promote a run first")
    run_dir = runs_root / version

    onnx_path = run_dir / "onnx" / "card_corners.onnx"
    if not onnx_path.is_file():
        raise SystemExit(f"no ONNX export at {onnx_path}; run 'set-card-corners export' for this version first")

    readme = hf_export.render_model_card(
        frontmatter={
            "license": "agpl-3.0",
            "library_name": "onnx",
            "tags": ["object-detection", "keypoint-detection", "yolo", "yolo11-pose", "card-detection"],
            "pipeline_tag": "keypoint-detection",
        },
        body=_corner_model_card_body(),
    )

    hf_export.stage_files(args.out_dir, [(onnx_path, "onnx/card_corners.onnx")])
    hf_export.write_text(args.out_dir, "README.md", readme)
    print(f"staged {version} into {args.out_dir} -- inspect with 'git status', then commit/push/tag it yourself")


def _prepare_github(args: argparse.Namespace) -> None:
    from setdetect.model import hf_export

    runs_root = pl.Path(args.runs_root)
    version = args.version or versioning.current_version(runs_root)
    if version is None:
        raise SystemExit(f"no promoted version under {runs_root}; pass --version explicitly or promote a run first")
    run_dir = runs_root / version

    onnx_path = run_dir / "onnx" / "card_corners.onnx"
    if not onnx_path.is_file():
        raise SystemExit(f"no ONNX export at {onnx_path}; run 'set-card-corners export' for this version first")

    readme = hf_export.render_model_card(
        frontmatter={
            "license": "agpl-3.0",
            "library_name": "onnx",
            "tags": ["object-detection", "keypoint-detection", "yolo", "yolo11-pose", "card-detection"],
            "pipeline_tag": "keypoint-detection",
        },
        body=_corner_model_card_body(),
    )

    hf_export.stage_files(args.out_dir, [(onnx_path, "onnx/card_corners.onnx")])
    hf_export.write_text(args.out_dir, "README.md", readme)
    print(f"staged {version} into {args.out_dir} -- inspect with 'git status', then commit/push it yourself")


def _promote(args: argparse.Namespace) -> None:
    versioning.promote(pl.Path(args.runs_root), args.version)
    print(f"promoted {args.version} -> {args.runs_root}/current")


def _list_versions(args: argparse.Namespace) -> None:
    runs_root = pl.Path(args.runs_root)
    current = versioning.current_version(runs_root)
    for version in versioning.list_versions(runs_root):
        marker = "  (current)" if version == current else ""
        print(f"{version}{marker}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-corners")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "synth-gen",
        help="Generate the synthetic validation split + data.yaml for training",
    )
    add_dataset_args(gen_parser)
    _add_data_root_arg(gen_parser)
    _add_generation_args(gen_parser)
    add_spec_args(gen_parser)
    add_shading_args(gen_parser)
    add_shadow_args(gen_parser)
    add_hard_shadow_args(gen_parser)
    add_card_cover_args(gen_parser)
    add_blur_args(gen_parser)
    add_rot_arg(gen_parser)
    add_persp_arg(gen_parser)

    train_parser = subparsers.add_parser(
        "synth-train",
        help="Train the corner-keypoint model on boards generated on the fly per epoch",
    )
    _add_data_root_arg(train_parser)
    _add_runs_root_arg(train_parser)
    add_dataset_args(train_parser)
    _add_training_args(train_parser)
    add_spec_args(train_parser)
    add_shading_args(train_parser)
    add_shadow_args(train_parser)
    add_hard_shadow_args(train_parser)
    add_card_cover_args(train_parser)
    add_blur_args(train_parser)
    add_rot_arg(train_parser)
    add_persp_arg(train_parser)

    boards_parser = subparsers.add_parser(
        "show-boards",
        help="Run corner-keypoint inference on real board photos and show the detections",
    )
    boards_parser.add_argument(
        "--raw-root",
        type=pl.Path,
        required=True,
        help="Directory containing the raw real photos to test against.",
    )
    boards_parser.add_argument(
        "--weights",
        type=pl.Path,
        default=None,
        help="Model weights; defaults to the promoted 'current' run's best.pt under --runs-root",
    )
    _add_runs_root_arg(boards_parser)
    boards_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: %(default)s)",
    )
    boards_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: %(default)s)",
    )
    boards_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference, e.g. 0 or cpu (default: auto)",
    )
    boards_parser.add_argument(
        "--out-dir",
        type=pl.Path,
        default=None,
        help="Save annotated images to this directory instead of showing them",
    )

    monitor_parser = subparsers.add_parser(
        "monitor-training",
        help="Plot corner training metrics from a run's results.csv",
    )
    monitor_parser.add_argument(
        "results",
        type=pl.Path,
        nargs="?",
        default=None,
        help="Path to a results.csv (default: newest under --runs-root)",
    )
    _add_runs_root_arg(monitor_parser)
    monitor_parser.add_argument(
        "--out",
        type=pl.Path,
        default=None,
        help="Save the figure to this path instead of showing it",
    )

    ex_parser = subparsers.add_parser(
        "export",
        help="Export the trained corner-keypoint model to ONNX",
    )
    ex_parser.add_argument(
        "--weights",
        type=pl.Path,
        default=None,
        help="Model weights (default: promoted 'current' run's best.pt under --runs-root)",
    )
    _add_runs_root_arg(ex_parser)
    ex_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: %(default)s)",
    )

    promote_parser = subparsers.add_parser(
        "promote",
        help="Mark a trained run version as the 'current' default for inference/export",
    )
    promote_parser.add_argument("version", type=str, help="Version name under --runs-root to promote")
    _add_runs_root_arg(promote_parser)

    list_parser = subparsers.add_parser(
        "list-versions",
        help="List trained run versions and which one is current",
    )
    _add_runs_root_arg(list_parser)

    pp_parser = subparsers.add_parser(
        "prepare-hf",
        help="Stage this model's ONNX export + a generated model card into a local "
        "directory for manual Hugging Face publishing",
    )
    _add_runs_root_arg(pp_parser)
    pp_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version under --runs-root to stage (default: the promoted 'current' version)",
    )
    pp_parser.add_argument(
        "--out-dir",
        type=pl.Path,
        required=True,
        help="Directory to stage files into (e.g. your local clone of the Hugging Face repo)",
    )

    pg_parser = subparsers.add_parser(
        "prepare-github",
        help="Stage this model's ONNX export + a generated model card into a local "
        "directory for manual GitHub publishing",
    )
    _add_runs_root_arg(pg_parser)
    pg_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version under --runs-root to stage (default: the promoted 'current' version)",
    )
    pg_parser.add_argument(
        "--out-dir",
        type=pl.Path,
        required=True,
        help="Directory to stage files into (e.g. a subfolder of your local clone of the GitHub repo)",
    )

    args = parser.parse_args(argv)
    if args.command == "synth-gen":
        _synth_gen(args)
    elif args.command == "synth-train":
        _synth_train(args)
    elif args.command == "show-boards":
        _show_boards(args)
    elif args.command == "monitor-training":
        _monitor_train(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "promote":
        _promote(args)
    elif args.command == "list-versions":
        _list_versions(args)
    elif args.command == "prepare-hf":
        _prepare_hf(args)
    elif args.command == "prepare-github":
        _prepare_github(args)


if __name__ == "__main__":
    main()
