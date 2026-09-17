import argparse
import dataclasses
import pathlib as pl

import cv2
import numpy as np

from setdetect.model import versioning
from setdetect.synth_data.card_synth import make_card_augmenter
from setdetect.synth_data.effects import Effects, make_effects
from setdetect.ui.data_args import add_img_root_arg
from setdetect.ui.synth_args import (
    add_blur_args,
    add_card_aug_args,
    add_card_cover_args,
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


def _add_runs_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root",
        type=pl.Path,
        required=True,
        help="Directory holding this model's versioned training runs (vNN-YYYYMMDD "
        "subdirectories, plus a 'current' symlink once one is promoted)",
    )


def _add_synth_effect_args(parser: argparse.ArgumentParser) -> None:
    add_spec_args(parser)
    add_shading_args(parser)
    add_shadow_args(parser)
    add_hard_shadow_args(parser)
    add_card_cover_args(parser)
    add_card_aug_args(parser)
    add_blur_args(parser)
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="Disable all per-card and whole-board augmentation (photometric, shading, "
        "shadows, specular, blur, card-cover)",
    )


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


def _train(args: argparse.Namespace) -> None:
    from setdetect.model.card_classification import train as train_fn

    runs_root: pl.Path = args.runs_root
    version = args.version or versioning.next_version_name(runs_root)
    out_dir = runs_root / version

    train_fn(
        data_root=args.data_root,
        cards_root=args.cards_root,
        tex_root=args.tex_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=out_dir,
        boards_per_epoch=args.boards_per_epoch,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        effects=_effects_from_args(args),
    )
    versioning.write_run_metadata(
        out_dir,
        version=version,
        family="set-card-class",
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "boards_per_epoch": args.boards_per_epoch,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
        },
    )
    print(f"\nrun 'set-card-classification promote {version}' to make this the default")


def _show_batch(args: argparse.Namespace) -> None:
    import math

    from setdetect.model.card_classification import (
        CARD_SIZE,
        COLOR_INV,
        COUNT_INV,
        FILL_INV,
        MEAN,
        OTHER,
        SHAPE_INV,
        STD,
        SyntheticCardDataset,
    )

    ds = SyntheticCardDataset(
        args.cards_root,
        args.tex_root,
        boards_per_epoch=args.boards_per_epoch,
        effects=_effects_from_args(args),
        seed=args.seed,
    )
    ds.set_epoch(0)

    all_images = []
    global_idx = 0

    for board_i in range(args.boards_per_epoch):
        imgs, count_t, color_t, fill_t, shape_t = ds[board_i]
        n_cards = imgs.size(0)
        print(f"board {board_i}  ({n_cards} cards)")
        for card_i in range(n_cards):
            c, co, f, sh = (
                int(count_t[card_i].item()),
                int(color_t[card_i].item()),
                int(fill_t[card_i].item()),
                int(shape_t[card_i].item()),
            )
            is_set = not (c == OTHER and co == OTHER and f == OTHER and sh == OTHER)
            tag = "set  " if is_set else "other"
            if is_set:
                label_str = f"{COUNT_INV[c]:6s} {COLOR_INV[co]:7s} {FILL_INV[f]:8s} {SHAPE_INV[sh]:9s}"
            else:
                label_str = "---    ---      ---       ---"
            print(f"  {global_idx:3d}  [{tag}]  {label_str}")
            all_images.append(imgs[card_i])
            global_idx += 1

    n = len(all_images)
    cols = min(8, n)
    rows = math.ceil(n / cols)
    cell_w, cell_h = CARD_SIZE
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        img_rgb = all_images[i].permute(1, 2, 0).numpy()
        img_rgb = img_rgb * np.array(STD) + np.array(MEAN)
        img_rgb = np.clip(img_rgb, 0, 1)
        img_bgr = (img_rgb[:, :, ::-1] * 255).astype(np.uint8)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = img_bgr

    win_name = f"Batch preview  ({n} cards, {args.boards_per_epoch} boards)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, grid)
    print(f"\n{n} cards from {args.boards_per_epoch} boards — press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _quantize(args: argparse.Namespace) -> None:
    from setdetect.model.card_classification import quantize_onnx

    weights: pl.Path = args.weights or (versioning.resolve_current(args.runs_root) / "onnx" / "model.onnx")
    if not weights.exists():
        raise SystemExit(f"no ONNX model at {weights}")
    out: pl.Path = args.out or weights.with_name("model_quantized.onnx")
    quantize_onnx(weights, out, data_root=args.data_root, num_samples=args.num_samples)
    orig_mb = weights.stat().st_size / (1024 * 1024)
    data_file = weights.parent / (weights.name + ".data")
    if data_file.exists():
        orig_mb += data_file.stat().st_size / (1024 * 1024)
    quant_mb = out.stat().st_size / (1024 * 1024)
    ratio = (1 - quant_mb / orig_mb) * 100 if orig_mb > 0 else 0
    print(f"quantized to {out}  ({quant_mb:.1f} MB, {ratio:.0f}% smaller than {orig_mb:.1f} MB)")


def _export(args: argparse.Namespace) -> None:
    from setdetect.model.card_classification import export_onnx

    weights: pl.Path = args.weights or (versioning.resolve_current(args.runs_root) / "best.pt")
    if not weights.exists():
        raise SystemExit(f"no checkpoint at {weights}")
    out: pl.Path = args.out or (weights.parent / "onnx" / "model.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(weights, out, opset=args.opset, dynamic_batch=args.dynamic_batch)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"exported to {out}  ({size_mb:.1f} MB, opset {args.opset})")


def _classifier_model_card_body() -> str:
    return "\n".join(
        [
            "# set-card-class",
            "",
            "ResNet18 ONNX model that classifies a cropped image of a SET® card "
            "into its four attributes: count, color, fill, shape.",
            "",
            "## Input / output",
            "",
            "- Input: RGB card crop, resized to 224x320, normalized with ImageNet "
            "mean/std (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).",
            "- Output: four classification heads, in order count/color/fill/shape, "
            "each one of its three SET values plus `other` (low-confidence / "
            "non-card crop).",
            "- `onnx/model.onnx` (fp32) and, if present, `onnx/model_quantized.onnx` (int8).",
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

    runs_root: pl.Path = args.runs_root
    version = args.version or versioning.current_version(runs_root)
    if version is None:
        raise SystemExit(f"no promoted version under {runs_root}; pass --version explicitly or promote a run first")
    run_dir = runs_root / version

    onnx_dir = run_dir / "onnx"
    model_onnx = onnx_dir / "model.onnx"
    if not model_onnx.is_file():
        raise SystemExit(f"no ONNX export at {model_onnx}; run 'set-card-classification export' for this version first")

    artifact_files = [(model_onnx, "onnx/model.onnx")]
    for name in ("model.onnx.data", "model_quantized.onnx"):
        path = onnx_dir / name
        if path.is_file():
            artifact_files.append((path, f"onnx/{name}"))
        else:
            print(f"no {name} found, skipping")

    readme = hf_export.render_model_card(
        frontmatter={
            "license": "agpl-3.0",
            "library_name": "onnx",
            "tags": ["image-classification", "resnet18", "multi-task", "card-classification"],
            "pipeline_tag": "image-classification",
        },
        body=_classifier_model_card_body(),
    )

    hf_export.stage_files(args.out_dir, artifact_files)
    hf_export.write_text(args.out_dir, "README.md", readme)
    print(f"staged {version} into {args.out_dir} -- inspect with 'git status', then commit/push/tag it yourself")


def _prepare_github(args: argparse.Namespace) -> None:
    from setdetect.model import hf_export

    runs_root: pl.Path = args.runs_root
    version = args.version or versioning.current_version(runs_root)
    if version is None:
        raise SystemExit(f"no promoted version under {runs_root}; pass --version explicitly or promote a run first")
    run_dir = runs_root / version

    onnx_dir = run_dir / "onnx"
    model_onnx = onnx_dir / "model.onnx"
    if not model_onnx.is_file():
        raise SystemExit(f"no ONNX export at {model_onnx}; run 'set-card-classification export' for this version first")

    artifact_files = [(model_onnx, "onnx/model.onnx")]
    for name in ("model.onnx.data", "model_quantized.onnx"):
        path = onnx_dir / name
        if path.is_file():
            artifact_files.append((path, f"onnx/{name}"))
        else:
            print(f"no {name} found, skipping")

    readme = hf_export.render_model_card(
        frontmatter={
            "license": "agpl-3.0",
            "library_name": "onnx",
            "tags": ["image-classification", "resnet18", "multi-task", "card-classification"],
            "pipeline_tag": "image-classification",
        },
        body=_classifier_model_card_body(),
    )

    hf_export.stage_files(args.out_dir, artifact_files)
    hf_export.write_text(args.out_dir, "README.md", readme)
    print(f"staged {version} into {args.out_dir} -- inspect with 'git status', then commit/push it yourself")


def _promote(args: argparse.Namespace) -> None:
    versioning.promote(args.runs_root, args.version)
    print(f"promoted {args.version} -> {args.runs_root}/current")


def _list_versions(args: argparse.Namespace) -> None:
    runs_root: pl.Path = args.runs_root
    current = versioning.current_version(runs_root)
    for version in versioning.list_versions(runs_root):
        marker = "  (current)" if version == current else ""
        print(f"{version}{marker}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tr = subparsers.add_parser(
        "train",
        help="Train the card classifier on synthetic data, validate on labeled photos",
    )
    add_img_root_arg(tr)
    _add_runs_root_arg(tr)
    tr.add_argument(
        "--cards-root",
        type=pl.Path,
        required=True,
        help="Root of the masked card-cutout dataset (raw/ + masked/).",
    )
    tr.add_argument(
        "--tex-root",
        type=pl.Path,
        required=True,
        help="Root of the texture dataset (e.g. your local huggingface/nyuuzyou/texturecan mirror)",
    )
    tr.add_argument("--epochs", type=int, default=50, help="Training epochs (default: %(default)s)")
    tr.add_argument("--batch-size", type=int, default=4, help="Batch size (default: %(default)s)")
    tr.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: %(default)s)")
    tr.add_argument(
        "--version",
        type=str,
        default=None,
        help="Run version name (default: auto 'vNN-YYYYMMDD', next after existing runs)",
    )
    tr.add_argument(
        "--boards-per-epoch",
        type=int,
        default=256,
        help="Synthetic boards per epoch (default: %(default)s)",
    )
    tr.add_argument("--seed", type=int, default=42, help="RNG seed (default: %(default)s)")
    tr.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    tr.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: %(default)s)")
    tr.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train backbone from scratch instead of ImageNet-pretrained",
    )
    _add_synth_effect_args(tr)

    sb = subparsers.add_parser(
        "show-batch",
        help="Display one synthetic training batch as a grid with labels in the terminal",
    )
    sb.add_argument(
        "--cards-root",
        type=pl.Path,
        required=True,
        help="Root of the masked card-cutout dataset (raw/ + masked/).",
    )
    sb.add_argument(
        "--tex-root",
        type=pl.Path,
        required=True,
        help="Root of the texture dataset (e.g. your local huggingface/nyuuzyou/texturecan mirror)",
    )
    sb.add_argument(
        "--boards-per-epoch",
        type=int,
        default=4,
        help="Synthetic boards in the preview batch (default: %(default)s)",
    )
    sb.add_argument("--seed", type=int, default=42, help="RNG seed (default: %(default)s)")
    _add_synth_effect_args(sb)

    ex = subparsers.add_parser(
        "export",
        help="Export the trained classifier to ONNX",
    )
    _add_runs_root_arg(ex)
    ex.add_argument(
        "--weights",
        type=pl.Path,
        default=None,
        help="Checkpoint to export (default: promoted 'current' run's best.pt under --runs-root)",
    )
    ex.add_argument(
        "--out",
        type=pl.Path,
        default=None,
        help="Output .onnx path (default: same dir as weights → model.onnx)",
    )
    ex.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: %(default)s)",
    )
    ex.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow dynamic batch size in the exported model",
    )

    qu = subparsers.add_parser(
        "quantize",
        help="Quantize an exported ONNX classifier to int8",
    )
    _add_runs_root_arg(qu)
    qu.add_argument(
        "--weights",
        type=pl.Path,
        default=None,
        help="Input ONNX model (default: promoted 'current' run's onnx/model.onnx under --runs-root)",
    )
    qu.add_argument(
        "--out",
        type=pl.Path,
        default=None,
        help="Output quantized .onnx path (default: same dir → model_quantized.onnx)",
    )
    add_img_root_arg(qu)
    qu.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of calibration images (default: %(default)s)",
    )

    pr = subparsers.add_parser(
        "promote",
        help="Mark a trained run version as the 'current' default for inference/export",
    )
    pr.add_argument("version", type=str, help="Version name under --runs-root to promote")
    _add_runs_root_arg(pr)

    lv = subparsers.add_parser(
        "list-versions",
        help="List trained run versions and which one is current",
    )
    _add_runs_root_arg(lv)

    pp = subparsers.add_parser(
        "prepare-hf",
        help="Stage this model's ONNX export + a generated model card into a local "
        "directory for manual Hugging Face publishing",
    )
    _add_runs_root_arg(pp)
    pp.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version under --runs-root to stage (default: the promoted 'current' version)",
    )
    pp.add_argument(
        "--out-dir",
        type=pl.Path,
        required=True,
        help="Directory to stage files into (e.g. your local clone of the Hugging Face repo)",
    )

    pg = subparsers.add_parser(
        "prepare-github",
        help="Stage this model's ONNX export + a generated model card into a local "
        "directory for manual GitHub publishing",
    )
    _add_runs_root_arg(pg)
    pg.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version under --runs-root to stage (default: the promoted 'current' version)",
    )
    pg.add_argument(
        "--out-dir",
        type=pl.Path,
        required=True,
        help="Directory to stage files into (e.g. a subfolder of your local clone of the GitHub repo)",
    )

    args = parser.parse_args(argv)
    if args.command == "train":
        _train(args)
    elif args.command == "show-batch":
        _show_batch(args)
    elif args.command == "export":
        _export(args)
    elif args.command == "quantize":
        _quantize(args)
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
