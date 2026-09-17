import argparse
import pathlib as pl
import shutil

from setdetect.model import versioning

DEFAULT_ANDROID_ASSETS = pl.Path(__file__).resolve().parents[4] / "android/app/src/main/assets/models"


def sync(corner_runs_root: pl.Path, class_runs_root: pl.Path, assets_dir: pl.Path) -> None:
    """Copy the promoted corner + classifier ONNX models into the Android assets dir."""
    corner_src = versioning.resolve_current(corner_runs_root) / "onnx" / "card_corners.onnx"
    class_src = versioning.resolve_current(class_runs_root) / "onnx" / "model_quantized.onnx"
    if not corner_src.is_file():
        raise SystemExit(f"missing {corner_src}")
    if not class_src.is_file():
        raise SystemExit(f"missing {class_src}")

    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corner_src, assets_dir / "card_corners.onnx")
    shutil.copy2(class_src, assets_dir / "card_classifier_quant.onnx")

    corner_version = versioning.current_version(corner_runs_root)
    class_version = versioning.current_version(class_runs_root)
    (assets_dir / "VERSION.txt").write_text(f"card_corners: {corner_version}\ncard_classifier_quant: {class_version}\n")
    print(f"synced card_corners ({corner_version}) and card_classifier_quant ({class_version}) to {assets_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="set-sync-android-assets",
        description="Copy the promoted corner + classifier ONNX models into the Android app's assets",
    )
    parser.add_argument(
        "--corner-runs-root",
        type=pl.Path,
        required=True,
        help="Corner model runs directory",
    )
    parser.add_argument(
        "--class-runs-root",
        type=pl.Path,
        required=True,
        help="Classifier runs directory",
    )
    parser.add_argument(
        "--assets-dir",
        type=pl.Path,
        default=DEFAULT_ANDROID_ASSETS,
        help="Android assets/models directory (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    sync(args.corner_runs_root, args.class_runs_root, args.assets_dir)


if __name__ == "__main__":
    main()
