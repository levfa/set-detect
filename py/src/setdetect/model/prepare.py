import dataclasses
import pathlib as pl
import shutil
import typing as tp

DEFAULT_ANDROID_ASSETS = pl.Path(__file__).resolve().parents[4] / "android/app/src/main/assets/models"

_VERSION_FILE = "VERSION.txt"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """What a model family hands off: its exported files, its model card, and its Android asset."""

    cli: str
    # Files under a run's onnx/ directory; the first is required, the rest are staged if present.
    onnx_files: tuple[str, ...]
    card_frontmatter: dict[str, str | list[str]]
    card_body: tp.Callable[[], str]
    # The onnx/ file shipped with the Android app, and its name there.
    android_file: str
    android_asset: str


def render_model_card(*, frontmatter: dict[str, str | list[str]], body: str) -> str:
    """Hand-format a README.md string: a YAML frontmatter block followed by a
    Markdown body."""
    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n" + body


def stage_files(out_dir: pl.Path, files: list[tuple[pl.Path, str]]) -> None:
    """Copy each (local_path, path_in_repo) pair into out_dir, creating parent dirs
    as needed. Only ever adds/overwrites the given files, never deletes or
    otherwise touches anything else already in out_dir."""
    for local_path, path_in_repo in files:
        dest = out_dir / path_in_repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)


def write_text(out_dir: pl.Path, path_in_repo: str, content: str) -> None:
    """Write a generated text file (README.md, config.json) into out_dir/path_in_repo."""
    dest = out_dir / path_in_repo
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)


def _exported(spec: ModelSpec, run_dir: pl.Path, name: str) -> pl.Path:
    path = run_dir / "onnx" / name
    if not path.is_file():
        raise SystemExit(f"no ONNX export at {path}; run '{spec.cli} export' for this version first")
    return path


def prepare_repo(spec: ModelSpec, run_dir: pl.Path, out_dir: pl.Path) -> list[str]:
    """Stage a run's ONNX files and generated model card into a model repository checkout.

    Returns the names of the optional files that were not found.
    """
    required, *optional = spec.onnx_files
    files = [(_exported(spec, run_dir, required), f"onnx/{required}")]
    missing = []
    for name in optional:
        path = run_dir / "onnx" / name
        if path.is_file():
            files.append((path, f"onnx/{name}"))
        else:
            missing.append(name)

    stage_files(out_dir, files)
    write_text(out_dir, "README.md", render_model_card(frontmatter=spec.card_frontmatter, body=spec.card_body()))
    return missing


def prepare_android(spec: ModelSpec, run_dir: pl.Path, version: str, out_dir: pl.Path) -> None:
    """Copy a run's model into the Android assets dir and record its version.

    Only this model's file and its line in VERSION.txt change; other models there are kept.
    """
    src = _exported(spec, run_dir, spec.android_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_dir / spec.android_asset)

    key = pl.Path(spec.android_asset).stem
    version_file = out_dir / _VERSION_FILE
    lines = version_file.read_text().splitlines() if version_file.is_file() else []
    lines = [line for line in lines if line.split(":", 1)[0].strip() != key]
    lines.append(f"{key}: {version}")
    version_file.write_text("\n".join(sorted(lines)) + "\n")
