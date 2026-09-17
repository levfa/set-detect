import os
import pathlib as pl
import urllib.request

REPO = "levfa/set-detect-models"
DEFAULT_REF = "main"

# name -> path within the set-detect-models repo
FILES: dict[str, str] = {
    "card_corners": "card-corners/onnx/card_corners.onnx",
    "card_classification": "card-classification/onnx/model.onnx",
    "card_classification_data": "card-classification/onnx/model.onnx.data",
}


def default_root() -> pl.Path:
    """Local cache directory for downloaded models, overridable via SETDETECT_MODELS_ROOT."""
    override = os.environ.get("SETDETECT_MODELS_ROOT")
    if override:
        return pl.Path(override)
    return pl.Path.home() / ".cache" / "setdetect" / "models"


def download(local_dir: pl.Path, ref: str = DEFAULT_REF) -> dict[str, pl.Path]:
    """Download the published inference models into local_dir, skipping files
    already present. Returns the name -> local path mapping from FILES."""
    paths: dict[str, pl.Path] = {}
    for name, path_in_repo in FILES.items():
        dest = local_dir / path_in_repo
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{path_in_repo}"
            urllib.request.urlretrieve(url, dest)
        paths[name] = dest
    return paths
