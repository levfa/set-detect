#!/usr/bin/env bash
# Downloads the published inference models (github.com/levfa/set-detect-models) into
# SETDETECT_MODELS_ROOT (default ~/.cache/setdetect/models), skipping files already
# present. Same repo, same cache layout as py/src/setdetect/model/models_repo.py, so
# both languages' tests can share one downloaded cache.
set -euo pipefail

REPO="levfa/set-detect-models"
REF="${SETDETECT_MODELS_REF:-main}"
ROOT="${SETDETECT_MODELS_ROOT:-$HOME/.cache/setdetect/models}"

FILES=(
    "card-corners/onnx/card_corners.onnx"
    "card-classification/onnx/model.onnx"
    "card-classification/onnx/model.onnx.data"
)

for path_in_repo in "${FILES[@]}"; do
    dest="$ROOT/$path_in_repo"
    if [[ -f "$dest" ]]; then
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    url="https://raw.githubusercontent.com/$REPO/$REF/$path_in_repo"
    echo "downloading $url -> $dest"
    curl -fsSL -o "$dest" "$url"
done
