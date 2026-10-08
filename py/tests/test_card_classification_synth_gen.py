import json
import pathlib as pl

import cv2
import numpy as np
import pytest

from setdetect.cli import card_classification
from setdetect.data.texturecan import DEFAULT_MANIFEST_NAME

CARDS_ROOT = pl.Path(__file__).resolve().parents[2] / "examples/data/img-real-card-cutouts"


@pytest.mark.skipif(not (CARDS_ROOT / "masked").is_dir(), reason="masked card dataset not available")
def test_synth_gen_writes_labeled_dataset(tmp_path):
    from setdetect.model.card_classification import CardClassDataset

    tex_root = tmp_path / "textures"
    tex_root.mkdir()
    cv2.imwrite(str(tex_root / "tex.png"), np.random.default_rng(0).integers(0, 255, (256, 256, 3), np.uint8))
    (tex_root / DEFAULT_MANIFEST_NAME).write_text(
        json.dumps({"uuid": "tex", "rel_path": "tex.png", "category": "noise", "mean_lum": 127.0}) + "\n"
    )
    data_root = tmp_path / "data"

    card_classification.main(
        [
            "synth-gen",
            "--data-root",
            str(data_root),
            "--cards-root",
            str(CARDS_ROOT),
            "--tex-root",
            str(tex_root),
            "--num-val",
            "2",
        ]
    )

    assert len(list((data_root / "raw").glob("*.png"))) == 2
    assert len(CardClassDataset(data_root)) >= 2
