import pathlib as pl

import pytest

from setdetect.cli import board_synth
from setdetect.synth_data.scenarios import SCENARIOS

CARDS_ROOT = pl.Path.home() / "data/set-cards/img-real-card-cutouts"


@pytest.mark.skipif(not CARDS_ROOT.is_dir(), reason="masked card dataset not available")
def test_generate_writes_all_scenario_sheets(tmp_path):
    board_synth.main(["generate", "--cards-root", str(CARDS_ROOT), "--out", str(tmp_path)])
    assert len(list(tmp_path.glob("*.png"))) == len(SCENARIOS)
