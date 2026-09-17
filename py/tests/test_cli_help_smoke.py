import importlib

import pytest

CLI_HELP_CASES = [
    ("setdetect.cli.card_cutouts_data", ["--help"]),
    ("setdetect.cli.card_cutouts_data", ["show-raw", "--help"]),
    ("setdetect.cli.backgrounds_data", ["--help"]),
    ("setdetect.cli.board_synth", ["--help"]),
    ("setdetect.cli.card_synth", ["--help"]),
    ("setdetect.cli.card_corners", ["--help"]),
    ("setdetect.cli.card_corners", ["list-versions", "--help"]),
    ("setdetect.cli.card_classification", ["--help"]),
    ("setdetect.cli.card_classification", ["list-versions", "--help"]),
    ("setdetect.cli.card_class_data", ["--help"]),
    ("setdetect.cli.card_detection", ["--help"]),
    ("setdetect.cli.card_detection", ["profile-arrangement-estimation", "--help"]),
    ("setdetect.cli.card_tracking", ["--help"]),
    ("setdetect.cli.sync_android_assets", ["--help"]),
]


@pytest.mark.parametrize("module_name,argv", CLI_HELP_CASES, ids=[f"{m}:{' '.join(a)}" for m, a in CLI_HELP_CASES])
def test_help_exits_zero_without_side_effects(module_name, argv, capsys):
    mod = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc_info:
        mod.main(argv)
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()
