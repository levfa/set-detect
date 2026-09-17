import importlib

import pytest

TOP_LEVEL_MODULES = [
    "setdetect.cli.card_cutouts_data",
    "setdetect.cli.backgrounds_data",
    "setdetect.cli.board_synth",
    "setdetect.cli.card_synth",
    "setdetect.cli.card_corners",
    "setdetect.cli.card_classification",
    "setdetect.cli.card_class_data",
    "setdetect.cli.card_detection",
    "setdetect.cli.card_tracking",
]


@pytest.mark.parametrize("module_name", TOP_LEVEL_MODULES)
def test_missing_subcommand_exits_with_usage_error(module_name):
    mod = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc_info:
        mod.main([])
    assert exc_info.value.code == 2


def test_sync_android_assets_missing_promoted_version_raises(tmp_path):
    from setdetect.cli.sync_android_assets import main

    with pytest.raises(SystemExit):
        main(
            [
                "--corner-runs-root",
                str(tmp_path / "corner_runs"),
                "--class-runs-root",
                str(tmp_path / "class_runs"),
                "--assets-dir",
                str(tmp_path / "assets"),
            ]
        )


def test_card_detection_show_boards_missing_weights_is_argparse_error(tmp_path):
    from setdetect.cli.card_detection import main

    with pytest.raises(SystemExit) as exc_info:
        main(["show-boards", "--data-root", str(tmp_path)])
    assert exc_info.value.code == 2


def test_card_corners_list_versions_on_empty_dir(tmp_path, capsys):
    from setdetect.cli.card_corners import main

    main(["list-versions", "--runs-root", str(tmp_path / "empty")])
    assert capsys.readouterr().out == ""


def test_card_classification_list_versions_on_empty_dir(tmp_path, capsys):
    from setdetect.cli.card_classification import main

    main(["list-versions", "--runs-root", str(tmp_path / "empty")])
    assert capsys.readouterr().out == ""
