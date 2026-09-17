from setdetect.cli.card_detection import main


def test_profile_arrangement_estimation_runs_end_to_end(capsys):
    main(["profile-arrangement-estimation", "--cards", "4", "--runs", "2", "--seed", "0"])
    out = capsys.readouterr().out
    assert "cards: 4  runs: 2  seed: 0" in out
    assert "total arrangement" in out
    assert "z-order" in out
