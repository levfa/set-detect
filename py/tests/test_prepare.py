import pytest

from setdetect.model import prepare


def test_render_model_card_frontmatter_and_body():
    card = prepare.render_model_card(
        frontmatter={"license": "agpl-3.0", "tags": ["a", "b"]},
        body="# Title\n\nBody text.",
    )
    lines = card.splitlines()
    assert lines[0] == "---"
    assert "license: agpl-3.0" in lines
    assert "tags:" in lines
    assert "  - a" in lines
    assert "  - b" in lines
    assert lines[lines.index("---", 1)] == "---"
    assert card.endswith("# Title\n\nBody text.")


def test_stage_files_copies_and_creates_parents(tmp_path):
    src = tmp_path / "source.onnx"
    src.write_bytes(b"fake-onnx-bytes")
    out_dir = tmp_path / "out"

    prepare.stage_files(out_dir, [(src, "onnx/model.onnx")])

    dest = out_dir / "onnx" / "model.onnx"
    assert dest.is_file()
    assert dest.read_bytes() == b"fake-onnx-bytes"


def test_write_text_writes_content_and_creates_parents(tmp_path):
    out_dir = tmp_path / "out"
    prepare.write_text(out_dir, "README.md", "hello world")
    dest = out_dir / "README.md"
    assert dest.is_file()
    assert dest.read_text() == "hello world"


def _spec(**overrides) -> prepare.ModelSpec:
    fields = {
        "cli": "set-card-classification",
        "onnx_files": ("model.onnx", "model_quantized.onnx"),
        "card_frontmatter": {"license": "agpl-3.0"},
        "card_body": lambda: "# Card",
        "android_file": "model_quantized.onnx",
        "android_asset": "card_classifier_quant.onnx",
    }
    return prepare.ModelSpec(**{**fields, **overrides})


def test_prepare_repo_stages_present_files_and_reports_missing_optional_ones(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "onnx").mkdir(parents=True)
    (run_dir / "onnx" / "model.onnx").write_bytes(b"fp32")
    out_dir = tmp_path / "out"

    missing = prepare.prepare_repo(_spec(), run_dir, out_dir)

    assert missing == ["model_quantized.onnx"]
    assert (out_dir / "onnx" / "model.onnx").read_bytes() == b"fp32"
    assert (out_dir / "README.md").read_text().endswith("# Card")


def test_prepare_repo_requires_the_first_file(tmp_path):
    with pytest.raises(SystemExit):
        prepare.prepare_repo(_spec(), tmp_path / "run", tmp_path / "out")


def test_prepare_android_keeps_the_other_model(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "onnx").mkdir(parents=True)
    (run_dir / "onnx" / "model_quantized.onnx").write_bytes(b"classifier")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "card_corners.onnx").write_bytes(b"corners")
    (assets / "VERSION.txt").write_text("card_corners: v03-20260916\ncard_classifier_quant: v02-20260817\n")

    prepare.prepare_android(_spec(), run_dir, "v07-20261013", assets)

    assert (assets / "card_corners.onnx").read_bytes() == b"corners"
    assert (assets / "card_classifier_quant.onnx").read_bytes() == b"classifier"
    assert (assets / "VERSION.txt").read_text() == "card_classifier_quant: v07-20261013\ncard_corners: v03-20260916\n"
