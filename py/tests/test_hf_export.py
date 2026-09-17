from setdetect.model import hf_export


def test_render_model_card_frontmatter_and_body():
    card = hf_export.render_model_card(
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

    hf_export.stage_files(out_dir, [(src, "onnx/model.onnx")])

    dest = out_dir / "onnx" / "model.onnx"
    assert dest.is_file()
    assert dest.read_bytes() == b"fake-onnx-bytes"


def test_write_text_writes_content_and_creates_parents(tmp_path):
    out_dir = tmp_path / "out"
    hf_export.write_text(out_dir, "README.md", "hello world")
    dest = out_dir / "README.md"
    assert dest.is_file()
    assert dest.read_text() == "hello world"
