from setdetect.data.img_dir import (
    STATUS_NOT_SET,
    STATUS_SET,
    STATUS_UNFINISHED,
    CardLabel,
    _infer_status,
    list_images,
    load_card_classes,
    quad_key,
    record_key,
    save_card_classes,
)


def test_quad_key_rounds_coordinates():
    quad = [[1.4, 2.6], [10.5, 0.49], [5.0, 5.0], [0.0, 0.0]]
    assert quad_key(quad) == ((1, 3), (10, 0), (5, 5), (0, 0))


def test_infer_status_all_attrs_set():
    record = {"count": "one", "color": "red", "fill": "open", "shape": "diamond"}
    assert _infer_status(record) == STATUS_SET


def test_infer_status_all_attrs_none():
    record = {"count": None, "color": None, "fill": None, "shape": None}
    assert _infer_status(record) == STATUS_NOT_SET


def test_infer_status_partial_is_unfinished():
    record = {"count": "one", "color": None, "fill": None, "shape": None}
    assert _infer_status(record) == STATUS_UNFINISHED


def test_infer_status_explicit_status_passthrough():
    record = {"status": "garbage", "count": None, "color": None, "fill": None, "shape": None}
    assert _infer_status(record) == "garbage"


def test_load_card_classes_missing_file_returns_empty(tmp_path):
    assert load_card_classes(tmp_path / "missing.jsonl") == {}


def test_save_and_load_card_classes_round_trip(tmp_path):
    label = CardLabel(
        path="raw/photo1.jpg",
        w=100,
        h=200,
        quad=[[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]],
        count="one",
        color="red",
        fill="open",
        shape="diamond",
        status=STATUS_SET,
    )
    labels = {record_key(label): label}
    path = tmp_path / "labels" / "card-classes.jsonl"
    save_card_classes(path, labels)

    loaded = load_card_classes(path)
    assert loaded == labels


def test_list_images_filters_by_suffix_and_sorts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "b.png").write_bytes(b"")
    (raw / "a.jpg").write_bytes(b"")
    (raw / "notes.txt").write_bytes(b"")
    (raw / "c.JPEG").write_bytes(b"")

    images = list_images(tmp_path)
    assert [p.name for p in images] == ["a.jpg", "b.png", "c.JPEG"]
