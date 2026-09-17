import json
import pathlib as pl
import typing as tp
from dataclasses import dataclass

from setdetect.data.texturecan import IMAGE_SUFFIXES

RecordKey = tuple[str, tuple[tuple[int, int], ...]]


def list_images(root: pl.Path) -> list[pl.Path]:
    """Raw photos under a dataset root's ``raw/`` subdirectory, sorted by filename."""
    raw_dir = pl.Path(root) / "raw"
    return [path for path in sorted(raw_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def labels_file(root: pl.Path) -> pl.Path:
    """Path of the single card-class labels file for the whole dataset."""
    return pl.Path(root) / "labels" / "card-classes.jsonl"


STATUS_SET = "set"
STATUS_NOT_SET = "not-set"
STATUS_GARBAGE = "garbage"
STATUS_UNFINISHED = "unfinished"


@dataclass
class CardLabel:
    """One labeled card detection.

    ``path`` is relative to the dataset root. ``quad`` holds the four corners in
    original image pixels as predicted. ``status`` is always set: "set" (all four
    attributes), "not-set" (a real detection that is not a SET card), "garbage"
    (detection to be ignored), or "unfinished" (partially labeled).
    """

    path: str
    w: int
    h: int
    quad: list[list[float]]
    count: str | None
    color: str | None
    fill: str | None
    shape: str | None
    status: str


Labels = dict[RecordKey, CardLabel]


def quad_key(quad: tp.Iterable[tp.Iterable[float]]) -> tuple[tuple[int, int], ...]:
    """Int-rounded corner sequence used to re-identify a detection; not stored."""
    return tuple((int(round(float(x))), int(round(float(y)))) for x, y in quad)


def record_key(label: CardLabel) -> RecordKey:
    return (label.path, quad_key(label.quad))


def _to_dict(label: CardLabel) -> dict[str, tp.Any]:
    return {
        "path": label.path,
        "w": label.w,
        "h": label.h,
        "quad": [[float(x), float(y)] for x, y in label.quad],
        "count": label.count,
        "color": label.color,
        "fill": label.fill,
        "shape": label.shape,
        "status": label.status,
    }


def _infer_status(record: dict[str, tp.Any]) -> str:
    if record.get("status") is not None:
        return record["status"]
    attrs = [record.get("count"), record.get("color"), record.get("fill"), record.get("shape")]
    if all(attr is not None for attr in attrs):
        return STATUS_SET
    if all(attr is None for attr in attrs):
        return STATUS_NOT_SET
    return STATUS_UNFINISHED


def _from_dict(record: dict[str, tp.Any]) -> CardLabel:
    return CardLabel(
        path=record["path"],
        w=int(record["w"]),
        h=int(record["h"]),
        quad=[[float(x), float(y)] for x, y in record["quad"]],
        count=record.get("count"),
        color=record.get("color"),
        fill=record.get("fill"),
        shape=record.get("shape"),
        status=_infer_status(record),
    )


def load_card_classes(path: pl.Path) -> Labels:
    """Load the card-class labels file (empty dict if missing)."""
    path = pl.Path(path)
    labels: Labels = {}
    if not path.is_file():
        return labels
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        label = _from_dict(json.loads(line))
        labels[record_key(label)] = label
    return labels


def save_card_classes(path: pl.Path, labels: Labels) -> None:
    """Write the card-class labels file, one JSON record per line."""
    path = pl.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for label in labels.values():
            file.write(json.dumps(_to_dict(label)) + "\n")
