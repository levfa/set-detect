import json
import pathlib as pl
import re
import typing as tp
from dataclasses import dataclass

import cv2
import huggingface_hub as hf
import numpy as np
import PIL.Image as pil_img
import PIL.ImageDraw as pil_draw
import PIL.ImageFont as pil_font

REPO_ID = "nyuuzyou/texturecan"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

DEFAULT_MANIFEST_NAME = ".backgrounds.jsonl"

_CATEGORY_RE = re.compile(r"^([a-z_]+?)_\d")


def download(local_dir: pl.Path, revision: str = "main") -> None:
    hf.snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )


def _scan_images(root: pl.Path) -> list[pl.Path]:
    return [image_path for image_path in sorted(root.rglob("*")) if image_path.suffix.lower() in IMAGE_SUFFIXES]


@dataclass(frozen=True)
class TextureFile:
    image_path: pl.Path


@dataclass(frozen=True)
class BackgroundEntry:
    uuid: str
    rel_path: str
    category: str
    mean_lum: float


def category_from_texture_set(texture_set: str) -> str:
    match = _CATEGORY_RE.match(texture_set)
    return match.group(1) if match else "other"


def read_category(image_path: pl.Path) -> str:
    json_path = image_path.with_suffix(".json")
    try:
        data = json.loads(json_path.read_text())
    except (OSError, ValueError):
        return "other"
    return category_from_texture_set(data.get("texture_set", ""))


def build_backgrounds(
    root: pl.Path,
    out: pl.Path | None = None,
    max_mean_lum: float = 215.0,
    min_dim: int = 512,
    cap_per_category: int = 100,
) -> list[BackgroundEntry]:
    root = pl.Path(root)
    by_category: dict[str, list[BackgroundEntry]] = {}
    for image_path in _scan_images(root):
        with pil_img.open(image_path) as im:
            width, height = im.size
            gray = np.asarray(im.convert("L"), dtype=np.float32)
        if min(width, height) < min_dim:
            continue
        lum = float(gray.mean())
        if lum > max_mean_lum:
            continue
        entry = BackgroundEntry(
            uuid=image_path.stem,
            rel_path=image_path.relative_to(root).as_posix(),
            category=read_category(image_path),
            mean_lum=lum,
        )
        by_category.setdefault(entry.category, []).append(entry)

    entries: list[BackgroundEntry] = []
    for category in sorted(by_category):
        members = sorted(by_category[category], key=lambda entry: entry.uuid)
        entries.extend(members[:cap_per_category])
    entries.sort(key=lambda entry: (entry.category, entry.uuid))

    if out is not None:
        out = pl.Path(out)
        with out.open("w") as file:
            for entry in entries:
                file.write(
                    json.dumps(
                        {
                            "uuid": entry.uuid,
                            "rel_path": entry.rel_path,
                            "category": entry.category,
                            "mean_lum": round(entry.mean_lum, 1),
                        }
                    )
                    + "\n"
                )
    return entries


def load_backgrounds(manifest: pl.Path) -> list[BackgroundEntry]:
    entries: list[BackgroundEntry] = []
    for line in pl.Path(manifest).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(
            BackgroundEntry(
                data["uuid"],
                data["rel_path"],
                data["category"],
                float(data["mean_lum"]),
            )
        )
    return entries


class TextureDataset:
    def __init__(self, root: pl.Path, manifest: pl.Path | None = None) -> None:
        self.root = pl.Path(root)
        self._textures = self._scan(self.root, manifest)

    @staticmethod
    def _scan(root: pl.Path, manifest: pl.Path | None) -> list[TextureFile]:
        textures = [TextureFile(image_path) for image_path in _scan_images(root)]
        if manifest is None:
            return textures
        allowed = {entry.uuid for entry in load_backgrounds(manifest)}
        return [texture for texture in textures if texture.image_path.stem in allowed]

    def random_texture(self, rng: np.random.Generator) -> np.ndarray:
        if not self:
            raise RuntimeError("texture dataset is empty")
        texture_file = self[int(rng.integers(len(self)))]
        texture = cv2.imread(str(texture_file.image_path))
        if texture is None:
            raise RuntimeError(f"Failed to read texture image: {texture_file.image_path}")
        return texture

    def __len__(self) -> int:
        return len(self._textures)

    @tp.overload
    def __getitem__(self, index: int) -> TextureFile: ...

    @tp.overload
    def __getitem__(self, index: slice) -> list[TextureFile]: ...

    def __getitem__(self, index: int | slice) -> TextureFile | list[TextureFile]:
        return self._textures[index]

    def __iter__(self) -> tp.Iterator[TextureFile]:
        return iter(self._textures)


def render_contact_sheet(
    entries: list[tuple[int, BackgroundEntry]],
    root: pl.Path,
    title: str,
    cols: int = 10,
    thumb: int = 128,
) -> pil_img.Image:
    pad = 4
    label_h = 26
    header_h = 24 if title else 0
    if not entries:
        return pil_img.new("RGB", (1, 1), (25, 25, 25))
    rows = (len(entries) + cols - 1) // cols
    width = cols * thumb + (cols + 1) * pad
    height = header_h + rows * (thumb + label_h) + (rows + 1) * pad
    sheet = pil_img.new("RGB", (width, height), (25, 25, 25))
    draw = pil_draw.Draw(sheet)
    font = pil_font.load_default()
    if title:
        draw.text((pad, pad), title, fill=(255, 255, 255), font=font)
    for position, (index, entry) in enumerate(entries):
        row, col = divmod(position, cols)
        x = pad + col * (thumb + pad)
        y = header_h + pad + row * (thumb + label_h + pad)
        try:
            with pil_img.open(root / entry.rel_path) as im:
                tile = im.convert("RGB")
                tile.thumbnail((thumb, thumb))
            sheet.paste(
                tile,
                (x + (thumb - tile.width) // 2, y + (thumb - tile.height) // 2),
            )
        except OSError:
            draw.rectangle(
                [x, y, x + thumb, y + thumb],
                fill=(60, 0, 0),
            )
        draw.text(
            (x + 2, y + thumb + 3),
            f"#{index} {entry.uuid[:8]}",
            fill=(255, 255, 255),
            font=font,
        )
    return sheet
