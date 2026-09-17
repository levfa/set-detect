import functools
import pathlib as pl
import shutil
from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np

from setdetect.data.card_cutouts import MaskedCardDataset
from setdetect.data.texturecan import TextureDataset
from setdetect.synth_data.board_synth import (
    CardQuad,
    make_image,
)
from setdetect.synth_data.effects import (
    BlurConfig,
    CardCoverConfig,
    HardShadowConfig,
    ShadeConfig,
    ShadowConfig,
    SpecConfig,
    make_effects,
)

_BORDER_MARGIN = 2.0


@dataclass(frozen=True)
class DatasetStats:
    images: int
    labels: int


@dataclass(frozen=True)
class SynthConfig:
    cards_root: pl.Path
    textures_root: pl.Path
    manifest: pl.Path | None
    samples_per_epoch: int = 2000
    num_cards: int = 12
    spec: SpecConfig | None = None
    shade: ShadeConfig | None = None
    shadow: ShadowConfig | None = None
    blur: BlurConfig | None = None
    hard_shadow: HardShadowConfig | None = None
    card_cover: CardCoverConfig | None = None
    rot_range: tuple[float, float] | None = None
    rot_outlier_prob: float = 0.15
    rot_outlier_range: tuple[float, float] = (60, 180)
    rel_min_separation: float = 0.85
    persp: float = 1.25


def _build_generator(cfg: SynthConfig, res: tuple[int, int]) -> functools.partial:
    """Build the ``make_image`` generator for a config.

    Providers are constructed once and captured in the partial; effect configs
    fall back to their defaults in ``make_effects``, so every caller sees the
    same stack.  Pass ``rng=`` at call time to control the random stream.
    """
    cards = MaskedCardDataset(cfg.cards_root)
    textures = TextureDataset(cfg.textures_root, cfg.manifest)
    if not textures:
        raise RuntimeError(f"no textures found in {cfg.textures_root}")
    effects = make_effects(
        cfg.spec, cfg.shade, cfg.shadow, cfg.blur, hard_shadow=cfg.hard_shadow, card_cover=cfg.card_cover
    )
    return functools.partial(
        make_image,
        lambda n, _rng: cards.random_cards(n, _rng),
        lambda _rng: textures.random_texture(_rng),
        num_cards=cfg.num_cards,
        res=res,
        rot_range=cfg.rot_range if cfg.rot_range is not None else (-45, 45),
        rot_outlier_prob=cfg.rot_outlier_prob,
        rot_outlier_range=cfg.rot_outlier_range,
        rel_min_separation=cfg.rel_min_separation,
        persp=cfg.persp,
        card_highlighter=effects.specular_highlighter,
        card_shader=effects.shader,
        shadow_drawer=effects.shadow_drawer,
        hard_shadow_drawer=effects.hard_shadow_drawer,
        card_blurrer=effects.card_blurrer,
        bg_blurrer=effects.bg_blurrer,
        defocus_blurrer=effects.defocus_blurrer,
        card_coverer=effects.card_coverer,
    )


def quads_to_label_data(
    width: int,
    height: int,
    quads: list[CardQuad],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert card quads (pixel coords) into normalized YOLO corner-keypoint labels.

    Drops quads clipped at the image border (``_BORDER_MARGIN``). Returns
    ``(cls, bboxes, keypoints)`` with ``bboxes`` in normalized xywh and
    ``keypoints`` as ``(n, 4, 3)``: normalized corner positions plus a
    visibility flag (``1.0`` for visible corners, ``0.0`` for occluded ones).
    """
    cls_list: list[float] = []
    box_list: list[list[float]] = []
    kp_list: list[np.ndarray] = []
    for quad in quads:
        q = quad.quad
        if not (
            np.all(q[:, 0] >= _BORDER_MARGIN)
            and np.all(q[:, 0] <= width - _BORDER_MARGIN)
            and np.all(q[:, 1] >= _BORDER_MARGIN)
            and np.all(q[:, 1] <= height - _BORDER_MARGIN)
        ):
            continue
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        cls_list.append(0.0)
        box_list.append([(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height])
        kp = np.empty((4, 3), np.float32)
        kp[:, :2] = q / np.array([width, height])
        kp[:, 2] = quad.visible.astype(np.float32)
        kp_list.append(kp)
    cls = np.zeros((len(box_list), 1), np.float32)
    bboxes = np.asarray(box_list, np.float32).reshape(-1, 4)
    keypoints = np.asarray(kp_list, np.float32).reshape(-1, 4, 3)
    return cls, bboxes, keypoints


def _write_label(
    label_dir: pl.Path,
    stem: str,
    quads: list[CardQuad],
    width: int,
    height: int,
) -> int:
    cls, bboxes, keypoints = quads_to_label_data(width, height, quads)
    lines = []
    for c, box, kp in zip(cls, bboxes, keypoints):
        kps = " ".join(f"{px:.6f} {py:.6f} {v:.6f}" for px, py, v in kp)
        lines.append(f"{c.item()} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {kps}")
    if lines:
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return len(lines)


def generate_dataset(
    cfg: SynthConfig,
    out_root: pl.Path,
    num_val: int = 200,
    seed: int = 0,
    res: tuple[int, int] = (640, 640),
) -> DatasetStats:
    """Generate the validation split + data.yaml.

    Only the val split is written to disk: the train split is drawn on the fly
    by ``train``, so this is the dataset preparation step for training.
    """
    generator = _build_generator(cfg, res)

    out_root = pl.Path(out_root)
    image_dir = out_root / "val" / "images"
    label_dir = out_root / "val" / "labels"
    for dir_ in (image_dir, label_dir):
        if dir_.is_dir():
            for path in dir_.iterdir():
                path.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)

    n_labels = 0
    for i in range(num_val):
        rng = np.random.default_rng(seed + i)
        img, quads = generator(rng=rng)
        stem = f"img_{i:06d}"
        cv2.imwrite(str(image_dir / f"{stem}.png"), img)
        height, width = img.shape[:2]
        n_labels += _write_label(label_dir, stem, quads, width, height)
        if (i + 1) % 100 == 0 or (i + 1) == num_val:
            print(f"  [val] generated {i + 1}/{num_val} images ({n_labels} labels)")

    _write_data_yaml(out_root)
    return DatasetStats(images=num_val, labels=n_labels)


def _write_data_yaml(out_root: pl.Path) -> None:
    (out_root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
                # the train split is drawn on the fly by train; only the
                # val split exists on disk (ultralytics still reads the key)
                "train: train/images",
                "val: val/images",
                "names: {0: card}",
                "kpt_shape: [4, 3]",
                # fliplr swaps left/right corners: TL<->TR, BL<->BR
                "flip_idx: [1, 0, 3, 2]",
            ]
        )
        + "\n"
    )


def predict(
    weights: pl.Path,
    image_paths: list[pl.Path],
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        [str(path) for path in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    # (xy, visibility) per image; conf is None for [4,2] heads, always present for [4,3]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for result in results:
        kpts = result.keypoints  # pyright: ignore[reportAttributeAccessIssue]
        xy = kpts.xy.cpu().numpy()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        vis = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones_like(xy)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        out.append((xy, vis))
    return out


def train(
    cfg: SynthConfig,
    data_yaml: pl.Path,
    model: pl.Path,
    epochs: int = 100,
    imgsz: int = 640,
    device: str | None = None,
    project: pl.Path | None = None,
    name: str = "card-corners",
    seed: int = 0,
) -> pl.Path:
    # ultralytics is heavy, so it stays imported lazily inside this function
    # (and the trainer/dataset classes are defined here to keep module import light)
    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.utils import colorstr
    from ultralytics.utils.torch_utils import unwrap_model

    class SyntheticCornerDataset(YOLODataset):
        """YOLO corner-keypoint dataset whose samples are generated fresh on every access.

        ``get_image_and_label`` calls ``make_image`` per request, so each epoch (and each
        mosaic-free batch) draws new synthetic boards. Providers are built lazily on first
        access, keeping the dataset cheap to pickle into DataLoader workers. Each access
        derives an ``rng`` from numpy's global state, which ultralytics' ``seed_worker``
        reseeds per worker and per epoch, so consecutive epochs never repeat a board.
        """

        def __init__(self, cfg: SynthConfig, **kwargs) -> None:
            self.cfg = cfg
            self._generator = None
            super().__init__(**kwargs)

        @property
        def nkpt(self) -> int:
            return self.data.get("kpt_shape", (0, 0))[0]  # pyright: ignore[reportOptionalMemberAccess]

        def _make_generator(self) -> functools.partial:
            return _build_generator(self.cfg, (self.imgsz, self.imgsz))

        def _generate(self) -> tuple[np.ndarray, list]:
            if self._generator is None:
                self._generator = self._make_generator()
            rng = np.random.default_rng()
            return self._generator(rng=rng)

        def get_img_files(self, img_path: str | list[str]) -> list[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
            self.im_files = [f"synth/{i:06d}.jpg" for i in range(self.cfg.samples_per_epoch)]
            return self.im_files

        def get_labels(self) -> list[dict]:
            return [
                {
                    "im_file": im_file,
                    "shape": (self.imgsz, self.imgsz),
                    "cls": np.zeros((0, 1), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "segments": [],
                    "keypoints": np.zeros((0, self.nkpt, 3), dtype=np.float32),
                    "normalized": True,
                    "bbox_format": "xywh",
                }
                for im_file in self.im_files
            ]

        def _touch_buffer(self, i: int) -> None:
            if self.augment:
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    self.buffer.pop(0)

        def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False) -> tuple:
            img, _ = self._generate()
            self._touch_buffer(i)
            return img, img.shape[:2], img.shape[:2]

        def get_image_and_label(self, index: int) -> dict:
            label = deepcopy(self.labels[index])
            img, quads = self._generate()
            self._touch_buffer(index)
            height, width = img.shape[:2]
            label.pop("shape", None)
            label["img"] = img
            label["ori_shape"] = (height, width)
            label["resized_shape"] = (height, width)
            label["ratio_pad"] = (1.0, 1.0)
            cls, bboxes, keypoints = quads_to_label_data(width, height, quads)
            if not len(keypoints):
                keypoints = np.zeros((0, self.nkpt, 3), np.float32)
            label["cls"], label["bboxes"], label["keypoints"] = cls, bboxes, keypoints
            label["segments"] = []
            label["normalized"], label["bbox_format"] = True, "xywh"
            return self.update_labels_info(label)

    class SyntheticCornerTrainer(PoseTrainer):
        """Corner trainer that swaps the training split for on-the-fly synthetic boards.

        Generation parameters are set as a class attribute (``cfg``) before ``engine.train``,
        because ultralytics rejects unknown ``model.train`` kwargs via ``check_dict_alignment``.
        """

        cfg: SynthConfig | None = None

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> YOLODataset:
            if mode == "train" and self.cfg is not None:
                gs = max(int(unwrap_model(self.model).stride.max()), 32)  # pyright: ignore[reportCallIssue, reportArgumentType]
                return SyntheticCornerDataset(
                    img_path=img_path,
                    imgsz=self.args.imgsz,
                    batch_size=batch,
                    augment=True,
                    hyp=self.args,
                    rect=False,
                    stride=gs,
                    pad=0.0,
                    single_cls=self.args.single_cls or False,
                    prefix=colorstr("synth: "),
                    task="pose",
                    data=self.data,
                    cfg=self.cfg,
                )
            return super().build_dataset(img_path, mode, batch)  # pyright: ignore[reportReturnType]

    SyntheticCornerTrainer.cfg = cfg
    engine = YOLO(str(model))
    engine.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        project=str(project) if project is not None else None,
        name=name,
        seed=seed,
        trainer=SyntheticCornerTrainer,
        # rotation/flips come from the synthetic generator, not ultralytics: the
        # labels are canonicalized to the TR convention, so any on-dataset flip
        # (which mirrors board layouts and breaks corner-label order) is unwanted
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    return pl.Path(engine.trainer.best)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]


def export_onnx(
    weights_path: pl.Path,
    out_path: pl.Path | None = None,
    imgsz: int = 640,
) -> pl.Path:
    """Export a trained YOLO corner-keypoint model to ONNX format; return the output path."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    out = pl.Path(model.export(format="onnx", imgsz=imgsz))
    if out_path is None:
        out_path = out.with_name("card_corners.onnx")
    if out_path != out:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(out_path))
    return out_path
