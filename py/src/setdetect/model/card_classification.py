import dataclasses
import pathlib as pl

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

import setdetect.data.img_dir as img_dir
from setdetect.data.card_cutouts import MaskedCardDataset
from setdetect.data.texturecan import TextureDataset
from setdetect.synth_data.card_synth import (
    KIND_CARD,
    CardAugConfig,
    ExtractedCard,
    add_false_positive_quads,
    extract_cards,
    jitter_quads,
    make_board,
    make_card_augmenter,
)
from setdetect.synth_data.effects import Effects, make_effects

OTHER = 3
COUNT_MAP = {"one": 0, "two": 1, "three": 2, "other": OTHER}
COLOR_MAP = {"red": 0, "green": 1, "purple": 2, "other": OTHER}
FILL_MAP = {"open": 0, "solid": 1, "striped": 2, "other": OTHER}
SHAPE_MAP = {"diamond": 0, "oval": 1, "squiggle": 2, "other": OTHER}

COUNT_INV = {v: k for k, v in COUNT_MAP.items()}
COLOR_INV = {v: k for k, v in COLOR_MAP.items()}
FILL_INV = {v: k for k, v in FILL_MAP.items()}
SHAPE_INV = {v: k for k, v in SHAPE_MAP.items()}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CARD_SIZE: tuple[int, int] = (224, 320)  # width, height — portrait SET card (5:7 → 7:10)


class SetCardClassifier(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

        self.count_head = nn.Linear(self.feature_dim, 4)
        self.shape_head = nn.Linear(self.feature_dim, 4)
        self.fill_head = nn.Linear(self.feature_dim, 4)
        self.color_head = nn.Linear(self.feature_dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)

        return {
            "count": self.count_head(features),
            "color": self.color_head(features),
            "fill": self.fill_head(features),
            "shape": self.shape_head(features),
        }


def _warp_card(
    img: np.ndarray,
    quad: np.ndarray,
    size: tuple[int, int] = CARD_SIZE,
    crop_frac: float = 0.06,
) -> np.ndarray:
    """Perspective-warp a card region to a canonical rectangle.

    Applies the same crop-then-resize as ``extract_cards`` so that
    synthetic training and real validation cards are processed identically.
    """
    w, h = size
    src = quad.astype(np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    cw = max(int(round(crop_frac * w)), 0)
    ch = max(int(round(crop_frac * h)), 0)
    cropped = warped[ch : h - ch, cw : w - cw]
    if cropped.size == 0:
        cropped = warped
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _card_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 card image to a normalised float tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img_rgb)
    return transforms.Normalize(mean=MEAN, std=STD)(tensor)


def _card_labels(card: ExtractedCard) -> tuple[int, int, int, int]:
    """Return (count, color, fill, shape) index tuple for one card."""
    if card.kind == KIND_CARD and card.label is not None:
        count, color, shape, fill = card.label
        return (
            COUNT_MAP[count],
            COLOR_MAP[color],
            FILL_MAP[fill],
            SHAPE_MAP[shape],
        )
    return (OTHER, OTHER, OTHER, OTHER)


def _default_effects() -> Effects:
    """The full shared effect stack, plus the classifier-specific photometric augmenter.

    ``make_effects()`` covers everything ``board_synth.make_image`` knows how to do
    (specular, ambient shading, per-card cast shadow, whole-board hard shadow, blur, card
    cover); crops are extracted from *jittered* quads (``jitter_quads``, simulating real
    corner-detector imprecision), so effects that sit just outside a card's exact edge --
    the cast shadow, background blur -- can still land inside a training crop and are worth
    keeping, not just the effects that are always strictly inside it. ``augmenter`` is the
    one slot ``make_effects()`` never populates itself, so it's added here.
    """
    return dataclasses.replace(make_effects(), augmenter=make_card_augmenter(CardAugConfig()))


class SyntheticCardDataset(torch.utils.data.Dataset):
    """Generate board images on the fly for classifier training.

    Each ``__getitem__`` call produces one synthetic board and returns **all**
    card crops from that board as batched tensors.
    """

    def __init__(
        self,
        cards_root: pl.Path,
        tex_root: pl.Path,
        boards_per_epoch: int = 256,
        effects: Effects | None = None,
        seed: int = 0,
    ) -> None:
        self.cards = MaskedCardDataset(cards_root)
        self.tex = TextureDataset(tex_root)
        self.boards_per_epoch = boards_per_epoch
        self.effects = effects if effects is not None else _default_effects()
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.boards_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        board_seed = self.seed + self._epoch * self.boards_per_epoch + idx
        rng = np.random.default_rng(board_seed)

        img, quads = make_board(
            self.cards,
            self.tex.random_texture,
            num_cards=8,
            num_fake=4,
            effects=self.effects,
            min_vis=0.7,
            rng=rng,
        )
        add_false_positive_quads(quads, num_rot=3, num_edge=3, rng=rng)
        jitter_quads(quads, frac=0.06, rng=rng)
        crops = extract_cards(img, quads, CARD_SIZE)

        images, count, color, fill, shape = [], [], [], [], []
        for card in crops:
            images.append(_card_to_tensor(card.image))
            c, co, f, sh = _card_labels(card)
            count.append(c)
            color.append(co)
            fill.append(f)
            shape.append(sh)

        return (
            torch.stack(images),
            torch.tensor(count, dtype=torch.long),
            torch.tensor(color, dtype=torch.long),
            torch.tensor(fill, dtype=torch.long),
            torch.tensor(shape, dtype=torch.long),
        )


def _collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten board-level batches into a single flat batch."""
    images, count, color, fill, shape = [], [], [], [], []
    for imgs, c, co, f, sh in batch:
        images.append(imgs)
        count.append(c)
        color.append(co)
        fill.append(f)
        shape.append(sh)
    return (
        torch.cat(images),
        torch.cat(count),
        torch.cat(color),
        torch.cat(fill),
        torch.cat(shape),
    )


class CardClassDataset(torch.utils.data.Dataset):
    """Labeled card photos for classifier validation."""

    def __init__(
        self,
        data_root: pl.Path,
        size: tuple[int, int] = CARD_SIZE,
    ) -> None:
        self.data_root = pl.Path(data_root)
        self.size = size

        labels_path = img_dir.labels_file(self.data_root)
        all_labels = img_dir.load_card_classes(labels_path)
        valid_statuses = (img_dir.STATUS_SET, img_dir.STATUS_NOT_SET)
        self.samples = [label for label in all_labels.values() if label.status in valid_statuses]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int, int]:
        label = self.samples[idx]

        img_path = self.data_root / label.path
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"could not read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        quad = np.array(label.quad, dtype=np.float32)
        card = _warp_card(img, quad, self.size)

        tensor = transforms.ToTensor()(card)
        tensor = transforms.Normalize(mean=MEAN, std=STD)(tensor)

        is_set = label.status == img_dir.STATUS_SET
        count_idx = COUNT_MAP[label.count] if is_set and label.count else OTHER
        color_idx = COLOR_MAP[label.color] if is_set and label.color else OTHER
        fill_idx = FILL_MAP[label.fill] if is_set and label.fill else OTHER
        shape_idx = SHAPE_MAP[label.shape] if is_set and label.shape else OTHER

        return tensor, count_idx, color_idx, fill_idx, shape_idx


HEADS = ("count", "color", "fill", "shape")


def _evaluate(
    model: SetCardClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Run one validation pass; return (avg_loss, {head: accuracy})."""
    model.eval()
    criteria = {h: nn.CrossEntropyLoss() for h in HEADS}

    total_loss = 0.0
    correct: dict[str, int] = {h: 0 for h in HEADS}
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            targets = {h: batch[i + 1].to(device) for i, h in enumerate(HEADS)}

            outputs = model(images)
            loss = torch.stack([criteria[h](outputs[h], targets[h]) for h in HEADS]).sum()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            for h in HEADS:
                correct[h] += (outputs[h].argmax(dim=1) == targets[h]).sum().item()

    avg_loss = total_loss / max(total, 1)
    accs = {h: correct[h] / max(total, 1) for h in HEADS}
    return avg_loss, accs


def train(  # noqa: A001
    data_root: pl.Path,
    cards_root: pl.Path,
    tex_root: pl.Path,
    epochs: int,
    batch_size: int,
    lr: float,
    out_dir: pl.Path,
    boards_per_epoch: int,
    seed: int,
    device: str | None,
    num_workers: int,
    pretrained: bool,
    effects: Effects | None = None,
) -> None:
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"device: {dev}")

    train_ds = SyntheticCardDataset(
        cards_root,
        tex_root,
        boards_per_epoch=boards_per_epoch,
        effects=effects,
        seed=seed,
    )
    val_ds = CardClassDataset(data_root)
    if len(val_ds) < 2:
        raise ValueError(f"need at least 2 labeled cards in {img_dir.labels_file(data_root)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )

    model = SetCardClassifier(pretrained=pretrained).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criteria = {h: nn.CrossEntropyLoss() for h in HEADS}

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    best_val_loss = float("inf")

    def _ckpt(optimizer_state_dict: dict | None = None) -> dict:
        d: dict = {
            "model_state_dict": model.state_dict(),
            "label_maps": {
                "count": COUNT_MAP,
                "color": COLOR_MAP,
                "fill": FILL_MAP,
                "shape": SHAPE_MAP,
            },
            "config": {
                "pretrained": pretrained,
                "input_size": [CARD_SIZE[1], CARD_SIZE[0]],
            },
            "epoch": epoch,
            "val_loss": val_loss,
        }
        if optimizer_state_dict is not None:
            d["optimizer_state_dict"] = optimizer_state_dict
        return d

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            images = batch[0].to(dev)
            targets = {h: batch[i + 1].to(dev) for i, h in enumerate(HEADS)}

            outputs = model(images)
            loss = torch.stack([criteria[h](outputs[h], targets[h]) for h in HEADS]).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            n += images.size(0)

        scheduler.step()
        train_loss = epoch_loss / max(n, 1)
        val_loss, val_accs = _evaluate(model, val_loader, dev)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(_ckpt(), best_path)
            improved = " *"

        torch.save(_ckpt(optimizer.state_dict()), last_path)

        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            + "  ".join(f"{h}_acc={val_accs[h]:.3f}" for h in HEADS)
            + f"{improved}"
        )

    print(f"\nsaved best to {best_path}")
    print(f"saved last to {last_path}")


def predict_val(
    data_root: pl.Path,
    weights_path: pl.Path,
    device: str | None = None,
) -> list[dict]:
    """Run inference on the validation set; return per-card predictions."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    model = SetCardClassifier(pretrained=ckpt["config"]["pretrained"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    ds = CardClassDataset(data_root)
    inv_maps = [COUNT_INV, COLOR_INV, FILL_INV, SHAPE_INV]
    results = []
    for idx in range(len(ds)):
        tensor, c_gt, co_gt, f_gt, sh_gt = ds[idx]
        inp = tensor.unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(inp)
        preds = {h: out[h].argmax(dim=1).item() for h in HEADS}
        gts = {"count": c_gt, "color": co_gt, "fill": f_gt, "shape": sh_gt}
        all_correct = all(preds[h] == gts[h] for h in HEADS)
        wrong_heads = {h for h in HEADS if preds[h] != gts[h]}
        results.append(
            {
                "idx": idx,
                "label": ds.samples[idx],
                "gt": gts,
                "pred": preds,
                "correct": all_correct,
                "wrong_heads": wrong_heads,
                "gt_names": {h: inv_maps[i][gts[h]] for i, h in enumerate(HEADS)},
                "pred_names": {h: inv_maps[i][preds[h]] for i, h in enumerate(HEADS)},
            }
        )
    return results


def export_onnx(
    weights_path: pl.Path,
    out_path: pl.Path,
    opset: int = 18,
    dynamic_batch: bool = False,
) -> None:
    """Export a trained classifier to ONNX format."""
    try:
        import onnxscript  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("onnxscript is required for classifier export: pip install onnxscript") from None

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = SetCardClassifier(pretrained=ckpt["config"]["pretrained"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, CARD_SIZE[1], CARD_SIZE[0])
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, **{h: {0: "batch"} for h in HEADS}}

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=list(HEADS),
        dynamic_axes=dynamic_axes,
    )


def quantize_onnx(
    model_path: pl.Path,
    out_path: pl.Path,
    data_root: pl.Path,
    num_samples: int = 100,
) -> None:
    """Quantize an exported ONNX classifier to int8 using calibration data."""
    import os
    import subprocess
    import sys
    import tempfile

    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _CardDataReader(CalibrationDataReader):
        def __init__(self, dataset: CardClassDataset, limit: int) -> None:
            self._dataset = dataset
            self._limit = min(limit, len(dataset))
            self._idx = 0

        def get_next(self) -> dict[str, np.ndarray] | None:  # pyright: ignore[reportIncompatibleMethodOverride]
            if self._idx >= self._limit:
                return None
            tensor = self._dataset[self._idx][0]
            self._idx += 1
            return {"input": np.expand_dims(tensor.numpy(), 0)}

    ds = CardClassDataset(data_root)
    if len(ds) == 0:
        raise RuntimeError(f"no calibration samples found under {data_root}")
    reader = _CardDataReader(ds, num_samples)

    with tempfile.TemporaryDirectory(prefix="quant.") as tmp:
        tmp_path = pl.Path(tmp)
        preprocessed = tmp_path / "preprocessed.onnx"
        quantized = tmp_path / "quantized.onnx"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "onnxruntime.quantization.preprocess",
                "--input",
                str(model_path),
                "--output",
                str(preprocessed),
                "--skip_symbolic_shape",
                "True",
            ],
            check=True,
            cwd=tmp,
        )

        orig_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            quantize_static(
                model_input=str(preprocessed),
                model_output=str(quantized),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
        finally:
            os.chdir(orig_cwd)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        quantized.replace(out_path)
