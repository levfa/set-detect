import argparse
import pathlib as pl
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

import setdetect.data.img_dir as img_dir
import setdetect.model.card_corners as cn
from setdetect.cli.card_corners import _promoted_weights
from setdetect.data.img_dir import (
    STATUS_GARBAGE,
    STATUS_NOT_SET,
    STATUS_SET,
    STATUS_UNFINISHED,
)
from setdetect.model.card_classification import HEADS
from setdetect.ui.data_args import add_img_root_arg
from setdetect.ui.gui import ARROW_CODES, CORNER_COLORS, ESC_KEY, draw_label

_COLOR_KEYS = {"r": "red", "g": "green", "p": "purple"}
_FILL_KEYS = {"o": "open", "s": "solid", "t": "striped"}
_SHAPE_KEYS = {"d": "diamond", "e": "oval", "q": "squiggle"}
_COUNT_KEYS = {"1": "one", "2": "two", "3": "three"}

_MARKED_STATUSES = (STATUS_NOT_SET, STATUS_GARBAGE)


@dataclass
class _Detected:
    quad: np.ndarray
    vis: np.ndarray
    key: img_dir.RecordKey


def _print_legend() -> None:
    print(
        "\nlabeling keys\n"
        "  count:  1 = one       2 = two      3 = three\n"
        "  color:  r = red        g = green     p = purple\n"
        "  fill:   o = open       s = solid     t = striped\n"
        "  shape:  d = diamond    e = oval      q = squiggle\n"
        "  space   accept current label (or repeat the last one)\n"
        "  n       not a set card\n"
        "  x       garbage\n"
        "  u       back one card    c   clear current\n"
        "  <left>/<right>   previous/next card\n"
        "  <up>/<down>       previous/next image\n"
        "  esc     save and quit\n"
    )


class _Labeler:
    def __init__(
        self,
        images: list[tuple[str, pl.Path]],
        cards: list[list[_Detected]],
        records: img_dir.Labels,
        labels_path: pl.Path,
    ) -> None:
        self.images = images
        self.cards = cards
        self.records = records
        self.labels_path = labels_path
        self.imgs: dict[int, np.ndarray] = {}
        self.img_i = 0
        self.card_i = 0
        self.edit: dict[str, str | None] = {
            "count": None,
            "color": None,
            "fill": None,
            "shape": None,
        }
        self.marked: str | None = None
        self.sticky: tuple[str, str, str, str] | str | None = None
        self._load_card()

    def _load_image(self, img_i: int) -> np.ndarray:
        if img_i not in self.imgs:
            img = cv2.imread(str(self.images[img_i][1]))
            if img is None:
                raise RuntimeError(f"could not read {self.images[img_i][1]}")
            self.imgs[img_i] = img
        return self.imgs[img_i]

    def rel_path(self) -> str:
        return self.images[self.img_i][0]

    def current_card(self) -> _Detected:
        return self.cards[self.img_i][self.card_i]

    def _load_card(self) -> None:
        record = self.records.get(self.current_card().key)
        if record is None:
            self.marked = None
            self.edit = {"count": None, "color": None, "fill": None, "shape": None}
            return
        self.marked = self._mark_of(record.status)
        self.edit = {
            "count": record.count,
            "color": record.color,
            "fill": record.fill,
            "shape": record.shape,
        }

    @staticmethod
    def _is_done(record: img_dir.CardLabel | None) -> bool:
        return record is not None and record.status != STATUS_UNFINISHED

    @staticmethod
    def _mark_of(status: str) -> str | None:
        """The ``self.marked`` value for a stored status (marked iff not "set")."""
        return status if status in _MARKED_STATUSES else None

    def _attrs(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.edit["count"], self.edit["color"], self.edit["fill"], self.edit["shape"])

    def _attrs_complete(self) -> tuple[str, str, str, str]:
        count, color, fill, shape = self._attrs()
        assert count is not None and color is not None and fill is not None and shape is not None
        return count, color, fill, shape

    def _make_record(
        self,
        attrs: tuple[str | None, str | None, str | None, str | None],
        status: str,
    ) -> img_dir.CardLabel:
        card = self.current_card()
        img = self._load_image(self.img_i)
        h, w = img.shape[:2]
        count, color, fill, shape = attrs
        return img_dir.CardLabel(
            path=self.rel_path(),
            w=w,
            h=h,
            quad=[[float(x), float(y)] for x, y in card.quad],
            count=count,
            color=color,
            fill=fill,
            shape=shape,
            status=status,
        )

    def _commit(self, record: img_dir.CardLabel) -> None:
        self.records[self.current_card().key] = record
        img_dir.save_card_classes(self.labels_path, self.records)

    def _persist_edit(self) -> None:
        """Save the current (possibly partial) label so navigation doesn't lose it."""
        if self.marked is not None:
            return
        if not any(self.edit[attr] is not None for attr in HEADS):
            return
        status = STATUS_SET if self.edit_complete() else STATUS_UNFINISHED
        self._commit(self._make_record(self._attrs(), status))

    def first_pending(self) -> bool:
        for img_i, image_cards in enumerate(self.cards):
            for card_i, card in enumerate(image_cards):
                if not self._is_done(self.records.get(card.key)):
                    self.img_i, self.card_i = img_i, card_i
                    self._load_card()
                    return True
        return False

    def edit_complete(self) -> bool:
        return all(self.edit[attr] is not None for attr in HEADS)

    def label_text(self) -> str:
        if self.marked == STATUS_GARBAGE:
            return "garbage"
        if self.marked == STATUS_NOT_SET:
            return "not a set card"
        if self.edit_complete():
            return " ".join(self._attrs_complete())
        return " ".join(self.edit[attr] or "—" for attr in HEADS)

    def on_letter(self, attr: str, value: str) -> None:
        self.edit[attr] = value
        self.marked = None
        if self.edit_complete():
            self.accept(self._attrs_complete())

    def accept_space(self) -> None:
        if self.edit_complete():
            self.accept(self._attrs_complete())
            return
        if self.sticky is None:
            return
        if self.sticky in _MARKED_STATUSES:
            self.accept(self.sticky)
            return
        for attr, value in zip(HEADS, self.sticky):
            if self.edit[attr] is None:
                self.edit[attr] = value
        self.accept(self._attrs_complete())

    def accept(self, classification: str | tuple[str, str, str, str]) -> None:
        if isinstance(classification, str):
            attrs = (None, None, None, None)
            status = classification
        else:
            attrs = classification
            status = STATUS_SET
        self.sticky = classification
        self.marked = self._mark_of(status)
        self._commit(self._make_record(attrs, status))

    def back(self) -> None:
        self._persist_edit()
        if self.card_i > 0:
            self.card_i -= 1
        else:
            img_i = self.img_i - 1
            while img_i >= 0 and not self.cards[img_i]:
                img_i -= 1
            if img_i >= 0:
                self.img_i = img_i
                self.card_i = len(self.cards[img_i]) - 1
        self._load_card()

    def clear(self) -> None:
        self.edit = {"count": None, "color": None, "fill": None, "shape": None}
        self.marked = None
        card = self.current_card()
        if card.key in self.records:
            del self.records[card.key]
            img_dir.save_card_classes(self.labels_path, self.records)

    def card_next(self) -> None:
        self._persist_edit()
        if self.card_i < len(self.cards[self.img_i]) - 1:
            self.card_i += 1
            self._load_card()

    def card_prev(self) -> None:
        self._persist_edit()
        if self.card_i > 0:
            self.card_i -= 1
            self._load_card()

    def image_next(self) -> None:
        self._persist_edit()
        img_i = self.img_i + 1
        while img_i < len(self.cards) and not self.cards[img_i]:
            img_i += 1
        if img_i < len(self.cards):
            self.img_i, self.card_i = img_i, 0
            self._load_card()

    def image_prev(self) -> None:
        self._persist_edit()
        img_i = self.img_i - 1
        while img_i >= 0 and not self.cards[img_i]:
            img_i -= 1
        if img_i >= 0:
            self.img_i, self.card_i = img_i, 0
            self._load_card()

    def render(self, win_name: str) -> None:
        img = self._load_image(self.img_i).copy()
        image_cards = self.cards[self.img_i]
        for card_i, card in enumerate(image_cards):
            quad = card.quad.astype(np.int32)
            if card_i == self.card_i:
                cv2.polylines(img, [quad], True, (0, 255, 0), 4)
                for i, (x, y) in enumerate(quad):
                    color = CORNER_COLORS[i] if card.vis[i] >= 0.5 else (200, 200, 200)
                    cv2.circle(img, (int(x), int(y)), 6, color, -1)
            else:
                cv2.polylines(img, [quad], True, (90, 90, 90), 2)
        draw_label(img, self.label_text(), (10, 25))
        cv2.imshow(win_name, img)

    def print_status(self) -> None:
        line = (
            f"{self.rel_path()}  image {self.img_i + 1}/{len(self.cards)}  "
            f"card {self.card_i + 1}/{len(self.cards[self.img_i])}  "
            f"label: {self.label_text()}"
        )
        print("\r" + line + "\x1b[K", end="", flush=True)


def _label_boards(args: argparse.Namespace) -> None:
    labels_path = img_dir.labels_file(args.data_root)
    records = img_dir.load_card_classes(labels_path)

    weights = (
        pl.Path(args.weights).expanduser()
        if args.weights is not None
        else _promoted_weights(pl.Path(args.corner_runs_root))
    )

    images = [(str(path.relative_to(args.data_root)), path) for path in img_dir.list_images(args.data_root)]
    if not images:
        raise SystemExit(f"no images found under {args.data_root}")

    print(f"detecting cards with {weights} (conf {args.conf}, imgsz {args.imgsz}) ...")
    detections = cn.predict(
        weights,
        [path for _, path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )

    cards: list[list[_Detected]] = []
    for (rel, _), (xy, vis) in zip(images, detections):
        if vis.ndim == 3:
            vis = vis[..., 0]
        order = sorted(
            range(len(xy)),
            key=lambda i: (float(xy[i, :, 1].min()), float(xy[i, :, 0].min())),
        )
        cards.append(
            [
                _Detected(
                    quad=np.asarray(xy[i], np.float32),
                    vis=np.asarray(vis[i], np.float32),
                    key=(rel, img_dir.quad_key(xy[i])),
                )
                for i in order
            ]
        )
    n_detected = sum(len(image_cards) for image_cards in cards)
    print(f"detected {n_detected} cards in {len(images)} images")

    labeler = _Labeler(images, cards, records, labels_path)
    if not labeler.first_pending():
        print(f"all {n_detected} detected cards already labeled in {labels_path}")
        return

    win_name = "Label Cards"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    _print_legend()
    try:
        while True:
            labeler.render(win_name)
            labeler.print_status()
            key = cv2.waitKey(0)
            if key == ESC_KEY:
                break
            if key in ARROW_CODES:
                direction = ARROW_CODES[key]
                if direction == "left":
                    labeler.card_prev()
                elif direction == "right":
                    labeler.card_next()
                elif direction == "up":
                    labeler.image_prev()
                else:
                    labeler.image_next()
                continue
            ch = chr(key) if 0 < key < 256 else None
            if ch in _COLOR_KEYS:
                labeler.on_letter("color", _COLOR_KEYS[ch])
            elif ch in _FILL_KEYS:
                labeler.on_letter("fill", _FILL_KEYS[ch])
            elif ch in _SHAPE_KEYS:
                labeler.on_letter("shape", _SHAPE_KEYS[ch])
            elif ch in _COUNT_KEYS:
                labeler.on_letter("count", _COUNT_KEYS[ch])
            elif ch == " ":
                labeler.accept_space()
            elif ch == "n":
                labeler.accept(STATUS_NOT_SET)
            elif ch == "x":
                labeler.accept(STATUS_GARBAGE)
            elif ch == "u":
                labeler.back()
            elif ch == "c":
                labeler.clear()
    finally:
        cv2.destroyWindow(win_name)

    labeler._persist_edit()
    print()
    img_dir.save_card_classes(labels_path, labeler.records)
    counts = Counter(record.status for record in labeler.records.values())
    print(
        f"saved {labels_path} "
        f"({counts[STATUS_SET]} set cards, {counts[STATUS_NOT_SET]} not a set card, "
        f"{counts[STATUS_UNFINISHED]} unfinished, {counts[STATUS_GARBAGE]} garbage)"
    )


def _show_boards(args: argparse.Namespace) -> None:
    from collections import defaultdict

    from setdetect.model.card_classification import (
        CARD_SIZE,
        HEADS,
        CardClassDataset,
        _warp_card,
        predict_val,
    )

    out_dir: pl.Path | None = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    ds = CardClassDataset(args.data_root)
    n_set = sum(1 for s in ds.samples if s.status == img_dir.STATUS_SET)
    n_other = len(ds.samples) - n_set

    use_inference = not args.no_inference
    predictions = None
    if use_inference:
        if not args.weights.exists():
            raise SystemExit(f"no checkpoint at {args.weights}")
        print(f"loading {args.weights} ...")
        predictions = predict_val(args.data_root, args.weights)
        n_correct = sum(1 for r in predictions if r["correct"])
        n_wrong = len(predictions) - n_correct
        pct = n_correct / max(len(predictions), 1) * 100
        print(f"{len(predictions)} cards: {n_correct} correct, {n_wrong} wrong ({pct:.1f}%)")
        per_head = {h: sum(1 for r in predictions if r["pred"][h] == r["gt"][h]) for h in HEADS}
        print("  " + "  ".join(f"{h}: {per_head[h]}/{len(predictions)}" for h in HEADS))
        print()

    # build visible indices (filter out errors-only if requested)
    indices = list(range(len(ds)))
    if use_inference and args.errors_only:
        assert predictions is not None
        indices = [i for i in indices if not predictions[i]["correct"]]

    if not indices:
        print("no cards to show")
        return

    # group cards by source image
    by_path: dict[str, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        by_path[ds.samples[i].path].append(i)

    img_cache: dict[str, np.ndarray] = {}

    def _load_board(path: str) -> np.ndarray:
        if path not in img_cache:
            full = args.data_root / path
            img = cv2.imread(str(full))
            if img is None:
                raise RuntimeError(f"could not read {full}")
            img_cache[path] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_cache[path].copy()

    def _quad_str(quad: list[list[float]]) -> str:
        return "[" + ",".join(f"[{int(round(x))},{int(round(y))}]" for x, y in quad) + "]"

    def _print_line(idx: int) -> None:
        label = ds.samples[idx]
        is_set = label.status == img_dir.STATUS_SET
        tag = "set" if is_set else "not-set"
        qs = _quad_str(label.quad)
        if use_inference:
            assert predictions is not None
            r = predictions[idx]
            gt_str = "/".join(r["gt_names"][h] for h in HEADS)
            if r["correct"]:
                print(f"{idx:03d}  {tag}  {label.path}  {qs}  gt: {gt_str}  ok")
            else:
                pred_str = "/".join(r["pred_names"][h] for h in HEADS)
                wrong = " ".join(r["wrong_heads"])
                print(f"{idx:03d}  {tag}  {label.path}  {qs}  gt: {gt_str}  pred: {pred_str}  WRONG {wrong}")
        else:
            if is_set:
                gt_str = "/".join(x or "other" for x in [label.count, label.color, label.fill, label.shape])
            else:
                gt_str = "other/other/other/other"
            print(f"{idx:03d}  {tag}  {label.path}  {qs}  gt: {gt_str}")

    def _board_vis(idx: int) -> np.ndarray:
        label = ds.samples[idx]
        board = _load_board(label.path)
        for other_i in by_path[label.path]:
            other_quad = np.array(ds.samples[other_i].quad, dtype=np.int32)
            if other_i == idx:
                cv2.polylines(board, [other_quad], True, (0, 255, 0), 3)
            else:
                cv2.polylines(board, [other_quad], True, (90, 90, 90), 2)
        return board

    # show cards
    win_name = "show-boards"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cur = 0
    saved = 0
    try:
        while True:
            idx = indices[cur]
            _print_line(idx)

            if out_dir is not None:
                label = ds.samples[idx]
                is_set = label.status == img_dir.STATUS_SET
                tag = "set" if is_set else "not-set"
                quad = np.array(label.quad, dtype=np.float32)
                img_rgb = _load_board(label.path)
                card = _warp_card(img_rgb, quad, CARD_SIZE)
                if use_inference:
                    assert predictions is not None
                    r = predictions[idx]
                    if r["correct"]:
                        fname = f"{idx:03d}_{tag}_ok.png"
                    else:
                        diff = "_".join(f"{h[0]}:{r['gt_names'][h]}->{r['pred_names'][h]}" for h in r["wrong_heads"])
                        fname = f"{idx:03d}_{tag}_{diff}.png"
                else:
                    if is_set:
                        parts = [
                            label.count or "other",
                            label.color or "other",
                            label.fill or "other",
                            label.shape or "other",
                        ]
                    else:
                        parts = ["other"] * 4
                    fname = f"{idx:03d}_{tag}_{'_'.join(parts)}.png"
                cv2.imwrite(str(out_dir / fname), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
                saved += 1

            vis = _board_vis(idx)
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(0)
            if key == ESC_KEY:
                break
            at_end = cur >= len(indices) - 1
            if key in ARROW_CODES and ARROW_CODES[key] == "left":
                cur = max(cur - 1, 0)
            elif at_end:
                break
            else:
                cur += 1
    finally:
        cv2.destroyWindow(win_name)

    print()
    if out_dir is not None:
        print(f"saved {saved} cards to {out_dir}")
    if not use_inference:
        print(f"{n_set} set,  {n_other} not-set")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-class-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pa = subparsers.add_parser(
        "label-boards",
        help="Interactively label detected cards in real board photos into labels/card-classes.jsonl",
    )
    add_img_root_arg(pa)
    pa.add_argument(
        "--weights",
        type=pl.Path,
        default=None,
        help="Corner-keypoint model weights (default: promoted 'current' run's best.pt under --corner-runs-root)",
    )
    pa.add_argument(
        "--corner-runs-root",
        type=pl.Path,
        required=True,
        help="Directory holding the corner model's versioned training runs (used to find "
        "the promoted 'current' weights when --weights is omitted)",
    )
    pa.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: %(default)s)",
    )
    pa.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: %(default)s)",
    )
    pa.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference, e.g. 0 or cpu (default: auto)",
    )

    sv = subparsers.add_parser(
        "show-boards",
        help="Run inference on the validation split; print details to console",
    )
    add_img_root_arg(sv)
    sv.add_argument(
        "--out-dir",
        type=pl.Path,
        default=None,
        help="Save warped cards to this directory (default: display only, no saving)",
    )
    sv.add_argument(
        "--weights",
        type=pl.Path,
        required=True,
        help="Checkpoint to load for inference",
    )
    sv.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip inference, print labels without prediction comparison",
    )
    sv.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show cards where the model predicted incorrectly",
    )

    args = parser.parse_args(argv)
    if args.command == "label-boards":
        _label_boards(args)
    elif args.command == "show-boards":
        _show_boards(args)


if __name__ == "__main__":
    main()
