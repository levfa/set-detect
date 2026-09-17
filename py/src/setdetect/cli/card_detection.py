import argparse

import cv2
import numpy as np

from setdetect.card_detection.detector import (
    _CANON,
    _CARD_AR,
    CORNER_VISIBLE_THRESHOLD,
    CardDetector,
    DetectedCard,
    _fit_arrangement,
    _occlusion_order,
    cards_arrangement,
)
from setdetect.ui.data_args import add_img_root_arg
from setdetect.ui.detection_args import (
    add_detection_args,
    detection_kwargs_from_args,
)
from setdetect.ui.gui import (
    ARROW_CODES,
    COLOR_LETTERS,
    CORNER_COLORS,
    COUNT_LETTERS,
    ESC_KEY,
    FILL_LETTERS,
    SHAPE_LETTERS,
    draw_label,
)


def _show_boards(args: argparse.Namespace) -> None:
    card_detector = CardDetector(**detection_kwargs_from_args(args))

    import setdetect.data.img_dir as img_dir

    images = [(str(path.relative_to(args.data_root)), path) for path in img_dir.list_images(args.data_root)]
    if not images:
        raise SystemExit(f"no images found under {args.data_root}")

    print(f"running pipeline on {len(images)} images ...")

    results = []
    for rel, path in images:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"  skipping {path} (could not read)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det = card_detector.detect(img_rgb)
        results.append((rel, img_bgr, det))

    print(f"processed {len(results)} images")
    for rel, _, pred in results:
        print(f"  {rel}: {len(pred.detected_cards)} cards detected")
    print()

    win_name = "Game State — Images"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cur = 0
    try:
        while cur < len(results):
            _, img_bgr, pred = results[cur]
            vis = img_bgr.copy()
            for dc in pred.detected_cards:
                quad = dc.corners.astype(np.int32)
                cv2.polylines(vis, [quad], True, (0, 255, 0), 3)
                for ci, (x, y) in enumerate(quad):
                    corner_color = (
                        CORNER_COLORS[ci] if dc.corner_visibility[ci] >= CORNER_VISIBLE_THRESHOLD else (200, 200, 200)
                    )
                    cv2.circle(vis, (int(x), int(y)), 6, corner_color, -1)

                count_i = int(dc.count_probs.argmax())
                color_i = int(dc.color_probs.argmax())
                shape_i = int(dc.shape_probs.argmax())
                fill_i = int(dc.fill_probs.argmax())

                label = (
                    f"{COUNT_LETTERS[count_i]}{COLOR_LETTERS[color_i]}{SHAPE_LETTERS[shape_i]}{FILL_LETTERS[fill_i]}"
                )
                cx = int(quad[:, 0].mean())
                cy = int(quad[:, 1].mean()) - 10
                draw_label(vis, label, (cx - 60, cy))

            cv2.imshow(win_name, vis)
            key = cv2.waitKey(0)
            ch = chr(key) if 0 < key < 256 else None
            if key == ESC_KEY or ch == "q":
                break
            if key in ARROW_CODES:
                direction = ARROW_CODES[key]
                if direction in ("left", "up"):
                    cur = max(cur - 1, 0)
                else:
                    cur = min(cur + 1, len(results) - 1)
            else:
                cur = min(cur + 1, len(results) - 1)
    finally:
        cv2.destroyWindow(win_name)


def _show_arrangement(args: argparse.Namespace) -> None:
    card_detector = CardDetector(**detection_kwargs_from_args(args))

    import setdetect.data.img_dir as img_dir

    images = [(str(path.relative_to(args.data_root)), path) for path in img_dir.list_images(args.data_root)]
    if not images:
        raise SystemExit(f"no images found under {args.data_root}")

    print(f"running pipeline on {len(images)} images ...")

    results = []
    for rel, path in images:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"  skipping {path} (could not read)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det = card_detector.detect(img_rgb)
        results.append((rel, img_bgr, det))

    print(f"processed {len(results)} images")
    for rel, _, det in results:
        print(f"  {rel}: {len(det.detected_cards)} cards detected")
    print()

    orig_win = "Original"
    arr_win = "Arrangement"
    cv2.namedWindow(orig_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(arr_win, cv2.WINDOW_NORMAL)
    arr_sz = 1024
    cur = 0
    try:
        while cur < len(results):
            _, img_bgr, det = results[cur]

            left = img_bgr.copy()
            for dc in det.detected_cards:
                quad = dc.corners.astype(np.int32)
                cv2.polylines(left, [quad], True, (0, 255, 0), 3)
                for ci, (x, y) in enumerate(quad):
                    corner_color = (
                        CORNER_COLORS[ci] if dc.corner_visibility[ci] >= CORNER_VISIBLE_THRESHOLD else (200, 200, 200)
                    )
                    cv2.circle(left, (int(x), int(y)), 6, corner_color, -1)

                count_i = int(dc.count_probs.argmax())
                color_i = int(dc.color_probs.argmax())
                shape_i = int(dc.shape_probs.argmax())
                fill_i = int(dc.fill_probs.argmax())
                label = (
                    f"{COUNT_LETTERS[count_i]}{COLOR_LETTERS[color_i]}{SHAPE_LETTERS[shape_i]}{FILL_LETTERS[fill_i]}"
                )
                cx = int(quad[:, 0].mean())
                cy = int(quad[:, 1].mean()) - 10
                draw_label(left, label, (cx - 60, cy))

            right = np.full((arr_sz, arr_sz, 3), 255, dtype=np.uint8)
            arr = det.matches_arrangement
            draw_ids = sorted(range(len(arr.card_poses)), key=lambda i: arr.card_z_orders[i])
            for i in draw_ids:
                rcx, rcy, angle = arr.card_poses[i]
                px_x = int(round(rcx * arr_sz))
                px_y = int(round(rcy * arr_sz))
                size = (arr.card_width * arr_sz, arr.card_height * arr_sz)
                angle_deg = float(np.rad2deg(angle))
                box = cv2.boxPoints(((rcx * arr_sz, rcy * arr_sz), size, angle_deg))
                box = np.round(box).astype(np.int32)
                cv2.fillPoly(right, [box], (120, 200, 120))
                cv2.polylines(right, [box], True, (0, 0, 0), 2)
                cv2.circle(right, (px_x, px_y), 5, (0, 0, 255), -1)

                if i < len(det.matches):
                    dc = det.matches[i][0]
                    count_i = int(dc.count_probs.argmax())
                    color_i = int(dc.color_probs.argmax())
                    shape_i = int(dc.shape_probs.argmax())
                    fill_i = int(dc.fill_probs.argmax())
                    label = (
                        f"{COUNT_LETTERS[count_i]}{COLOR_LETTERS[color_i]}"
                        f"{SHAPE_LETTERS[shape_i]}{FILL_LETTERS[fill_i]}"
                    )
                    draw_label(right, label, (px_x - 30, px_y - 10))

            cv2.imshow(orig_win, left)
            cv2.imshow(arr_win, right)
            key = cv2.waitKey(0)
            ch = chr(key) if 0 < key < 256 else None
            if key == ESC_KEY or ch == "q":
                break
            if key in ARROW_CODES:
                direction = ARROW_CODES[key]
                if direction in ("left", "up"):
                    cur = max(cur - 1, 0)
                else:
                    cur = min(cur + 1, len(results) - 1)
            else:
                cur = min(cur + 1, len(results) - 1)
    finally:
        cv2.destroyWindow(orig_win)
        cv2.destroyWindow(arr_win)


def _dummy_arrangement_cards(num_cards: int, seed: int) -> list[DetectedCard]:
    """Deterministic, slightly-overlapping card quads for arrangement profiling.

    Cards sit on a grid with slight rotation/jitter, a mild projective warp, and
    per-corner noise; corners falling inside a neighbor's quad are marked hidden
    so the z-order path is exercised.
    """
    rng = np.random.default_rng(seed)
    cols = max(1, int(np.ceil(np.sqrt(num_cards))))
    rows = max(1, int(np.ceil(num_cards / cols)))
    card_w = 1.2
    card_h = card_w / _CARD_AR
    base = (_CANON - np.array([_CARD_AR / 2, 0.5])) * np.array([card_w, card_h])
    cell_w = (base[:, 0].max() - base[:, 0].min()) - 0.22
    cell_h = (base[:, 1].max() - base[:, 1].min()) - 0.1
    total_w = cols * cell_w
    total_h = rows * cell_h

    quads: list[np.ndarray] = []
    for i in range(num_cards):
        row_i, col_i = divmod(i, cols)
        cx = (col_i + 0.5) * cell_w - total_w / 2 + rng.uniform(-0.05, 0.05)
        cy = (row_i + 0.5) * cell_h - total_h / 2 + rng.uniform(-0.05, 0.05)
        ang = rng.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        quads.append(base @ rot.T + np.array([cx, cy]))

    hom = np.array([[260.0, -18.0, 420.0], [12.0, 265.0, 310.0], [0.00012, -0.00011, 1.0]])

    def project(pts: np.ndarray) -> np.ndarray:
        w = np.hstack([pts, np.ones((len(pts), 1))]) @ hom.T
        return w[:, :2] / w[:, 2:]

    vis = [np.ones(4, np.float32) for _ in range(num_cards)]
    polys = [q.astype(np.float32) for q in quads]
    for i, quad_i in enumerate(quads):
        for k, (x, y) in enumerate(quad_i):
            for j, poly in enumerate(polys):
                if j == i:
                    continue
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                    vis[i][k] = 0.0
                    break

    return [
        DetectedCard(
            corners=(project(q) + rng.normal(0.0, 1.5, q.shape)).astype(np.float32),
            corner_visibility=v,
            count_probs=np.zeros(4),
            color_probs=np.zeros(4),
            shape_probs=np.zeros(4),
            fill_probs=np.zeros(4),
        )
        for q, v in zip(quads, vis)
    ]


def _profile_arrangement_estimation(args: argparse.Namespace) -> None:
    """Time ``cards_arrangement`` over repeated runs on a fixed dummy board."""
    import time

    board = _dummy_arrangement_cards(args.cards, args.seed)
    corners = np.array([dc.corners.astype(np.float64) for dc in board])
    n = len(board)
    cards_arrangement(board)  # warmup: one-time numpy/least-squares init

    totals = np.empty(args.runs)
    for k in range(args.runs):
        t0 = time.perf_counter()
        cards_arrangement(board)
        totals[k] = time.perf_counter() - t0

    fits = np.empty(args.runs)
    zorders = np.empty(args.runs)
    for k in range(args.runs):
        t0 = time.perf_counter()
        if n:
            _fit_arrangement(corners)
            t1 = time.perf_counter()
            _occlusion_order(board)
            t2 = time.perf_counter()
            fits[k] = t1 - t0
            zorders[k] = t2 - t1
        else:
            fits[k] = 0.0
            zorders[k] = 0.0

    rest = totals - fits - zorders
    rows_out = [("total arrangement", totals), ("  fit", fits), ("  z-order", zorders), ("  rest", rest)]
    print(f"cards: {n}  runs: {args.runs}  seed: {args.seed}")
    for name, arr in rows_out:
        mean = float(np.mean(arr)) * 1e3
        std = float(np.std(arr)) * 1e3
        lo = float(np.min(arr)) * 1e3
        hi = float(np.max(arr)) * 1e3
        print(f"  {name}: mean {mean:.3f} ms  std {std:.3f} ms  (min {lo:.3f}, max {hi:.3f})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sv = subparsers.add_parser(
        "show-boards",
        help="Run detection + classification on real board photos and show the result",
    )
    add_img_root_arg(sv)
    add_detection_args(sv)

    sa = subparsers.add_parser(
        "show-arrangement",
        help="Show detected cards alongside their abstracted arrangement",
    )
    add_img_root_arg(sa)
    add_detection_args(sa)

    pp = subparsers.add_parser(
        "profile-arrangement-estimation",
        help="Profile the arrangement estimation on a dummy board (mean/std over runs)",
    )
    pp.add_argument("--cards", type=int, default=12, help="number of dummy cards (default: 12)")
    pp.add_argument("--runs", type=int, default=100, help="number of timing runs (default: 100)")
    pp.add_argument("--seed", type=int, default=0, help="dummy board seed (default: 0)")

    args = parser.parse_args(argv)
    if args.command == "show-boards":
        _show_boards(args)
    elif args.command == "show-arrangement":
        _show_arrangement(args)
    elif args.command == "profile-arrangement-estimation":
        _profile_arrangement_estimation(args)


if __name__ == "__main__":
    main()
