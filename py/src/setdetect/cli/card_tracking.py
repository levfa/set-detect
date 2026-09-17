import argparse
import pathlib as pl
import queue
import time
import typing as tp

import cv2
import numpy as np

import setdetect.util.frame_grabber as fg
from setdetect.card_detection.detector import CardDetector
from setdetect.card_tracking.detection_tracker import TimedDetectionTracker
from setdetect.card_tracking.sparse_propagator import SparsePropagator
from setdetect.ui.data_args import add_video_root_arg
from setdetect.ui.detection_args import add_detection_args, detection_kwargs_from_args

_CARD_COLOR = (0, 200, 255)  # gold/amber in BGR
_DETECTED_COLOR = (0, 255, 0)  # green in BGR
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _list_videos(root: pl.Path) -> list[pl.Path]:
    """All video files directly under root, sorted by name."""
    return sorted(p for p in pl.Path(root).iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES)


def _limit_resolution(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize ``img`` so its longest side is at most ``max_side``, aspect preserved.

    Returns the (possibly unchanged) image and the uniform factor mapping its
    coordinates back to the original image space.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = longest / max_side
    small = cv2.resize(
        img,
        (int(round(w / scale)), int(round(h / scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, h / small.shape[0]


def _add_seed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed-rows",
        type=int,
        default=8,
        help="Cell grid rows for feature seeding (default: %(default)s)",
    )
    parser.add_argument(
        "--seed-cols",
        type=int,
        default=8,
        help="Cell grid cols for feature seeding (default: %(default)s)",
    )
    parser.add_argument(
        "--seed-max-corners-per-cell",
        type=int,
        default=8,
        help="Max goodFeaturesToTrack corners per cell (default: %(default)s)",
    )
    parser.add_argument(
        "--seed-quality",
        type=float,
        default=0.01,
        help="goodFeaturesToTrack quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--seed-min-dist",
        type=float,
        default=10.0,
        help="goodFeaturesToTrack minimum point distance (default: %(default)s)",
    )


def _seed_kwargs_from_args(args: argparse.Namespace) -> dict[str, tp.Any]:
    return {
        "rows": args.seed_rows,
        "cols": args.seed_cols,
        "max_corners_per_cell": args.seed_max_corners_per_cell,
        "quality_level": args.seed_quality,
        "min_distance": args.seed_min_dist,
    }


def _add_movement_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--move-lk-win-size",
        type=int,
        default=15,
        help="LK window size (applied to both axes) (default: %(default)s)",
    )
    parser.add_argument(
        "--move-lk-max-level",
        type=int,
        default=3,
        help="LK pyramid max level (default: %(default)s)",
    )
    parser.add_argument(
        "--move-fb-thresh",
        type=float,
        default=1.0,
        help="Forward-backward check displacement threshold in px (default: %(default)s)",
    )
    parser.add_argument(
        "--move-rbf-min-points",
        type=int,
        default=6,
        help="Minimum points for movement estimation (default: %(default)s)",
    )
    parser.add_argument(
        "--move-rbf-smoothing",
        type=float,
        default=1.0,
        help="RBF smoothing for the movement interpolation (default: %(default)s)",
    )
    parser.add_argument(
        "--move-rbf-max-points",
        type=int,
        default=100,
        help="Evenly subsample valid flow points down to this many for the RBF fit (default: %(default)s)",
    )


def _est_movement_kwargs_from_args(args: argparse.Namespace) -> dict[str, tp.Any]:
    return {
        "lk_win_size": (args.move_lk_win_size, args.move_lk_win_size),
        "lk_max_level": args.move_lk_max_level,
        "fb_thresh": args.move_fb_thresh,
        "rbf_min_points": args.move_rbf_min_points,
        "rbf_smoothing": args.move_rbf_smoothing,
        "rbf_max_points": args.move_rbf_max_points,
    }


def _add_propagation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--propagation-history-size",
        type=int,
        default=60,
        help="Fixed-size buffer of per-frame snapshots; how many frames back a detection can still be propagated "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--propagation-scale",
        type=float,
        default=0.5,
        help="Downscale factor for both feature seeding and LK; coordinates stay full-res (default: %(default)s)",
    )
    _add_seed_args(parser)
    _add_movement_args(parser)


def _propagation_kwargs_from_args(args: argparse.Namespace) -> dict[str, tp.Any]:
    return {
        "history_size": args.propagation_history_size,
        "scale": args.propagation_scale,
        "seed_kwargs": _seed_kwargs_from_args(args),
        "est_movement_kwargs": _est_movement_kwargs_from_args(args),
    }


def _show_propagation(args: argparse.Namespace) -> None:
    """Step videos frame-by-frame, detecting cards and propagating them."""
    detector = CardDetector(**detection_kwargs_from_args(args))

    videos = _list_videos(args.data_root)
    if not videos:
        raise SystemExit(f"no videos found under {args.data_root}")

    win_name = "Propagation"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for vid_path in videos:
        print(f"\n=== {vid_path.name} ===")
        propagator = SparsePropagator(**_propagation_kwargs_from_args(args))
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print("  could not open, skipping")
            continue

        anchor_frame_id: int | None = None
        anchor_corners: list[np.ndarray] = []

        ok, img_bgr = cap.read()
        if not ok:
            cap.release()
            continue
        while True:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            vis = img_bgr.copy()

            t0 = time.perf_counter()
            step = propagator.next_img(img_gray)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  next_img(dt={dt_ms:.2f} ms, {step.pts_mask.sum()}/{len(step.pts)} active)")

            if anchor_frame_id is not None:
                t0 = time.perf_counter()
                quads = np.concatenate(anchor_corners)
                propagated_quads = propagator.propagate(anchor_frame_id, quads)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"  propagate(dt={dt_ms:.2f} ms, {len(anchor_corners)} cards, "
                    f"frame_id: {anchor_frame_id} -> {step.frame_id})"
                )
                if propagated_quads is not None:
                    polys = [q.astype(np.int32) for q in np.split(propagated_quads, len(anchor_corners))]
                    cv2.polylines(vis, polys, True, _CARD_COLOR, 3)

            cv2.imshow(win_name, vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyWindow(win_name)
                return
            if key == ord("d"):
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                det = detector.detect(img_rgb)
                det_ms = (time.perf_counter() - t0) * 1000.0
                corners = [dc.corners for dc in det.detected_cards]

                anchor_frame_id = step.frame_id
                anchor_corners = corners

                print(f"  d: {len(corners)} cards (det={det_ms:.0f} ms)")
                for quad in corners:
                    cv2.polylines(vis, [quad.astype(np.int32)], True, _DETECTED_COLOR, 3)
                cv2.imshow(win_name, vis)
                cv2.waitKey(0)
            else:
                ok, img_bgr = cap.read()
                if not ok:
                    break
        cap.release()

    cv2.destroyWindow(win_name)


def _show_videos(args: argparse.Namespace) -> None:
    # tracker
    tracker = TimedDetectionTracker(
        detection_kwargs=detection_kwargs_from_args(args), propagation_kwargs=_propagation_kwargs_from_args(args)
    )
    tracker.open()

    # data
    videos = _list_videos(args.data_root)
    if not videos:
        raise SystemExit(f"no videos found under {args.data_root}")

    # frame grabbing
    f_grabber = None
    dispatcher = None
    display_q: queue.Queue = queue.Queue(maxsize=1)
    quit_requested = False

    # gui
    win_name = "Game State — Video"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    thickness = None

    # stats
    import collections

    on_frame_times = collections.defaultdict(list)
    detection_sub_times = collections.defaultdict(list)

    # frame callback
    def on_frame(frame: fg.Frame):
        # preprocessing
        t0 = time.perf_counter()
        img_bgr = frame.data
        small_bgr, to_orig = _limit_resolution(img_bgr, args.det_imgsz)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        on_frame_times["preprocessing"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        detection = tracker.detect(small_rgb)
        on_frame_times["detection"].append(time.perf_counter() - t0)
        for key, val in tracker.times.items():
            detection_sub_times[key].append(val)

        # visualization
        t0 = time.perf_counter()
        if thickness is None:
            return
        vis = img_bgr.copy()
        if detection is not None and detection.matches:
            for dc, _ in detection.matches:
                quad = (dc.corners * to_orig).astype(np.int32)
                cv2.polylines(vis, [quad], True, _CARD_COLOR, thickness)
        try:
            display_q.get_nowait()
        except queue.Empty:
            pass
        try:
            display_q.put_nowait(vis)
        except queue.Full:
            pass
        on_frame_times["visualization"].append(time.perf_counter() - t0)

    try:
        for vid_path in videos:
            if quit_requested:
                break
            dispatcher = fg.AsyncDispatcher(on_frame)
            f_grabber = fg.FrameGrabber(vid_path, dispatcher, 30.0)
            cap = f_grabber.cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            thickness = max(2, min(h, w) // 480)
            f_grabber.start()
            try:
                while f_grabber._thread.is_alive() or not display_q.empty():
                    try:
                        vis = display_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    cv2.imshow(win_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        quit_requested = True
                        break
            finally:
                dispatcher.stop()
                dispatcher.clear()
                f_grabber.stop()
                tracker.clear()
                # stats
                print(vid_path)
                for key, vals in on_frame_times.items():
                    print(f"\t{key}: {np.mean(vals) * 1e3:.2f} +- {np.std(vals) * 1e3:.2f} ms")
                print("\tdetection details:")
                for key, vals in detection_sub_times.items():
                    print(f"\t\t{key}: {np.mean(vals) * 1e3:.2f} +- {np.std(vals) * 1e3:.2f} ms")

                on_frame_times = collections.defaultdict(list)
                detection_sub_times = collections.defaultdict(list)
                while True:
                    try:
                        display_q.get_nowait()
                    except queue.Empty:
                        break
    finally:
        while True:
            try:
                display_q.get_nowait()
            except queue.Empty:
                break
        tracker.close()
        cv2.destroyWindow(win_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-tracking")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (default: 0, -1 for CPU)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp = subparsers.add_parser(
        "show-propagation",
        help="Step videos frame-by-frame; detect cards on 'd' and propagate them with optical flow",
    )
    add_video_root_arg(sp)
    add_detection_args(sp)
    _add_propagation_args(sp)

    vv = subparsers.add_parser(
        "show-videos",
        help="Run the game-state detection pipeline on all videos in a folder and display overlaid cards",
    )
    add_video_root_arg(vv)
    add_detection_args(vv)
    _add_propagation_args(vv)

    args = parser.parse_args(argv)
    if args.command == "show-propagation":
        _show_propagation(args)
    elif args.command == "show-videos":
        _show_videos(args)


if __name__ == "__main__":
    main()
