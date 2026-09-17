import argparse
import pathlib as pl

import cv2
import numpy as np

import setdetect.data.card_cutouts as cc
import setdetect.set_game.game as game
from setdetect.data.card_cutouts import LocationKey


def _add_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--data-root",
        type=pl.Path,
        required=True,
        help="Directory containing the card-cutout dataset (raw/ + masked/).",
    )


def _draw_marked(
    img: np.ndarray,
    entries: list[tuple[LocationKey, list[tuple[int, int]]]],
) -> None:
    for i, ((count, _, shape, fill), corners) in enumerate(entries):
        quad = np.array(corners, np.int32)
        cv2.polylines(img, [quad], True, (0, 0, 255), 3)
        for x, y in corners:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"{i + 1}: {count}-{shape}-{fill}",
            (corners[0][0] + 6, corners[0][1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )


def _show_locations(args: argparse.Namespace, rel_path: pl.Path, win_name: str) -> None:
    structure = cc.Structure()
    locations = cc.load_locations(args.data_root / rel_path)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for col in game.Color:
        entries = [(key, corners) for key, corners in locations.items() if key[1] == col.value]
        if not entries:
            continue
        img_path = args.data_root / structure.raw(col)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        _draw_marked(img, entries)
        cv2.imshow(win_name, img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyWindow(win_name)


def _show_raw(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    win_name = "Raw Data"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for col in game.Color:
        img_path = args.data_root / structure.raw(col)
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
    cv2.destroyWindow(win_name)


def _show_rough_corners(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    _show_locations(args, structure.rough_locations(), "Marked Cards")


def _show_exact_corners(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    _show_locations(args, structure.exact_corners(), "Exact Corners")


def _find_corners(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    rough_path = args.data_root / structure.rough_locations()
    corners_path = args.data_root / structure.exact_corners()
    locations = cc.load_locations(rough_path)
    images: dict[str, np.ndarray | None] = {}
    exact: cc.Locations = {}
    residuals: list[float] = []
    fallbacks = 0
    stray = 0
    for key, rough in locations.items():
        col = key[1]
        if col not in images:
            loaded = cv2.imread(str(args.data_root / structure.raw(game.Color(col))))
            images[col] = np.asarray(loaded) if loaded is not None else None
        img = images[col]
        if img is None:
            print(f"warning: missing image for {col}, skipping")
            continue
        fit = cc.find_exact_corners(img, rough)
        if fit is None:
            print(f"warning: corner fit failed for {key}, keeping rough corners")
            exact[key] = rough
            fallbacks += 1
            continue
        for got, exp in zip(fit.corners, rough):
            if np.linalg.norm(np.asarray(got) - np.asarray(exp)) > 30:
                stray += 1
        exact[key] = fit.corners
        residuals.extend(fit.edge_residuals)
    cc.save_locations(corners_path, exact)
    if residuals:
        r = np.asarray(residuals)
        print(f"edge residual: mean {r.mean():.2f}px, max {r.max():.2f}px")
    print(f"wrote {corners_path} ({len(exact)} cards)")
    if fallbacks:
        print(f"warnings: {fallbacks} fallbacks, {stray} corners >30px from rough")


def _fit_poses(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    corners_path = args.data_root / structure.exact_corners()
    poses_path = args.data_root / structure.poses()
    locations = cc.load_locations(corners_path)
    corners = {key: np.asarray(pts, np.float64) for key, pts in locations.items()}
    fits = cc.fit_card_poses_per_file(corners)
    cc.save_poses(poses_path, fits)
    print(f"wrote {poses_path} ({sum(len(fit.poses) for fit in fits.values())} cards)")
    for color, fit in fits.items():
        worst = max(fit.residuals.items(), key=lambda kv: kv[1])
        print(
            f"{color}: plane residual mean {fit.mean_residual_px:.2f}px, "
            f"max {fit.max_residual_px:.2f}px, worst card {worst[0]} ({worst[1]:.2f}px)"
        )


def _show_plane(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    fits = cc.load_poses(args.data_root / structure.poses())
    if not fits:
        print(f"missing {args.data_root / structure.poses()}; run 'fit-poses' first")
        return
    win_name = "Plane View"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for color, fit in fits.items():
        img_path = args.data_root / structure.raw(game.Color(color))
        if not img_path.is_file():
            continue
        world = np.vstack([pose.world_corners() for pose in fit.poses.values()])
        lo = world.min(axis=0) - 1.0
        hi = world.max(axis=0) + 1.0
        pxs = 20.0
        dsize = (int(round((hi[0] - lo[0]) * pxs)), int(round((hi[1] - lo[1]) * pxs)))
        plane_to_px = np.array(
            [[1.0 / pxs, 0.0, lo[0]], [0.0, 1.0 / pxs, lo[1]], [0.0, 0.0, 1.0]],
            np.float64,
        )
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        warped = cv2.warpPerspective(img, np.linalg.inv(fit.A @ plane_to_px), dsize)
        for i, (key, pose) in enumerate(fit.poses.items()):
            pts = ((pose.world_corners() - lo) * pxs).astype(np.int32)
            cv2.polylines(warped, [pts], True, (0, 0, 255), 2)
            cv2.putText(
                warped,
                f"{i + 1}: {key[0]}-{key[2]}-{key[3]}",
                (pts[0][0] + 4, pts[0][1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        cv2.imshow(win_name, warped)
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyWindow(win_name)


def _extract_cards(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    fits = cc.load_poses(args.data_root / structure.poses())
    if not fits:
        print(f"missing {args.data_root / structure.poses()}; run 'fit-poses' first")
        return
    corners_path = args.data_root / structure.masked_corners()
    out_dir = args.data_root / structure.masked_dir
    total = 0
    for color, fit in fits.items():
        img_path = args.data_root / structure.raw(game.Color(color))
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"missing {img_path}, skipping {color}")
            continue
        world_corners = {key: pose.world_corners() for key, pose in fit.poses.items()}
        images = cc.extract_cards(img, fit.A, world_corners)
        low: list[tuple[LocationKey, float]] = []
        for key, image in images.items():
            path = args.data_root / structure.masked(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
            alpha = image[..., 3]
            h, w = alpha.shape[:2]
            inner = np.zeros(alpha.shape, bool)
            inner[int(h * 0.15) : int(h * 0.85), int(w * 0.15) : int(w * 0.85)] = True
            cov = float((alpha[inner] > 128).mean())
            if cov < 0.95:
                low.append((key, round(cov, 3)))
        total += len(images)
        print(f"{color}: wrote {len(images)} masked cards to {out_dir}")
        if low:
            print("  low coverage:", low)
    print(f"total: {total} masked cards")
    all_keys = [key for fit in fits.values() for key in fit.poses]
    cc.save_locations(corners_path, {key: cc.mask_inset_corners() for key in all_keys})
    print(f"wrote {corners_path} ({len(all_keys)} cards)")


def _show_cutouts(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    fits = cc.load_poses(args.data_root / structure.poses())
    win_name = "Masked Cards"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    for key in [key for fit in fits.values() for key in fit.poses]:
        path = args.data_root / structure.masked(key)
        if not path.is_file():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        background = np.full(img.shape[:2] + (3,), 96, np.uint8)
        comp = (img[..., :3].astype(np.float32) * alpha + background * (1 - alpha)).astype(np.uint8)
        cv2.putText(
            comp,
            f"{key[0]} {key[1]} {key[2]} {key[3]}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.imshow(win_name, comp)
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyWindow(win_name)


def _mark_raw_cards(args: argparse.Namespace) -> None:
    structure = cc.Structure()
    dst_path = args.data_root / structure.rough_locations()
    locations = cc.load_locations(dst_path)

    win_name = "Mark Cards"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for col in game.Color:
        img_path = args.data_root / structure.raw(col)
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cards = [
            game.Card(color=col, shape=shape, fill=fill, count=count)
            for shape in (game.Shape.OVAL, game.Shape.DIAMOND, game.Shape.SQUIGGLE)
            for fill in (game.Fill.OPEN, game.Fill.SOLID, game.Fill.STRIPED)
            for count in (game.Count.ONE, game.Count.TWO, game.Count.THREE)
        ]
        card_i = 0
        while 0 <= card_i < len(cards):
            card = cards[card_i]
            key = card.label
            print(f"Marking card '{key}'")
            points: list[tuple[int, int]] = list(locations.get(key, []))
            clicked = [False]

            def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                    points.append((x, y))
                    clicked[0] = True

            cv2.setMouseCallback(win_name, on_mouse)
            while True:
                display = img.copy()
                for i, (x, y) in enumerate(points):
                    cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        display,
                        str(i + 1),
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                if len(points) == 4:
                    cv2.polylines(
                        display,
                        [np.array(points, np.int32)],
                        True,
                        (0, 255, 0),
                        2,
                    )
                cv2.putText(
                    display,
                    f"{card.label} ({card_i + 1}/{len(cards)})",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(win_name, display)
                key_code = cv2.waitKey(20) & 0xFF
                if len(points) == 4 and clicked[0]:
                    locations[key] = list(points)
                    cc.save_locations(dst_path, locations)
                    card_i += 1
                    break
                if key_code == ord("q"):
                    return
                elif key_code == ord("c"):
                    if points:
                        points.pop()
                elif key_code == ord("p"):
                    card_i -= 1
                    break
                elif key_code == ord("n"):
                    card_i += 1
                    break

    cv2.destroyWindow(win_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="set-card-cutouts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pa = subparsers.add_parser("show-raw", help="Show the raw per-color photos one by one")
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "mark-cards",
        help="Mark each card's rough location with 4 mouse clicks",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "show-rough-corners",
        help="Show the raw photos with rough (click-marked) corners overlaid",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "find-corners",
        help="Refine rough corners to exact corners (intersection of prolonged edges)",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "show-exact-corners",
        help="Show the raw photos with exact (refined) corners overlaid",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "fit-poses",
        help="Fit the shared plane homography and per-card poses from exact corners",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "show-plane",
        help="Show each raw photo warped to its fitted plane, with card rectangles overlaid",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "extract-cards",
        help="Warp every card upright and save masked cutout PNGs",
    )
    _add_data_root_arg(pa)

    pa = subparsers.add_parser(
        "show-cutouts",
        help="Show the extracted card cutouts one by one",
    )
    _add_data_root_arg(pa)

    args = parser.parse_args(argv)
    if args.command == "show-raw":
        _show_raw(args)
    elif args.command == "mark-cards":
        _mark_raw_cards(args)
    elif args.command == "show-rough-corners":
        _show_rough_corners(args)
    elif args.command == "find-corners":
        _find_corners(args)
    elif args.command == "show-exact-corners":
        _show_exact_corners(args)
    elif args.command == "fit-poses":
        _fit_poses(args)
    elif args.command == "show-plane":
        _show_plane(args)
    elif args.command == "extract-cards":
        _extract_cards(args)
    elif args.command == "show-cutouts":
        _show_cutouts(args)


if __name__ == "__main__":
    main()
