"""Command line interface: ``pcd-viewer [FOLDER | FILE] [options]``."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pcd_viewer.cloud import field_stats, list_pcd_files
from pcd_viewer.pcd import read_pcd


def _print_file_info(path: Path) -> None:
    header, data = read_pcd(path)
    print(f"File:      {path}")
    print(f"Version:   {header.version}")
    print(f"Data:      {header.data}")
    print(f"Points:    {header.points} (width={header.width}, height={header.height})")
    print(f"Viewpoint: {' '.join(f'{v:g}' for v in header.viewpoint)}")
    print("Fields:")
    for s in field_stats(header, data):
        finite = not math.isnan(s.min)
        rng = f"min={s.min:.4g}  max={s.max:.4g}" if finite else "no finite values"
        print(f"  {s.name:<14} {s.type:<8} {rng}  unique={s.unique_text}")


def _print_folder_info(folder: Path) -> None:
    files = list_pcd_files(folder)
    print(f"Folder: {folder}  ({len(files)} .pcd files)")
    for path in files:
        try:
            header, _ = read_pcd(path)
        except Exception as exc:
            print(f"  {path.name}  ERROR: {exc}")
            continue
        print(f"  {path.name}  {header.points:>9} pts  {header.data:<17} {' '.join(header.fields)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browse and view PCD point cloud files.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="folder with .pcd files, or a single .pcd file (opens its folder with it selected)",
    )
    parser.add_argument(
        "-c",
        "--color-by",
        help="initial field used for coloring (default: reflectivity, intensity or z); "
        "'none' for white",
    )
    parser.add_argument("-s", "--point-size", type=float, default=3.0, help="point size in pixels")
    parser.add_argument("--cmap", default="viridis", help="matplotlib colormap name")
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="use the full value range for colors (default: 1st-99th percentile)",
    )
    parser.add_argument(
        "-i", "--info", action="store_true", help="print header/field info and exit (no GUI)"
    )
    args = parser.parse_args(argv)

    if args.path is not None and not args.path.exists():
        parser.error(f"No such file or directory: {args.path}")

    if args.info:
        if args.path is None:
            parser.error("--info needs a file or folder")
        if args.path.is_dir():
            _print_folder_info(args.path)
        else:
            _print_file_info(args.path)
        return

    color_by = args.color_by
    if color_by is not None and color_by.lower() == "none":
        color_by = ""  # solid color

    from pcd_viewer.app import run  # imported lazily: Qt/VTK are slow to load

    raise SystemExit(
        run(
            args.path,
            color_by=color_by,
            point_size=args.point_size,
            cmap=args.cmap,
            full_range=args.full_range,
        )
    )


if __name__ == "__main__":
    main()
