"""Visualize one S3DIS room colored by its ground-truth labels (Chunk 1).

S3DIS stores each room as a folder: a whole-room point file plus an
`Annotations/` subfolder holding one `.txt` per object instance. The class
label is encoded in each annotation file's NAME (e.g. `chair_2.txt` -> "chair").
We read every annotation file, tag its points with that class, then open an
interactive Open3D window painting each point by its class color.

Run (from the repo root):
    python scripts/visualize_room.py --area Area_1 --room conferenceRoom_1

Drag to rotate, scroll to zoom, close the window to exit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Open3D's bundled GLFW can't create a GL window natively on Wayland (it fails
# with a GLEW init error). When we're on a Wayland session that also has an X
# server available (XWayland, i.e. DISPLAY is set), drop the Wayland hints so
# GLFW falls back to X11 via XWayland — which renders fine. Must happen before
# Open3D (and its GLFW) is imported. No-op on plain X11 or headless setups.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np
import open3d as o3d

# Make the repo-root `pcseg` package importable when this file is run directly
# (`python scripts/visualize_room.py`), since Python otherwise only puts the
# scripts/ folder on the path. Goes away once we package the project properly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcseg import S3DIS_CLASSES, S3DIS_COLORS  # noqa: E402
from pcseg.io import parse_room  # noqa: E402  (shared raw-room reader, Chunk 3)

DEFAULT_DATA_ROOT = Path("data/Stanford3dDataset_v1.2_Aligned_Version")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--area", default="Area_1")
    parser.add_argument("--room", default="conferenceRoom_1")
    args = parser.parse_args()

    room_dir = args.data_root / args.area / args.room
    print(f"Loading {room_dir} ...")
    xyzrgb, labels = parse_room(room_dir)
    xyz = xyzrgb[:, :3]

    # Paint each point by its class color (Open3D wants RGB floats in [0, 1]).
    palette = np.asarray(S3DIS_COLORS, dtype=np.float64) / 255.0
    colors = palette[labels]

    # Legend + per-class counts, so you can name what each color is on screen.
    present = np.unique(labels)
    print(f"\n{len(xyz):,} points across {len(present)} classes:")
    for idx in present:
        rgb = tuple(int(c) for c in S3DIS_COLORS[idx])
        n = int(np.sum(labels == idx))
        print(f"  {S3DIS_CLASSES[idx]:<10} RGB{str(rgb):<16} {n:>10,} pts")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("\nOpening viewer — drag to rotate, scroll to zoom, close window to exit.")
    o3d.visualization.draw_geometries([pcd], window_name=f"{args.area}/{args.room}")


if __name__ == "__main__":
    main()
