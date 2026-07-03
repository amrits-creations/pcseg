from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcseg import S3DIS_CLASSES, S3DIS_COLORS
from pcseg.preprocessing import parse_room

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

    palette = np.asarray(S3DIS_COLORS, dtype=np.float64) / 255.0
    colors = palette[labels]

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
