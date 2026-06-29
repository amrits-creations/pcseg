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

# class name -> integer index (0..12)
CLASS_TO_IDX = {name: i for i, name in enumerate(S3DIS_CLASSES)}

DEFAULT_DATA_ROOT = Path("data/Stanford3dDataset_v1.2_Aligned_Version")


def load_xyzrgb(path: Path) -> np.ndarray:
    """Load an (N, 6) array of x,y,z,r,g,b from one annotation file.

    Fast path is np.loadtxt. A couple of S3DIS files contain a stray
    non-ASCII character that breaks it (a well-known dataset quirk), so we
    fall back to a tolerant line-by-line parse only when that happens.
    """
    try:
        return np.loadtxt(path)
    except ValueError:
        rows = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    continue
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue  # drop the rare malformed line
        return np.asarray(rows, dtype=np.float64)


def load_room(room_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a room's Annotations/ into (xyz [N,3], labels [N])."""
    ann_dir = room_dir / "Annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"No Annotations/ folder in {room_dir}. "
            "Check --data-root/--area/--room point at an extracted S3DIS room."
        )

    all_xyz: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for ann_file in sorted(ann_dir.glob("*.txt")):
        class_name = ann_file.stem.rsplit("_", 1)[0]  # "chair_2" -> "chair"
        label = CLASS_TO_IDX.get(class_name, CLASS_TO_IDX["clutter"])
        pts = load_xyzrgb(ann_file)
        if pts.size == 0:
            continue
        all_xyz.append(pts[:, :3])
        all_labels.append(np.full(len(pts), label, dtype=np.int64))

    if not all_xyz:
        raise RuntimeError(f"No points parsed from {room_dir}")
    return np.concatenate(all_xyz), np.concatenate(all_labels)


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
    xyz, labels = load_room(room_dir)

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
