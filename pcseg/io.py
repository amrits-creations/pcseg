"""Raw S3DIS reading — torch-free (Chunk 3).

S3DIS stores each room as a folder: a whole-room point file plus an
`Annotations/` subfolder holding one `.txt` per object instance, where the
class label is encoded in the file NAME (e.g. `chair_2.txt` -> "chair").

`parse_room` turns that folder into two plain NumPy arrays: an (N, 6) block of
`x y z r g b` and an (N,) array of integer class labels (0..12). It is the
single source of truth for "read a raw room", reused by both the offline
preprocessing script (Chunk 3) and the interactive viewer (Chunk 1). Kept
torch-free so the viewer doesn't drag in torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pcseg import S3DIS_CLASSES

# class name -> integer index (0..12), the order S3DIS_CLASSES defines.
CLASS_TO_IDX = {name: i for i, name in enumerate(S3DIS_CLASSES)}


def _load_xyzrgb(path: Path) -> np.ndarray:
    """Load an (n, 6) array of x,y,z,r,g,b from one annotation file.

    Fast path is np.loadtxt. A couple of S3DIS files contain a stray non-ASCII
    character that breaks it (a well-known dataset quirk), so we fall back to a
    tolerant line-by-line parse only when that happens.
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


def parse_room(room_dir: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a room's Annotations/ into (xyzrgb [N,6] float32, labels [N] int64).

    Every annotation file's points are tagged with the class its filename
    encodes; an unrecognized name falls back to "clutter" (matches how S3DIS
    is conventionally handled).
    """
    room_dir = Path(room_dir)
    ann_dir = room_dir / "Annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"No Annotations/ folder in {room_dir}. "
            "Check the path points at an extracted S3DIS room."
        )

    xyzrgb_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for ann_file in sorted(ann_dir.glob("*.txt")):
        class_name = ann_file.stem.rsplit("_", 1)[0]  # "chair_2" -> "chair"
        label = CLASS_TO_IDX.get(class_name, CLASS_TO_IDX["clutter"])
        pts = _load_xyzrgb(ann_file)
        if pts.size == 0:
            continue
        xyzrgb_parts.append(pts[:, :6])
        label_parts.append(np.full(len(pts), label, dtype=np.int64))

    if not xyzrgb_parts:
        raise RuntimeError(f"No points parsed from {room_dir}")
    xyzrgb = np.concatenate(xyzrgb_parts).astype(np.float32)
    labels = np.concatenate(label_parts)
    return xyzrgb, labels
