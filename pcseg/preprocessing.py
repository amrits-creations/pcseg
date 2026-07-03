from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from pcseg import S3DIS_CLASSES

CLASS_TO_IDX = {name: i for i, name in enumerate(S3DIS_CLASSES)}


def _load_xyzrgb(path: Path) -> np.ndarray:
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
                    continue
        return np.asarray(rows, dtype=np.float64)


def parse_room(room_dir: Path | str) -> tuple[np.ndarray, np.ndarray]:
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
        class_name = ann_file.stem.rsplit("_", 1)[0]
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


def find_rooms(raw_root: Path, limit: int | None) -> list[tuple[str, Path]]:
    rooms: list[tuple[str, Path]] = []
    for area_dir in sorted(raw_root.glob("Area_*")):
        if not area_dir.is_dir():
            continue
        room_dirs = [d for d in sorted(area_dir.iterdir()) if (d / "Annotations").is_dir()]
        if limit is not None:
            room_dirs = room_dirs[:limit]
        rooms.extend((area_dir.name, d) for d in room_dirs)
    return rooms


def make_splits(
    processed_root: Path | str,
    val_per_area: int = 2,
    seed: int = 42,
) -> dict[str, list[Path]]:
    processed_root = Path(processed_root)
    rng = random.Random(seed)
    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    for area_dir in sorted(processed_root.glob("Area_*")):
        rooms = sorted(area_dir.glob("*.npy"))
        if not rooms:
            continue
        if area_dir.name == "Area_5":
            test.extend(rooms)
            continue
        held = set(rng.sample(rooms, min(val_per_area, len(rooms))))
        for room in rooms:
            (val if room in held else train).append(room)

    return {"train": train, "val": val, "test": test}


DEFAULT_RAW_ROOT = Path("data/Stanford3dDataset_v1.2_Aligned_Version")
DEFAULT_OUT_ROOT = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw S3DIS rooms into model-ready .npy files and create train/val/test split."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most this many rooms PER AREA (quick test)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-process rooms even if the .npy already exists")
    parser.add_argument("--val-per-area", type=int, default=2,
                        help="rooms held out per training area for validation")
    args = parser.parse_args()

    if not args.raw_root.is_dir():
        sys.exit(f"Raw dataset not found at {args.raw_root}. See the README to download it.")

    rooms = find_rooms(args.raw_root, args.limit)
    if not rooms:
        sys.exit(f"No rooms with an Annotations/ folder under {args.raw_root}.")

    print(f"Preprocessing {len(rooms)} rooms -> {args.out_root}")
    for i, (area, room_dir) in enumerate(rooms, 1):
        out_path = args.out_root / area / f"{room_dir.name}.npy"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i}/{len(rooms)}] skip (exists) {area}/{room_dir.name}")
            continue
        t0 = time.time()
        xyzrgb, labels = parse_room(room_dir)
        room = np.concatenate([xyzrgb, labels[:, None].astype(np.float32)], axis=1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, room)
        print(f"  [{i}/{len(rooms)}] {area}/{room_dir.name:<22} "
              f"{len(room):>9,} pts  ({time.time() - t0:4.1f}s)")

    splits = make_splits(args.out_root, val_per_area=args.val_per_area)
    manifest = {
        split: [str(p.relative_to(args.out_root)) for p in paths]
        for split, paths in splits.items()
    }
    splits_path = args.out_root / "splits.json"
    splits_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote split manifest -> {splits_path}")
    print(f"  train: {len(manifest['train'])} rooms | "
          f"val: {len(manifest['val'])} rooms | test: {len(manifest['test'])} rooms")


if __name__ == "__main__":
    main()
