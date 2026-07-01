"""Preprocess raw S3DIS rooms into model-ready .npy files (Chunk 3, offline).

Run ONCE after the dataset is downloaded (see the README "Dataset" section).
For every room it reads the Annotations/ folder, tags each point with its class
integer, and writes one flat float32 array

    data/processed/<Area>/<room>.npy   =   [x y z r g b label]   shape (N, 7)

collapsing each room's messy folder-of-txt into a single fast-loading file.
Then it freezes the train/val/test ROOM split to

    data/processed/splits.json

(Area 5 = test; a few held-out train-area rooms = val) so training reads the
exact same split every time. Re-runs skip rooms already written unless
--overwrite is given.

Examples (from the repo root):
    python scripts/preprocess.py                 # all rooms
    python scripts/preprocess.py --limit 2       # 2 rooms per area (quick check)
    python scripts/preprocess.py --overwrite     # rebuild everything
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make the repo-root `pcseg` package importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcseg.data import make_splits  # noqa: E402
from pcseg.io import parse_room  # noqa: E402

DEFAULT_RAW_ROOT = Path("data/Stanford3dDataset_v1.2_Aligned_Version")
DEFAULT_OUT_ROOT = Path("data/processed")


def find_rooms(raw_root: Path, limit: int | None) -> list[tuple[str, Path]]:
    """List (area_name, room_dir) for every room that has an Annotations/ folder."""
    rooms: list[tuple[str, Path]] = []
    for area_dir in sorted(raw_root.glob("Area_*")):
        if not area_dir.is_dir():
            continue
        room_dirs = [d for d in sorted(area_dir.iterdir()) if (d / "Annotations").is_dir()]
        if limit is not None:
            room_dirs = room_dirs[:limit]
        rooms.extend((area_dir.name, d) for d in room_dirs)
    return rooms


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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

    # Freeze the split over whatever is now in out_root (paths stored relative
    # to out_root so the manifest stays portable across machines/Colab).
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
