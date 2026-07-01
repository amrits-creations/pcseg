"""S3DIS data pipeline for training (Chunk 3): rooms -> model-ready blocks.

Two things live here:

  * make_splits()  -- the canonical train / val / test ROOM split.
        test  = every room in Area 5 (a whole held-out building; the split the
                S3DIS papers report on, so our number is comparable).
        val   = a small, deterministic, held-out slice of the TRAIN areas, used
                for early-stopping / picking the best checkpoint (keeps Area 5
                pristine — we never tune against the test set).
        train = the remaining train-area rooms (Areas 1,2,3,4,6).

  * S3DISDataset  -- a PyTorch Dataset. One item = ONE ready-to-train block:
        a (num_points, 9) float tensor + a (num_points,) label tensor. Each
        __getitem__ cuts a random 1m x 1m vertical column from a room, samples
        it to a fixed point count, and builds the standard 9-dim point feature
        (block-local xyz | rgb/255 | room-normalized xyz). Blocks are cut LIVE,
        so every epoch sees fresh random blocks (free augmentation).

The 9-dim feature follows the well-known PointNet++ S3DIS recipe so it stays
interview-defensible. See BUILD_PROCESS.md, Chunk 3, for the why.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# --- canonical S3DIS hyperparameters (the PointNet++ recipe) ---------------
NUM_POINTS = 4096        # points sampled per block (fixed so blocks batch cleanly)
BLOCK_SIZE = 1.0         # block footprint in metres (1m x 1m, full room height)
MIN_BLOCK_POINTS = 1024  # if a random block is sparser than this, try another spot
TEST_AREA = "Area_5"     # held-out building -> the paper-standard test set


def make_splits(
    processed_root: Path | str,
    val_per_area: int = 2,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Map the preprocessed room .npy files into train/val/test path lists.

    `val_per_area` rooms are held out from EACH training area for validation
    (deterministic given `seed` + the sorted room order, so the split is
    reproducible and can be frozen to splits.json). Area 5 is test in full.
    """
    processed_root = Path(processed_root)
    rng = random.Random(seed)
    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    for area_dir in sorted(processed_root.glob("Area_*")):
        rooms = sorted(area_dir.glob("*.npy"))
        if not rooms:
            continue
        if area_dir.name == TEST_AREA:
            test.extend(rooms)
            continue
        held = set(rng.sample(rooms, min(val_per_area, len(rooms))))
        for room in rooms:
            (val if room in held else train).append(room)

    return {"train": train, "val": val, "test": test}


def _npy_nrows(path: Path) -> int:
    """Point count of a room .npy without loading it (reads the header only)."""
    return int(np.load(path, mmap_mode="r").shape[0])


class S3DISDataset(Dataset):
    """Feeds (num_points, 9) blocks sampled live from preprocessed rooms.

    Args:
        room_paths: list of `<room>.npy` files (each an (N, 7) xyzrgb+label
            array, as written by scripts/preprocess.py).
        num_points: points per block (fixed tensor size).
        block_size: block footprint in metres.
        training: if True, blocks are sampled with fresh randomness each call;
            if False, sampling is seeded by the item index so a val/test pass
            is reproducible.
        samples_per_epoch: how many blocks make one epoch. Defaults to roughly
            one pass over every point (total_points // num_points).
    """

    def __init__(
        self,
        room_paths,
        num_points: int = NUM_POINTS,
        block_size: float = BLOCK_SIZE,
        training: bool = True,
        samples_per_epoch: int | None = None,
    ):
        self.room_paths = [Path(p) for p in room_paths]
        if not self.room_paths:
            raise ValueError("S3DISDataset got an empty room list")
        self.num_points = num_points
        self.block_size = block_size
        self.training = training

        # Sample rooms proportional to their size, so big rooms (more blocks of
        # real scene) get visited more often than small ones.
        npoints = np.array([_npy_nrows(p) for p in self.room_paths], dtype=np.int64)
        self.room_prob = npoints / npoints.sum()
        # One epoch ~= one pass over every point in the dataset.
        self.length = samples_per_epoch or max(1, int(npoints.sum()) // num_points)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(None if self.training else idx)

        room_idx = int(rng.choice(len(self.room_paths), p=self.room_prob))
        room = np.load(self.room_paths[room_idx])  # (N, 7): x y z r g b label
        xyz = room[:, :3]
        rgb = room[:, 3:6]
        labels = room[:, 6].astype(np.int64)

        # Room extent, for the room-normalized ([0,1]) part of the feature.
        coord_min = xyz.min(axis=0)
        extent = np.maximum(xyz.max(axis=0) - coord_min, 1e-6)

        # --- cut a random 1m x 1m vertical column ---------------------------
        # Centre the block on a random point's (x, y); retry a few times if the
        # column came out too sparse, then take whatever we have.
        block_idxs = np.arange(len(xyz))
        center = xyz[rng.integers(len(xyz)), :2]
        for _ in range(10):
            c = xyz[rng.integers(len(xyz)), :2]
            in_x = np.abs(xyz[:, 0] - c[0]) <= self.block_size / 2
            in_y = np.abs(xyz[:, 1] - c[1]) <= self.block_size / 2
            idxs = np.where(in_x & in_y)[0]
            if len(idxs) >= MIN_BLOCK_POINTS:
                center, block_idxs = c, idxs
                break
            if len(idxs) > 0:
                center, block_idxs = c, idxs  # keep the best non-empty attempt

        # --- force exactly num_points (subsample, or upsample w/ replacement) -
        replace = len(block_idxs) < self.num_points
        choice = rng.choice(block_idxs, self.num_points, replace=replace)
        sel_xyz = xyz[choice]
        sel_rgb = rgb[choice]
        sel_lab = labels[choice]

        # --- build the 9-dim point feature ----------------------------------
        feat = np.empty((self.num_points, 9), dtype=np.float32)
        feat[:, 0] = sel_xyz[:, 0] - center[0]      # block-local x
        feat[:, 1] = sel_xyz[:, 1] - center[1]      # block-local y
        feat[:, 2] = sel_xyz[:, 2]                  # z (kept absolute)
        feat[:, 3:6] = sel_rgb / 255.0              # colour
        feat[:, 6:9] = (sel_xyz - coord_min) / extent  # room-normalized xyz

        return torch.from_numpy(feat), torch.from_numpy(sel_lab)
