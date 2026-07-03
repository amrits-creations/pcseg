from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_POINTS = 4096
BLOCK_SIZE = 1.0
MIN_BLOCK_POINTS = 1024
TEST_AREA = "Area_5"


def _npy_nrows(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


class S3DISDataset(Dataset):
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

        npoints = np.array([_npy_nrows(p) for p in self.room_paths], dtype=np.int64)
        self.room_prob = npoints / npoints.sum()
        self.length = samples_per_epoch or max(1, int(npoints.sum()) // num_points)

        self.valid_centers_per_room = self._precompute_valid_centers()

    def _precompute_valid_centers(self) -> dict[int, np.ndarray]:
        valid_centers = {}
        for room_idx, room_path in enumerate(self.room_paths):
            room = np.load(room_path)
            xyz = room[:, :3]
            coord_min = xyz.min(axis=0)
            coord_max = xyz.max(axis=0)

            margin = self.block_size / 2
            valid_mask = (
                (xyz[:, 0] - coord_min[0] >= margin) &
                (coord_max[0] - xyz[:, 0] >= margin) &
                (xyz[:, 1] - coord_min[1] >= margin) &
                (coord_max[1] - xyz[:, 1] >= margin)
            )
            valid_centers[room_idx] = np.where(valid_mask)[0]
        return valid_centers

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(None if self.training else idx)

        room_idx = int(rng.choice(len(self.room_paths), p=self.room_prob))
        room = np.load(self.room_paths[room_idx])
        xyz = room[:, :3]
        rgb = room[:, 3:6]
        labels = room[:, 6].astype(np.int64)

        coord_min = xyz.min(axis=0)
        extent = np.maximum(xyz.max(axis=0) - coord_min, 1e-6)

        valid_idxs = self.valid_centers_per_room[room_idx]
        center_idx = rng.choice(valid_idxs)
        c = xyz[center_idx, :2]
        in_x = np.abs(xyz[:, 0] - c[0]) <= self.block_size / 2
        in_y = np.abs(xyz[:, 1] - c[1]) <= self.block_size / 2
        block_idxs = np.where(in_x & in_y)[0]

        replace = len(block_idxs) < self.num_points
        choice = rng.choice(block_idxs, self.num_points, replace=replace)
        sel_xyz = xyz[choice]
        sel_rgb = rgb[choice]
        sel_lab = labels[choice]

        feat = np.empty((self.num_points, 9), dtype=np.float32)
        feat[:, 0] = sel_xyz[:, 0] - c[0]
        feat[:, 1] = sel_xyz[:, 1] - c[1]
        feat[:, 2] = sel_xyz[:, 2]
        feat[:, 3:6] = sel_rgb / 255.0
        feat[:, 6:9] = (sel_xyz - coord_min) / extent

        return torch.from_numpy(feat), torch.from_numpy(sel_lab)
