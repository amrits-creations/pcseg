"""CPU smoke test for the Chunk-3 data pipeline.

No real dataset needed: we synthesize a couple of tiny "rooms" on disk and push
them through S3DISDataset + a DataLoader, asserting the shapes/dtypes/ranges a
model will rely on. Its only job is "the plumbing runs and the tensors are
correctly shaped" before we spend Colab GPU time. Runs in well under a second.

    pytest tests/test_dataset.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcseg import NUM_CLASSES
from pcseg.data import NUM_POINTS, S3DISDataset, make_splits


def _make_fake_room(path, n=5000, seed=0):
    """Write a random (n, 7) xyzrgb+label room.npy spanning a ~3m cube."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 3.0, size=(n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(n, 3)).astype(np.float32)
    lab = rng.integers(0, NUM_CLASSES, size=(n, 1)).astype(np.float32)
    np.save(path, np.concatenate([xyz, rgb, lab], axis=1))


def test_single_item_shapes_and_ranges(tmp_path):
    paths = [tmp_path / f"room_{i}.npy" for i in range(2)]
    for i, p in enumerate(paths):
        _make_fake_room(p, seed=i)

    ds = S3DISDataset(paths, training=True)
    pts, lab = ds[0]

    assert pts.shape == (NUM_POINTS, 9)
    assert lab.shape == (NUM_POINTS,)
    assert pts.dtype == torch.float32
    assert lab.dtype == torch.int64
    assert 0 <= int(lab.min()) and int(lab.max()) < NUM_CLASSES
    # colour channels normalized to [0, 1]
    assert pts[:, 3:6].min() >= 0.0 and pts[:, 3:6].max() <= 1.0 + 1e-6
    # room-normalized coords in [0, 1]
    assert pts[:, 6:9].min() >= -1e-6 and pts[:, 6:9].max() <= 1.0 + 1e-6


def test_dataloader_batches(tmp_path):
    paths = [tmp_path / f"room_{i}.npy" for i in range(2)]
    for i, p in enumerate(paths):
        _make_fake_room(p, seed=i)

    ds = S3DISDataset(paths, training=True, samples_per_epoch=8)
    loader = DataLoader(ds, batch_size=4)
    batch_pts, batch_lab = next(iter(loader))

    assert batch_pts.shape == (4, NUM_POINTS, 9)
    assert batch_lab.shape == (4, NUM_POINTS)


def test_make_splits_holds_out_area5_and_val(tmp_path):
    for area in ("Area_1", "Area_2", "Area_5"):
        d = tmp_path / area
        d.mkdir()
        for r in range(3):
            _make_fake_room(d / f"room_{r}.npy", n=10)

    splits = make_splits(tmp_path, val_per_area=1, seed=42)

    assert len(splits["test"]) == 3
    assert all("Area_5" in str(p) for p in splits["test"])
    assert len(splits["val"]) == 2          # 1 per training area (Area_1, Area_2)
    assert len(splits["train"]) == 4         # 6 train rooms - 2 val
    train = {str(p) for p in splits["train"]}
    val = {str(p) for p in splits["val"]}
    assert train.isdisjoint(val)
