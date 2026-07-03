from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcseg.data import NUM_POINTS, S3DISDataset  # noqa: E402
from pcseg.model import PointNetSemSeg, feature_transform_regularizer  # noqa: E402

DEFAULT_PROCESSED_ROOT = Path("data/processed")

FEATURE_REG_WEIGHT = 0.001


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(model, loader, optimizer, criterion, device, max_steps=None):
    model.train()
    total_loss, total_correct, total_points, steps = 0.0, 0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)

        logits, trans_feat = model(feats)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss = loss + FEATURE_REG_WEIGHT * feature_transform_regularizer(trans_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.numel()
        total_loss += loss.item()
        steps += 1
        print(f"  step {steps:>4}  loss {loss.item():.4f}")
        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(steps, 1), total_correct / max(total_points, 1)


def _write_fake_room(path: Path, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 3.0, size=(n, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(n, 3)).astype(np.float32)
    labels = rng.integers(0, 13, size=(n, 1)).astype(np.float32)
    np.save(path, np.concatenate([xyz, rgb, labels], axis=1))


def build_datasets(processed_root: Path, splits_path: Path, num_points: int):
    splits = json.loads(splits_path.read_text())
    train_paths = [processed_root / p for p in splits["train"]]
    val_paths = [processed_root / p for p in splits["val"]]
    train_ds = S3DISDataset(train_paths, num_points=num_points, training=True)
    val_ds = (
        S3DISDataset(val_paths, num_points=num_points, training=False)
        if val_paths
        else None
    )
    return train_ds, val_ds


def build_smoke_dataset(num_points: int) -> S3DISDataset:
    tmp = Path(tempfile.mkdtemp(prefix="pcseg_smoke_"))
    rooms = [tmp / "roomA.npy", tmp / "roomB.npy"]
    for i, r in enumerate(rooms):
        _write_fake_room(r, seed=i)
    return S3DISDataset(rooms, num_points=num_points, training=True, samples_per_epoch=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_PROCESSED_ROOT / "splits.json")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic 2-step run on CPU to validate the pipeline, then exit.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.device = "cpu"
        args.num_points = 2048
        args.batch_size = 2
        args.num_workers = 0

    device = pick_device(args.device)
    print(f"device: {device}")

    if not args.smoke:
        wandb.init(
            project="pcseg",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_points": args.num_points,
                "lr": args.lr,
                "feature_reg_weight": FEATURE_REG_WEIGHT,
                "device": str(device),
            },
        )

    if args.smoke:
        train_ds = build_smoke_dataset(args.num_points)
        val_ds = None
    else:
        train_ds, val_ds = build_datasets(args.processed_root, args.splits, args.num_points)
    print(f"train blocks/epoch: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = PointNetSemSeg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.smoke:
        print("running smoke test: 2 training steps on synthetic data")
        loss, acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_steps=2
        )
        print(f"smoke OK — mean loss {loss:.4f}, point acc {acc:.3f}")
        return

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  epoch {epoch}: mean loss {loss:.4f}, point acc {acc:.3f}")
        wandb.log({"train/loss": loss, "train/point_acc": acc}, step=epoch)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"pointnet_{wandb.run.id}.pt"
    torch.save({"model_state": model.state_dict(), "config": dict(wandb.config)}, ckpt_path)
    artifact = wandb.Artifact("pointnet-semseg", type="model")
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
