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

from pcseg import NUM_CLASSES  # noqa: E402
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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run the model over the val set and return (loss, point_acc, mIoU).

    mIoU = mean over classes of IoU_c = intersection_c / union_c, where
    intersection/union are accumulated point counts across the whole val set.
    Classes absent from the val set (union == 0) are skipped, not counted as 0.
    """
    model.eval()
    inter = torch.zeros(NUM_CLASSES, dtype=torch.long)
    union = torch.zeros(NUM_CLASSES, dtype=torch.long)
    total_loss, total_correct, total_points, steps = 0.0, 0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)

        logits, _ = model(feats)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_points += labels.numel()
        total_loss += loss.item()
        steps += 1

        preds = preds.reshape(-1).cpu()
        labs = labels.reshape(-1).cpu()
        for c in range(NUM_CLASSES):
            pred_c = preds == c
            lab_c = labs == c
            inter[c] += (pred_c & lab_c).sum()
            union[c] += (pred_c | lab_c).sum()

    present = union > 0
    iou = inter[present].float() / union[present].float()
    miou = iou.mean().item() if present.any() else 0.0
    return total_loss / max(steps, 1), total_correct / max(total_points, 1), miou


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


def build_smoke_datasets(num_points: int) -> tuple[S3DISDataset, S3DISDataset]:
    tmp = Path(tempfile.mkdtemp(prefix="pcseg_smoke_"))
    train_rooms = [tmp / "roomA.npy", tmp / "roomB.npy"]
    val_rooms = [tmp / "roomV.npy"]
    for i, r in enumerate(train_rooms + val_rooms):
        _write_fake_room(r, seed=i)
    train_ds = S3DISDataset(
        train_rooms, num_points=num_points, training=True, samples_per_epoch=8
    )
    val_ds = S3DISDataset(
        val_rooms, num_points=num_points, training=False, samples_per_epoch=4
    )
    return train_ds, val_ds


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
        train_ds, val_ds = build_smoke_datasets(args.num_points)
    else:
        train_ds, val_ds = build_datasets(args.processed_root, args.splits, args.num_points)
    print(f"train blocks/epoch: {len(train_ds)}")
    print(f"val blocks/epoch: {len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        if val_ds
        else None
    )

    model = PointNetSemSeg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.smoke:
        print("running smoke test: 2 training steps + 1 eval pass on synthetic data")
        loss, acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_steps=2
        )
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device)
        print(
            f"smoke OK — train loss {loss:.4f}, acc {acc:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}, mIoU {val_miou:.3f}"
        )
        return

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs}")
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  epoch {epoch}: train loss {loss:.4f}, point acc {acc:.3f}")
        log = {"train/loss": loss, "train/point_acc": acc}
        if val_loader is not None:
            val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device)
            print(f"  epoch {epoch}: val loss {val_loss:.4f}, acc {val_acc:.3f}, mIoU {val_miou:.3f}")
            log.update({"val/loss": val_loss, "val/point_acc": val_acc, "val/mIoU": val_miou})
        wandb.log(log, step=epoch)

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
