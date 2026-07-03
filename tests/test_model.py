from __future__ import annotations

import torch
import torch.nn as nn

from pcseg import NUM_CLASSES
from pcseg.model import PointNetSemSeg, feature_transform_regularizer


def test_forward_backward_shapes():
    torch.manual_seed(0)
    B, N, C = 2, 2048, 9
    feats = torch.randn(B, N, C)
    labels = torch.randint(0, NUM_CLASSES, (B, N))

    model = PointNetSemSeg()
    logits, trans_feat = model(feats)

    assert logits.shape == (B, N, NUM_CLASSES)
    assert trans_feat.shape == (B, 64, 64)

    loss = nn.CrossEntropyLoss()(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
    loss = loss + 0.001 * feature_transform_regularizer(trans_feat)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
