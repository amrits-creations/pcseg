from __future__ import annotations

import torch
import torch.nn as nn

from pcseg import NUM_CLASSES


class TNet(nn.Module):

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential( # Same thing as multiplying each point by the same 3 numbers (input channels = k = 3).
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), # (clearing my confusion) output channels = number of filters = 64 here.
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential( # Transform the 1024-element vector (after max pool) into a 3*3 (n*n) vector so that we can turn it into a n*n matrix to transform the points in the set to realign them.
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv(x)
        x = x.max(dim=2)[0] # Produces a single 1024-element vector from the entire set of points.
        x = self.fc(x)
        identity = torch.eye(self.k, device=x.device).flatten().view(1, self.k * self.k)
        x = x + identity # If the matrix produces a 0 output (like randomly initialized matrices often do), we don't zero out the points that we're supposed to realign.
        return x.view(B, self.k, self.k) # Reshape into a k*k vector usign a view.


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    d = trans.shape[1]
    eye = torch.eye(d, device=trans.device).unsqueeze(0)
    prod = torch.bmm(trans, trans.transpose(2, 1))
    return torch.mean(torch.linalg.norm(prod - eye, dim=(1, 2)))


class PointNetSemSeg(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES, in_channel: int = 9):
        super().__init__()
        self.in_channel = in_channel
        self.input_tnet = TNet(k=3)
        self.mlp1 = nn.Sequential( # Here too, the conv1d layers serve as simple MLPs.
            nn.Conv1d(in_channel, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.feature_tnet = TNet(k=64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[1]
        x = x.permute(0, 2, 1) # Rearranged for conv1d (batch, channels, points)
        xyz, feat = x[:, :3, :], x[:, 3:, :] 
        # xyz -> shape: (32, 3, 1024) - the x, y, z coordinates
        # feat -> shape: (32, 6, 1024) - the additional features

        t_in = self.input_tnet(xyz)
        xyz = torch.bmm(t_in, xyz)
        x = torch.cat([xyz, feat], dim=1)

        x = self.mlp1(x)

        t_feat = self.feature_tnet(x)
        x = torch.bmm(t_feat, x)
        point_feat = x

        x = self.mlp2(x)
        global_feat = x.max(dim=2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, N)

        x = torch.cat([point_feat, global_feat], dim=1)
        logits = self.seg_head(x)
        return logits.permute(0, 2, 1), t_feat
