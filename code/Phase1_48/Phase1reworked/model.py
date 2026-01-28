import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataset import NUM_CLASSES


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class AdaptiveAtrousPyramid(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, dilations=(1, 2, 4, 8)):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=7,
                dilation=d,
                padding=3 * d
            ) for d in dilations
        ])

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels * len(dilations), len(dilations), 1),
            nn.Softmax(dim=1)
        )

        self.se = SEBlock(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        feats = [F.relu(b(x)) for b in self.branches]
        stacked = torch.cat(feats, dim=1)
        weights = self.gate(stacked)

        out = 0
        for i, f in enumerate(feats):
            out = out + f * weights[:, i:i+1]

        return self.proj(self.se(out))


class EpochEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.pyramid = AdaptiveAtrousPyramid()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        # x: [B, 21, 3000]
        start_time = time.time()
        B, T, L = x.shape
        x = x.view(B * T, 1, L)
        pyramid_start = time.time()
        f = self.pyramid(x)
        pyramid_time = time.time() - pyramid_start
        f = self.pool(f).squeeze(-1)
        fc_start = time.time()
        f = self.fc(f)
        fc_time = time.time() - fc_start
        total_time = time.time() - start_time
        if B > 1:  # Log for batches, not single samples
            print(f"[DEBUG] EpochEncoder forward: Pyramid={pyramid_time:.4f}s, FC={fc_time:.4f}s, Total={total_time:.4f}s")
        return f.view(B, T, -1)


class SleepTransformer(nn.Module):
    def __init__(self, embed_dim=128, heads=4, layers=2, dropout=0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.cls = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x):
        h = self.encoder(x)
        center = h[:, h.size(1) // 2]
        return self.cls(center)


class SleepStagingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EpochEncoder()
        self.context = SleepTransformer()

    def forward(self, x):
        feats = self.encoder(x)
        return self.context(feats)
