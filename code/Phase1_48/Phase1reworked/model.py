import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        B, T, L = x.shape
        x = x.view(B * T, 1, L)
        f = self.pyramid(x)
        f = self.pool(f).squeeze(-1)
        f = self.fc(f)
        result = f.view(B, T, -1)
        return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class SleepTransformer(nn.Module):
    def __init__(self, embed_dim=128, heads=4, layers=4, dropout=0.2, max_len=512):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
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

    def forward(self, x, padding_mask=None):
        # Add positional encoding BEFORE transformer
        x = self.positional_encoding(x)
        h = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        return self.cls(h)


class SleepStagingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EpochEncoder()
        self.context = SleepTransformer()

    def forward(self, x, padding_mask=None):
        feats = self.encoder(x)
        return self.context(feats, padding_mask)

