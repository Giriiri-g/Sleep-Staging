import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import NUM_CLASSES


# -------------------------------
# Squeeze-and-Excitation (1D)
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, T]
        s = x.mean(dim=-1)                  # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


# -------------------------------
# Adaptive Atrous Pyramid
# -------------------------------
class AdaptiveAtrousPyramid(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, dilations=(1, 2, 4, 8)):
        super().__init__()

        self.dilations = dilations

        self.branches = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=7,
                dilation=d,
                padding=3 * d
            )
            for d in dilations
        ])

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels * len(dilations), len(dilations), kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.se = SEBlock(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, 3000]

        feats = [F.relu(conv(x)) for conv in self.branches]   # list of [B, C, T]
        stacked = torch.cat(feats, dim=1)                     # [B, C*K, T]

        weights = self.gate(stacked)                           # [B, K, 1]

        fused = 0
        for i, f in enumerate(feats):
            fused = fused + f * weights[:, i:i+1, :]

        fused = self.se(fused)
        fused = self.proj(fused)

        return fused


# -------------------------------
# Epoch Encoder
# -------------------------------
class EpochEncoder(nn.Module):
    def __init__(self, hidden_channels=64, embed_dim=128):
        super().__init__()

        self.pyramid = AdaptiveAtrousPyramid(
            in_channels=1,
            hidden_channels=hidden_channels
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, embed_dim)

    def forward(self, x):
        # x: [B, T, 3000]
        B, T, L = x.shape

        x = x.view(B * T, 1, L)
        f = self.pyramid(x)                 # [B*T, C, T']
        f = self.pool(f).squeeze(-1)        # [B*T, C]
        f = self.fc(f)                      # [B*T, D]

        return f.view(B, T, -1)              # [B, T, D]


# -------------------------------
# Transformer Sequence Model
# -------------------------------
class SleepTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x, src_key_padding_mask):
        B, T, _ = x.shape
        attn_mask = self._build_local_attention_mask(T, x.device, window=10)
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(h)

    def _build_local_attention_mask(self, T, device, window=10):
        mask = torch.ones(T, T, device=device, dtype=torch.bool)
        for i in range(T):
            start = max(0, i - window)
            end = min(T, i + window + 1)
            mask[i, start:end] = False
        return mask


# -------------------------------
# Full Model
# -------------------------------
class SleepStagingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EpochEncoder()
        self.sequence_model = SleepTransformer()

    def forward(self, x, lengths):
        # x: [B, T, 3000]

        features = self.encoder(x)           # [B, T, D]

        mask = torch.arange(x.size(1), device=lengths.device)
        mask = mask.unsqueeze(0) >= lengths.unsqueeze(1)

        logits = self.sequence_model(features, src_key_padding_mask=mask)
        return logits
