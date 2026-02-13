import torch
import torch.nn as nn
import math
from dataset import NUM_CLASSES


# ==========================================
# Positional Encoding (unchanged)
# ==========================================
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

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


# ==========================================
# Transformer (unchanged)
# ==========================================
class SleepTransformer(nn.Module):
    def __init__(self, embed_dim=128, heads=4, layers=3, dropout=0.2, max_len=512):
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
        x = self.positional_encoding(x)
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.cls(h)


# ==========================================
# SleepStagingModel (UPDATED)
# ==========================================
class SleepStagingModel(nn.Module):
    def __init__(
        self,
        input_dim=34,      # spectral feature dimension
        embed_dim=128,
        heads=4,
        layers=3,
        dropout=0.2
    ):
        super().__init__()

        # Replace CNN EpochEncoder with simple projection
        self.input_projection = nn.Linear(input_dim, embed_dim)

        self.context = SleepTransformer(
            embed_dim=embed_dim,
            heads=heads,
            layers=layers,
            dropout=dropout
        )

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, 34]
        """
        feats = self.input_projection(x)
        return self.context(feats, padding_mask)
