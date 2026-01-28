import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class FocalLoss(nn.Module):
    """
    Focal loss for epoch-wise classification
    logits: [B, C]
    targets: [B]
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: [B, C]
        # targets: [B]

        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()
