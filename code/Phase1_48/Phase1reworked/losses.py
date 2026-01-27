import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for sequence classification
    """

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [B, T, C]
        targets: [B, T]
        """
        B, T, C = logits.shape

        logits = logits.view(-1, C)
        targets = targets.view(-1)

        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]

        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)

        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()
