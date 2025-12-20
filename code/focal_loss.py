"""
Focal Loss for Class Imbalance
===============================
Focal loss implementation to address class imbalance in multiclass binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the predicted probability for the true class
    - alpha_t is a weighting factor for class t
    - gamma is the focusing parameter (gamma > 0 reduces the relative loss for well-classified examples)
    
    Args:
        alpha: Weighting factor. Can be:
            - float: Same weight for all classes
            - list/tensor: Per-class weights [weight_negative, weight_positive] for each class
            - None: No weighting (alpha=1)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                self.alpha = torch.tensor([1.0 - alpha, alpha], dtype=torch.float32)
            else:
                self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) - logits
            targets: (batch, num_classes) - binary targets
        
        Returns:
            loss: Focal loss value
        """
        # Compute BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Get probabilities
        p_t = torch.sigmoid(inputs)
        
        # For positive class: p_t, for negative class: 1 - p_t
        # We need to select the right probability based on target
        p_t_selected = torch.where(targets == 1, p_t, 1 - p_t)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t_selected) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Handle multi-dimensional alpha (per-class weights)
            if len(self.alpha.shape) > 1:
                # alpha shape: (num_classes, 2) - per-class positive/negative weights
                batch_size, num_classes = targets.shape
                alpha_selected = torch.zeros_like(targets)
                for i in range(num_classes):
                    alpha_selected[:, i] = torch.where(
                        targets[:, i] == 1,
                        self.alpha[i, 1],
                        self.alpha[i, 0]
                    )
                focal_loss = alpha_selected * focal_weight * bce_loss
            else:
                # Single alpha for all classes: [alpha_neg, alpha_pos]
                alpha_t = torch.where(targets == 1, self.alpha[1], self.alpha[0])
                focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    
    Args:
        pos_weight: Weight for positive class. Can be:
            - float: Same weight for all classes
            - tensor: Per-class positive weights (num_classes,)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight=None, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        if pos_weight is not None:
            if isinstance(pos_weight, (list, tuple)):
                self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            elif isinstance(pos_weight, (int, float)):
                self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) - logits
            targets: (batch, num_classes) - binary targets
        
        Returns:
            loss: Weighted BCE loss value
        """
        if self.pos_weight is not None:
            if self.pos_weight.device != inputs.device:
                self.pos_weight = self.pos_weight.to(inputs.device)
            
            # Expand pos_weight to match number of classes if needed
            if self.pos_weight.shape[0] == 1:
                # Same weight for all classes
                pos_weight = self.pos_weight.expand(targets.shape[1])
            else:
                pos_weight = self.pos_weight
            
            return F.binary_cross_entropy_with_logits(
                inputs, targets,
                pos_weight=pos_weight,
                reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                inputs, targets,
                reduction=self.reduction
            )

