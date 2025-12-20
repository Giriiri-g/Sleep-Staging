"""
Training script for MESA Transformer
====================================

Trains the MESA Transformer model on preprocessed MESA data and provides
class-wise performance metrics.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from mesa_transformer import MESATransformer
from mesa_dataloader import MESADataset, create_mesa_dataloader


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor per class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = p if target class, otherwise 1-p
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> Dict:
    """
    Compute class-wise and overall metrics.
    
    Args:
        y_true: True labels (flattened)
        y_pred: Predicted labels (flattened)
        class_names: List of class names
    
    Returns:
        Dictionary with metrics
    """
    # Overall metrics
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    
    # Weighted and macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    metrics = {
        'overall_accuracy': overall_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm
    }
    
    return metrics


def print_metrics(metrics: Dict, class_names: list, stage: str = ""):
    """Print metrics in a readable format"""
    stage_str = f" [{stage}]" if stage else ""
    print(f"\n{'='*70}")
    print(f"Performance Metrics{stage_str}")
    print(f"{'='*70}")
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"\nMacro Averages:")
    print(f"  Precision: {metrics['macro_precision']*100:.2f}%")
    print(f"  Recall:    {metrics['macro_recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['macro_f1']*100:.2f}%")
    
    print(f"\nWeighted Averages:")
    print(f"  Precision: {metrics['weighted_precision']*100:.2f}%")
    print(f"  Recall:    {metrics['weighted_recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['weighted_f1']*100:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for class_name in class_names:
        class_idx = class_names.index(class_name)
        class_metrics = metrics['per_class'][class_name]
        print(f"{class_name:<10} "
              f"{class_metrics['precision']*100:>10.2f}% "
              f"{class_metrics['recall']*100:>10.2f}% "
              f"{class_metrics['f1']*100:>10.2f}% "
              f"{class_metrics['support']:>10}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"{'':<10}", end="")
    for class_name in class_names:
        print(f"{class_name:>10}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)  # (batch, seq_len, num_channels, time_steps)
        labels = labels.to(device)  # (batch, seq_len)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(features)
        logits = output['logits']  # (batch, seq_len, num_classes)
        
        # Flatten for loss computation
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)  # (batch*seq_len, num_classes)
        labels_flat = labels.view(-1)  # (batch*seq_len)
        
        # Filter out invalid labels (shouldn't happen but safety check)
        valid_mask = (labels_flat >= 0) & (labels_flat < num_classes)
        if valid_mask.sum() == 0:
            continue
        
        logits_flat_valid = logits_flat[valid_mask]
        labels_flat_valid = labels_flat[valid_mask]
        
        # Compute loss
        loss = criterion(logits_flat_valid, labels_flat_valid)
        
        # Backward pass
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits_flat, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels_flat.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            batch_acc = accuracy_score(labels_flat.cpu().numpy(), predictions)
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_labels), np.array(all_predictions)


def validate_epoch(model, dataloader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output = model(features)
            logits = output['logits']
            
            # Flatten for loss computation
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            # Filter out invalid labels
            valid_mask = (labels_flat >= 0) & (labels_flat < num_classes)
            if valid_mask.sum() > 0:
                logits_flat_valid = logits_flat[valid_mask]
                labels_flat_valid = labels_flat[valid_mask]
                loss = criterion(logits_flat_valid, labels_flat_valid)
                total_loss += loss.item()
            
            # Predictions (use all for metrics)
            predictions = torch.argmax(logits_flat, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels_flat.cpu().numpy())
    
    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, np.array(all_labels), np.array(all_predictions)


def train(
    preprocessed_dir: str,
    csv_path: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seq_len: int = 20,
    train_split: float = 0.7,
    val_split: float = 0.15,
    device: str = None,
    checkpoint_dir: str = "checkpoints_mesa",
        resume_from: str = None,
        max_samples: int = None
):
    """
    Train MESA Transformer model.
    
    Args:
        preprocessed_dir: Directory containing preprocessed .pt files
        csv_path: Path to CSV file with mesaid and sleep_stages columns
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seq_len: Sequence length (number of epochs per sample)
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (test = 1 - train - val)
        device: Device to use ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        max_samples: Maximum number of samples to use (None = use all). Used to speed up training.
    """
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = MESADataset(
        preprocessed_dir=preprocessed_dir,
        csv_path=csv_path,
        seq_len=seq_len,
        filter_unscored=True
    )
    
    class_names = full_dataset.class_names
    num_classes = full_dataset.num_classes
    
    # Limit dataset size if specified
    total_size = len(full_dataset)
    if max_samples is not None and max_samples < total_size:
        print(f"\nLimiting dataset from {total_size:,} to {max_samples:,} samples for faster training")
        # Randomly sample indices
        indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42))[:max_samples]
        full_dataset = Subset(full_dataset, indices)
        total_size = len(full_dataset)
    
    # Split dataset
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model...")
    model = MESATransformer(
        num_channels=3,
        time_steps=3840,  # 30s * 128 Hz
        seq_len=seq_len,
        d_model=256,
        num_classes=num_classes,
        dropout=0.1,
        return_attention=False
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Compute class weights from training data to handle class imbalance
    print("\nComputing class weights from training data...")
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    # Sample a subset of training data to compute class distribution
    sample_size = min(10000, len(train_dataset))
    sample_indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))[:sample_size]
    
    for idx in sample_indices:
        _, labels = train_dataset[idx]
        for label in labels.flatten():
            if 0 <= label < num_classes:
                class_counts[label] += 1
    
    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # Compute weights: stronger inverse frequency weighting
    # Use square root of inverse frequency for more balanced weights
    total_samples = class_counts.sum()
    # Inverse frequency with square root to make it less extreme
    class_weights = torch.sqrt(total_samples / (class_counts + 1e-5))
    
    # Normalize so max weight is reasonable (cap at 10x the minimum)
    min_weight = class_weights.min()
    max_weight_allowed = min_weight * 10.0
    class_weights = torch.clamp(class_weights, max=max_weight_allowed)
    
    # Normalize to sum to num_classes (helps with stability)
    class_weights = class_weights / class_weights.mean()
    
    print(f"Class distribution (sample): {class_counts.int().tolist()}")
    print(f"Class weights (sqrt inverse freq): {class_weights.tolist()}")
    
    class_weights = class_weights.to(device)
    
    # Use Focal Loss instead of CrossEntropy for better handling of class imbalance
    # Focal loss focuses learning on hard examples
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0,  # Focusing parameter (higher = more focus on hard examples)
        reduction='mean'
    )
    
    # Optimizer with stronger weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,  # Increased from 1e-5 for stronger regularization
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0  # Will store macro F1 score
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*70)
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validate
        val_loss, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device
        )
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Compute macro F1 for better model selection (considers all classes equally)
        _, _, val_f1_macro, _ = precision_recall_fscore_support(
            val_labels, val_preds, labels=list(range(num_classes)), average='macro', zero_division=0
        )
        
        # Save checkpoint if best validation macro F1 (better for imbalanced data)
        # Fallback to accuracy if F1 is same
        use_f1 = val_f1_macro > best_val_acc if isinstance(best_val_acc, float) else True
        if use_f1 or (val_f1_macro == best_val_acc and val_acc > best_val_acc):
            best_val_acc = val_f1_macro  # Store macro F1 instead of accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            torch.save(checkpoint, checkpoint_path / 'best_model.pth')
            print(f"  ✓ Saved best model (val macro F1: {val_f1_macro*100:.2f}%, val acc: {val_acc*100:.2f}%)")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
            torch.save(checkpoint, checkpoint_path / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("Final Evaluation on Test Set")
    print("="*70)
    
    # Load best model
    best_checkpoint = torch.load(checkpoint_path / 'best_model.pth', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_labels, test_preds = validate_epoch(
        model, test_loader, criterion, device
    )
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    
    # Compute and print detailed metrics
    test_metrics = compute_class_metrics(test_labels, test_preds, class_names)
    print_metrics(test_metrics, class_names, stage="Test Set")
    
    # Also compute validation metrics
    val_metrics = compute_class_metrics(val_labels, val_preds, class_names)
    print_metrics(val_metrics, class_names, stage="Validation Set")
    
    # Save metrics in the best model checkpoint for later visualization
    try:
        best_checkpoint_path = checkpoint_path / 'best_model.pth'
        if best_checkpoint_path.exists():
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
            best_checkpoint['test_metrics'] = test_metrics
            best_checkpoint['val_metrics'] = val_metrics
            best_checkpoint['class_names'] = class_names
            torch.save(best_checkpoint, best_checkpoint_path)
            print(f"\n✓ Saved metrics to {best_checkpoint_path} for visualization")
    except Exception as e:
        print(f"\n⚠ Could not save metrics to checkpoint: {e}")
    
    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MESA Transformer")
    parser.add_argument("--preprocessed_dir", type=str, default=r"C:\mesa",
                       help="Directory containing preprocessed .pt files")
    parser.add_argument("--csv_path", type=str, default="mesa_final.csv",
                       help="Path to CSV file with mesaid and sleep_stages")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Sequence length (epochs per sample)")
    parser.add_argument("--train_split", type=float, default=0.7,
                       help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.15,
                       help="Fraction of data for validation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_mesa",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use (None = use all). Used to speed up training.")
    
    args = parser.parse_args()
    
    train(
        preprocessed_dir=args.preprocessed_dir,
        csv_path=args.csv_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seq_len=args.seq_len,
        train_split=args.train_split,
        val_split=args.val_split,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        max_samples=args.max_samples
    )

