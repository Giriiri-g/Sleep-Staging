"""
Training Script for Comorbidity Classifier
==========================================
Train and evaluate the comorbidity classifier model.
"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from comorbidity_classifier_dataset import create_dataloader
from comorbidity_classifier import ComorbidityClassifier
from focal_loss import FocalLoss, WeightedBCELoss
from sklearn.model_selection import StratifiedShuffleSplit


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    for sleep_stages, seq_lengths, features, targets in tqdm(dataloader, desc="Training"):
        sleep_stages = sleep_stages.to(device)
        seq_lengths = seq_lengths.to(device)
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sleep_stages, seq_lengths, features)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate metrics (detach to avoid gradient tracking)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().detach().cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    
    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Per-class metrics
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # ROC AUC (per class)
    roc_auc = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            roc_auc.append(auc)
        except ValueError:
            roc_auc.append(0.0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'roc_auc': roc_auc
    }


def compute_class_weights(targets, method='balanced', multiplier=1.0):
    """
    Compute class weights for imbalanced data.
    
    Args:
        targets: (n_samples, n_classes) - binary targets
        method: 'balanced', 'inverse', 'aggressive', or 'very_aggressive'
        multiplier: Additional multiplier for weights (default: 1.0)
    
    Returns:
        pos_weights: (n_classes,) - positive class weights for each class
    """
    n_samples, n_classes = targets.shape
    pos_weights = np.zeros(n_classes)
    
    for i in range(n_classes):
        n_pos = targets[:, i].sum()
        n_neg = n_samples - n_pos
        
        if n_pos == 0:
            pos_weights[i] = 1.0
        elif method == 'balanced':
            # sklearn style: n_samples / (n_classes * n_pos)
            pos_weights[i] = (n_samples / (2.0 * n_pos)) * multiplier
        elif method == 'inverse':
            # Inverse frequency
            pos_weights[i] = (n_neg / n_pos) * multiplier
        elif method == 'aggressive':
            # More aggressive: 3x inverse frequency
            pos_weights[i] = (n_neg / n_pos) * 3.0 * multiplier
        elif method == 'very_aggressive':
            # Very aggressive: 5x inverse frequency
            pos_weights[i] = (n_neg / n_pos) * 5.0 * multiplier
        else:
            pos_weights[i] = 1.0 * multiplier
    
    return torch.tensor(pos_weights, dtype=torch.float32)


def optimize_thresholds(probs, targets, metric='f1'):
    """
    Optimize per-class thresholds to maximize metric.
    
    Args:
        probs: (n_samples, n_classes) - predicted probabilities
        targets: (n_samples, n_classes) - true binary targets
        metric: 'f1', 'f1_macro', or 'balanced_accuracy'
    
    Returns:
        thresholds: (n_classes,) - optimized thresholds for each class
    """
    n_classes = targets.shape[1]
    thresholds = np.zeros(n_classes)
    
    for i in range(n_classes):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        
        if y_true.sum() == 0:
            thresholds[i] = 0.5
            continue
        
        def score_func(threshold):
            y_pred = (y_prob >= threshold).astype(int)
            if metric == 'f1':
                return -f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced_accuracy':
                try:
                    from sklearn.metrics import balanced_accuracy_score
                    return -balanced_accuracy_score(y_true, y_pred)
                except ImportError:
                    # Fallback: compute manually
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        return -(sensitivity + specificity) / 2
                    return 0
            else:
                return -f1_score(y_true, y_pred, zero_division=0)
        
        # Optimize threshold
        result = minimize_scalar(score_func, bounds=(0, 1), method='bounded')
        thresholds[i] = result.x
    
    return thresholds


def create_stratified_splits(df, target_cols, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create stratified train/val/test splits for multiclass binary data.
    
    Args:
        df: DataFrame with data
        target_cols: List of target column names
        test_size: Test set proportion
        val_size: Validation set proportion (of training set)
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    # Create a combined stratification label
    # For multiclass binary, we create a label based on the combination of classes
    labels = []
    for _, row in df.iterrows():
        label_parts = []
        for col in target_cols:
            label_parts.append(str(int(row[col])))
        labels.append('_'.join(label_parts))
    
    df_strat = df.copy()
    df_strat['_stratify_label'] = labels
    
    # First split: train+val vs test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(df_strat, df_strat['_stratify_label']))
    
    train_val_df = df_strat.iloc[train_val_idx].copy()
    test_df = df_strat.iloc[test_idx].copy()
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split data
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx, val_idx = next(sss2.split(train_val_df, train_val_df['_stratify_label']))
    
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    
    # Remove stratification column
    for df_split in [train_df, val_df, test_df]:
        df_split.drop('_stratify_label', axis=1, inplace=True)
    
    return train_df, val_df, test_df


def plot_confusion_matrices(test_metrics, target_names, output_dir):
    """Plot confusion matrices for each class"""
    predictions = test_metrics['predictions']
    targets = test_metrics['targets']
    
    n_classes = len(target_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    for i, name in enumerate(target_names):
        cm = confusion_matrix(targets[:, i], predictions[:, i])
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[i], cbar_kws={'label': 'Normalized'})
        axes[i].set_title(f'{name}\n(Total: {cm.sum()})', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to {os.path.join(output_dir, 'confusion_matrices.png')}")


def plot_roc_curves(test_metrics, target_names, output_dir):
    """Plot ROC curves for each class"""
    probabilities = test_metrics['probabilities']
    targets = test_metrics['targets']
    roc_aucs = test_metrics['roc_auc']
    
    n_classes = len(target_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    for i, name in enumerate(target_names):
        if targets[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
            axes[i].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_aucs[i]:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            axes[i].set_xlabel('False Positive Rate', fontsize=10)
            axes[i].set_ylabel('True Positive Rate', fontsize=10)
            axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No positive samples', ha='center', va='center')
            axes[i].set_title(f'{name}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {os.path.join(output_dir, 'roc_curves.png')}")


def plot_pr_curves(test_metrics, target_names, output_dir):
    """Plot Precision-Recall curves for each class"""
    probabilities = test_metrics['probabilities']
    targets = test_metrics['targets']
    
    n_classes = len(target_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    for i, name in enumerate(target_names):
        if targets[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(targets[:, i], probabilities[:, i])
            ap = average_precision_score(targets[:, i], probabilities[:, i])
            axes[i].plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
            axes[i].set_xlabel('Recall', fontsize=10)
            axes[i].set_ylabel('Precision', fontsize=10)
            axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No positive samples', ha='center', va='center')
            axes[i].set_title(f'{name}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curves to {os.path.join(output_dir, 'pr_curves.png')}")


def plot_class_performance(test_metrics, target_names, output_dir):
    """Plot bar chart of per-class metrics"""
    precision = test_metrics['precision']
    recall = test_metrics['recall']
    f1 = test_metrics['f1']
    roc_auc = test_metrics['roc_auc']
    
    x = np.arange(len(target_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x - 0.5*width, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + 0.5*width, f1, width, label='F1', alpha=0.8)
    ax.bar(x + 1.5*width, roc_auc, width, label='ROC AUC', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class performance plot to {os.path.join(output_dir, 'class_performance.png')}")


def plot_training_history(train_history, val_history, output_dir):
    """Plot training history"""
    epochs = range(1, len(train_history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, [h['loss'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [h['loss'] for h in val_history], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Macro F1
    axes[0, 1].plot(epochs, [h['macro_f1'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, [h['macro_f1'] for h in val_history], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Macro F1', fontsize=11)
    axes[0, 1].set_title('Training and Validation Macro F1', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, [h['accuracy'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, [h['accuracy'] for h in val_history], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy', fontsize=11)
    axes[1, 0].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC AUC (mean)
    train_aucs = [np.mean(h['roc_auc']) if 'roc_auc' in h else 0 for h in train_history]
    val_aucs = [np.mean(h['roc_auc']) if 'roc_auc' in h else 0 for h in val_history]
    axes[1, 1].plot(epochs, train_aucs, 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, val_aucs, 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Mean ROC AUC', fontsize=11)
    axes[1, 1].set_title('Training and Validation Mean ROC AUC', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {os.path.join(output_dir, 'training_history.png')}")


def evaluate(model, dataloader, criterion, device, target_names=None, thresholds=None):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for sleep_stages, seq_lengths, features, targets in tqdm(dataloader, desc="Evaluating"):
            sleep_stages = sleep_stages.to(device)
            seq_lengths = seq_lengths.to(device)
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(sleep_stages, seq_lengths, features)
            loss = criterion(logits, targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            
            # Use optimized thresholds if provided
            probs_detached = probs.detach().cpu().numpy()
            if thresholds is not None:
                preds = np.zeros_like(probs_detached)
                for i, thresh in enumerate(thresholds):
                    preds[:, i] = (probs_detached[:, i] >= thresh).astype(int)
            else:
                preds = (probs >= 0.5).long().detach().cpu().numpy()
            
            all_preds.append(preds)
            all_probs.append(probs_detached)
            all_targets.append(targets.detach().cpu().numpy())
    
    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Per-class metrics
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # ROC AUC (per class)
    roc_auc = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            roc_auc.append(auc)
        except ValueError:
            roc_auc.append(0.0)
    
    # Classification report
    if target_names is None:
        target_names = ['insomnia', 'restless leg', 'apnea']
    
    report = classification_report(
        all_targets, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'roc_auc': roc_auc,
        'classification_report': report,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []
    
    for sleep_stages, seq_lengths, features, targets in tqdm(dataloader, desc="Training"):
        sleep_stages = sleep_stages.to(device)
        seq_lengths = seq_lengths.to(device)
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sleep_stages, seq_lengths, features)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate metrics (detach to avoid gradient tracking)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().detach().cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    
    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Per-class metrics
    precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # ROC AUC (per class)
    roc_auc = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            roc_auc.append(auc)
        except ValueError:
            roc_auc.append(0.0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'roc_auc': roc_auc
    }


def main():
    parser = argparse.ArgumentParser(description='Train Comorbidity Classifier')
    parser.add_argument('--csv_path', type=str, default='mesa_final.csv',
                        help='Path to CSV file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/comorbidity',
                        help='Directory to save checkpoints and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set size (of training set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='Maximum sequence length (None = use all)')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='RNN type')
    parser.add_argument('--rnn_hidden_dim', type=int, default=128,
                        help='RNN hidden dimension')
    parser.add_argument('--rnn_num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Sleep stage embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_stratified_split', action='store_true',
                        help='Use stratified sampling for train/test splits')
    parser.add_argument('--loss_type', type=str, default='bce', 
                        choices=['bce', 'weighted_bce', 'focal'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=3.0,
                        help='Focal loss gamma parameter (higher = more focus on hard examples)')
    parser.add_argument('--class_weight_method', type=str, default='balanced',
                        choices=['balanced', 'inverse', 'aggressive', 'very_aggressive', 'none'],
                        help='Method for computing class weights')
    parser.add_argument('--optimize_thresholds', action='store_true',
                        help='Optimize per-class thresholds on validation set')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                        choices=['f1', 'balanced_accuracy'],
                        help='Metric to optimize for threshold selection')
    parser.add_argument('--use_weighted_sampling', action='store_true',
                        help='Use weighted sampling in DataLoader')
    parser.add_argument('--weight_method', type=str, default='per_class',
                        choices=['inverse', 'inverse_sqrt', 'balanced', 'aggressive', 'per_class'],
                        help='Method for computing sample weights')
    parser.add_argument('--class_weight_multiplier', type=float, default=1.0,
                        help='Multiplier for class weights (increase for more aggressive weighting)')
    parser.add_argument('--early_stop_metric', type=str, default='macro_f1',
                        choices=['loss', 'macro_f1', 'macro_recall', 'roc_auc'],
                        help='Metric for early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.0 = no smoothing)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and split data
    print("Loading data...")
    df = pd.read_csv(args.csv_path)
    
    # Get target column names
    target_cols = ['insomnia', 'restless leg', 'apnea']
    
    # Split into train/test
    if args.use_stratified_split:
        print("Using stratified sampling for train/test splits...")
        train_df, val_df, test_df = create_stratified_splits(
            df, target_cols, 
            test_size=args.test_size, 
            val_size=args.val_size, 
            random_state=args.seed
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=args.seed, shuffle=True
        )
        train_df, val_df = train_test_split(
            train_df, test_size=args.val_size, random_state=args.seed, shuffle=True
        )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Print class distribution
    print("\nClass distribution in splits:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        for col in target_cols:
            pos_count = split_df[col].sum()
            pos_pct = 100 * pos_count / len(split_df)
            print(f"  {col}: {pos_count} positive ({pos_pct:.1f}%)")
    
    # Save splits
    train_df.to_csv(os.path.join(args.output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_split.csv'), index=False)
    
    # Create temporary CSV files for each split
    train_csv = os.path.join(args.output_dir, 'train_split.csv')
    val_csv = os.path.join(args.output_dir, 'val_split.csv')
    test_csv = os.path.join(args.output_dir, 'test_split.csv')
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, train_dataset = create_dataloader(
        train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        max_seq_len=args.max_seq_len,
        use_weighted_sampling=args.use_weighted_sampling,
        weight_method=args.weight_method
    )
    
    val_loader, val_dataset = create_dataloader(
        val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        scaler=train_dataset.get_scaler(),
        max_seq_len=args.max_seq_len
    )
    
    test_loader, test_dataset = create_dataloader(
        test_csv,
        batch_size=args.batch_size,
        shuffle=False,
        scaler=train_dataset.get_scaler(),
        max_seq_len=args.max_seq_len
    )
    
    # Create model
    num_features = len(train_dataset.feature_cols)
    model = ComorbidityClassifier(
        num_sleep_stages=6,
        embedding_dim=args.embedding_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        rnn_num_layers=args.rnn_num_layers,
        rnn_type=args.rnn_type,
        num_features=num_features,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compute class weights and setup loss function
    print("\nSetting up loss function...")
    train_targets = train_dataset.targets
    
    if args.loss_type == 'weighted_bce':
        pos_weights = compute_class_weights(
            train_targets, 
            method=args.class_weight_method,
            multiplier=args.class_weight_multiplier
        )
        pos_weights = pos_weights.to(device)
        criterion = WeightedBCELoss(pos_weight=pos_weights)
        print(f"Using WeightedBCE with positive weights: {pos_weights.cpu().numpy()}")
        print(f"  Weight multiplier: {args.class_weight_multiplier}")
    elif args.loss_type == 'focal':
        # Compute alpha for focal loss (inverse frequency)
        pos_weights = compute_class_weights(train_targets, method='inverse')
        # Normalize to sum to 1 for each class
        alpha_per_class = []
        for i in range(len(target_cols)):
            n_pos = train_targets[:, i].sum()
            n_neg = len(train_targets) - n_pos
            if n_pos > 0:
                alpha_neg = n_neg / len(train_targets)
                alpha_pos = n_pos / len(train_targets)
                alpha_per_class.append([alpha_neg, alpha_pos])
            else:
                alpha_per_class.append([0.5, 0.5])
        
        alpha_tensor = torch.tensor(alpha_per_class, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=args.focal_gamma)
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
        print(f"Alpha weights per class: {alpha_tensor.cpu().numpy()}")
    else:
        if args.class_weight_method != 'none':
            pos_weights = compute_class_weights(
                train_targets, 
                method=args.class_weight_method,
                multiplier=args.class_weight_multiplier
            )
            pos_weights = pos_weights.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            print(f"Using BCEWithLogitsLoss with positive weights: {pos_weights.cpu().numpy()}")
            print(f"  Weight multiplier: {args.class_weight_multiplier}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using standard BCEWithLogitsLoss")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Learning rate scheduler (use 'max' mode for F1/recall metrics)
    scheduler_mode = 'min' if args.early_stop_metric == 'loss' else 'max'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=0.5, patience=5
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_score = 0.0
    best_val_f1 = 0.0
    early_stop_counter = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        best_val_score = checkpoint.get('best_val_score', 0.0)
    
    # Training loop
    print("\nStarting training...")
    train_history = []
    val_history = []
    current_lr = args.lr
    optimal_thresholds = None
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        train_history.append(train_metrics)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Per-class F1: {train_metrics['f1']}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, train_dataset.target_cols)
        val_history.append(val_metrics)
        
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Per-class F1: {val_metrics['f1']}")
        print(f"  ROC AUC: {val_metrics['roc_auc']}")
        
        # Get metric for early stopping and scheduling
        if args.early_stop_metric == 'loss':
            val_score = -val_metrics['loss']  # Negative because lower is better
        elif args.early_stop_metric == 'macro_f1':
            val_score = val_metrics['macro_f1']
        elif args.early_stop_metric == 'macro_recall':
            val_score = val_metrics['macro_recall']
        elif args.early_stop_metric == 'roc_auc':
            val_score = np.mean(val_metrics['roc_auc'])
        else:
            val_score = val_metrics['macro_f1']
        
        # Learning rate scheduling
        old_lr = current_lr
        scheduler.step(val_score if scheduler_mode == 'max' else val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}")
        
        # Early stopping and model saving
        improved = False
        if val_score > best_val_score + args.min_delta:
            best_val_score = val_score
            best_val_f1 = val_metrics['macro_f1']
            early_stop_counter = 0
            improved = True
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_score,
                'best_val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model ({args.early_stop_metric}: {val_score:.4f}, F1: {best_val_f1:.4f})")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"No improvement in {args.early_stop_metric} for {args.early_stop_patience} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    
    # Try to load best model, fallback to latest checkpoint if not found
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Find latest checkpoint
        checkpoint_files = glob.glob(os.path.join(args.output_dir, 'checkpoint_epoch_*.pt'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            print(f"Best model not found. Loading latest checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoints found. Using current model state.")
    
    # Optimize thresholds on validation set for final test evaluation
    if args.optimize_thresholds:
        print("\nOptimizing thresholds on validation set for final evaluation...")
        model.eval()
        all_val_probs = []
        all_val_targets = []
        with torch.no_grad():
            for sleep_stages, seq_lengths, features, targets in val_loader:
                sleep_stages = sleep_stages.to(device)
                seq_lengths = seq_lengths.to(device)
                features = features.to(device)
                logits = model(sleep_stages, seq_lengths, features)
                probs = torch.sigmoid(logits)
                all_val_probs.append(probs.detach().cpu().numpy())
                all_val_targets.append(targets.detach().cpu().numpy())
        
        all_val_probs = np.vstack(all_val_probs)
        all_val_targets = np.vstack(all_val_targets)
        
        optimal_thresholds = optimize_thresholds(
            all_val_probs, all_val_targets, metric=args.threshold_metric
        )
        print(f"Using optimized thresholds: {optimal_thresholds}")
    else:
        optimal_thresholds = None
    
    test_metrics = evaluate(
        model, test_loader, criterion, device, 
        test_dataset.target_cols, 
        thresholds=optimal_thresholds
    )
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"\nPer-class metrics:")
    for i, name in enumerate(test_dataset.target_cols):
        print(f"  {name}:")
        print(f"    Precision: {test_metrics['precision'][i]:.4f}")
        print(f"    Recall: {test_metrics['recall'][i]:.4f}")
        print(f"    F1: {test_metrics['f1'][i]:.4f}")
        print(f"    ROC AUC: {test_metrics['roc_auc'][i]:.4f}")
        if optimal_thresholds is not None:
            print(f"    Threshold: {optimal_thresholds[i]:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        test_metrics['targets'],
        test_metrics['predictions'],
        target_names=test_dataset.target_cols,
        zero_division=0
    ))
    
    # Save results
    results = {
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'macro_precision': float(test_metrics['macro_precision']),
            'macro_recall': float(test_metrics['macro_recall']),
            'macro_f1': float(test_metrics['macro_f1']),
            'per_class': {
                name: {
                    'precision': float(test_metrics['precision'][i]),
                    'recall': float(test_metrics['recall'][i]),
                    'f1': float(test_metrics['f1'][i]),
                    'roc_auc': float(test_metrics['roc_auc'][i])
                }
                for i, name in enumerate(test_dataset.target_cols)
            }
        },
        'classification_report': test_metrics['classification_report'],
        'optimal_thresholds': optimal_thresholds.tolist() if optimal_thresholds is not None else None,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save training history
    history_dict = {
        'train_history': train_history,
        'val_history': val_history
    }
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2, default=str)
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.savez(
        os.path.join(args.output_dir, 'test_predictions.npz'),
        predictions=test_metrics['predictions'],
        probabilities=test_metrics['probabilities'],
        targets=test_metrics['targets']
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrices(test_metrics, test_dataset.target_cols, args.output_dir)
    plot_roc_curves(test_metrics, test_dataset.target_cols, args.output_dir)
    plot_pr_curves(test_metrics, test_dataset.target_cols, args.output_dir)
    plot_class_performance(test_metrics, test_dataset.target_cols, args.output_dir)
    
    # Plot training history if available
    if len(train_history) > 0 and len(val_history) > 0:
        plot_training_history(train_history, val_history, args.output_dir)
    
    print(f"\nResults and visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()

