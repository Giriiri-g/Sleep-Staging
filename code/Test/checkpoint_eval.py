"""
Sleep Staging Model Evaluator
==============================
Loads a saved checkpoint and evaluates performance on validation/test data.
Shows detailed metrics, confusion matrix, and per-class performance.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import from your original training script
# Make sure the training script is in the same directory or in PYTHONPATH
# If it's named 'sleep_staging_transformer.py', adjust the import accordingly
try:
    from sleep_staging_transformer import (
        HierarchicalTransformerModel,
        SleepEDFDataset,
        Colors,
        print_colored,
        print_debug
    )
except ImportError:
    print("Error: Could not import from training script.")
    print("Make sure 'sleep_staging_transformer.py' is in the same directory.")
    sys.exit(1)

warnings.filterwarnings('ignore')


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """
    Load model from checkpoint file.
    
    Returns:
        model, metadata, checkpoint_info
    """
    print_debug(f"Loading checkpoint from: {checkpoint_path}", "INFO")
    
    if not os.path.exists(checkpoint_path):
        print_debug(f"Checkpoint not found: {checkpoint_path}", "ERROR")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    metadata = checkpoint.get('metadata', {})
    
    # Print checkpoint info
    print_colored("=" * 80, Colors.BOLD)
    print_colored("Checkpoint Information", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.BOLD)
    print_debug(f"Epoch: {checkpoint['epoch']}", "INFO")
    print_debug(f"Training Loss: {checkpoint['train_loss']:.4f}", "INFO")
    print_debug(f"Validation Loss: {checkpoint['val_loss']:.4f}", "INFO")
    print_debug(f"Validation Accuracy: {checkpoint['val_acc']:.4f}", "INFO")
    if 'best_val_acc' in checkpoint:
        print_debug(f"Best Validation Accuracy: {checkpoint['best_val_acc']:.4f} (Epoch {checkpoint.get('best_epoch', 'N/A')})", "INFO")
    print_colored("=" * 80, Colors.BOLD)
    
    # Create model with saved hyperparameters
    if not metadata:
        print_debug("No metadata found in checkpoint. Using default parameters.", "WARNING")
        print_debug("This may cause errors if the model architecture doesn't match.", "WARNING")
        # You'll need to provide these manually or use defaults
        input_dim = 903  # Default for 7 channels with spectrogram
        hidden_dim = 256
        num_heads = 8
        num_encoder_layers_local = 4
        num_encoder_layers_global = 2
        num_classes = 6
        segment_size = 10
    else:
        input_dim = metadata['input_dim']
        hidden_dim = metadata['hidden_dim']
        num_heads = metadata['num_heads']
        num_encoder_layers_local = metadata['num_encoder_layers_local']
        num_encoder_layers_global = metadata['num_encoder_layers_global']
        num_classes = metadata['num_classes']
        segment_size = metadata.get('segment_size', 10)
    
    print_colored("\nModel Architecture", Colors.HEADER, bold=True)
    print_debug(f"Input Dimension: {input_dim}", "INFO")
    print_debug(f"Hidden Dimension: {hidden_dim}", "INFO")
    print_debug(f"Attention Heads: {num_heads}", "INFO")
    print_debug(f"Local Encoder Layers: {num_encoder_layers_local}", "INFO")
    print_debug(f"Global Encoder Layers: {num_encoder_layers_global}", "INFO")
    print_debug(f"Number of Classes: {num_classes}", "INFO")
    print_debug(f"Segment Size: {segment_size}", "INFO")
    
    # Create model
    model = HierarchicalTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers_local=num_encoder_layers_local,
        num_encoder_layers_global=num_encoder_layers_global,
        num_classes=num_classes,
        dropout=0.1
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print_debug(f"Model Parameters: {num_params:,}", "INFO")
    print_colored("=" * 80, Colors.BOLD)
    
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'val_acc': checkpoint['val_acc'],
        'best_val_acc': checkpoint.get('best_val_acc', checkpoint['val_acc']),
        'segment_size': segment_size
    }
    
    return model, metadata, checkpoint_info


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list = None
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    print_debug("Running evaluation...", "INFO")
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(features)
            
            # Reshape
            batch_size, num_segments, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            # Collect predictions
            predictions = torch.argmax(logits_flat, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels_flat.cpu().numpy())
            all_logits.append(logits_flat.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print_debug(f"Processed {batch_idx + 1}/{len(dataloader)} batches", "INFO")
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.vstack(all_logits)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    if class_names is None:
        class_names = ['Wake', 'REM', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'class_names': class_names
    }
    
    return results


def print_results(results: Dict):
    """Print evaluation results in a nice format"""
    print_colored("\n" + "=" * 80, Colors.BOLD)
    print_colored("Evaluation Results", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.BOLD)
    
    print_colored(f"\nOverall Accuracy: {results['accuracy']:.4f}", Colors.OKGREEN, bold=True)
    print_colored(f"Macro Precision: {results['macro_precision']:.4f}", Colors.OKGREEN)
    print_colored(f"Macro Recall: {results['macro_recall']:.4f}", Colors.OKGREEN)
    print_colored(f"Macro F1-Score: {results['macro_f1']:.4f}", Colors.OKGREEN)
    
    print_colored("\nPer-Class Metrics:", Colors.HEADER, bold=True)
    print_colored("-" * 80, Colors.GRAY)
    
    class_names = results['class_names']
    for i, name in enumerate(class_names):
        if i < len(results['precision']):
            print_colored(
                f"{name:12s} | Precision: {results['precision'][i]:.4f} | "
                f"Recall: {results['recall'][i]:.4f} | "
                f"F1: {results['f1'][i]:.4f} | "
                f"Support: {int(results['support'][i])}",
                Colors.OKCYAN
            )
    
    print_colored("\nConfusion Matrix:", Colors.HEADER, bold=True)
    print_colored("-" * 80, Colors.GRAY)
    
    cm = results['confusion_matrix']
    
    # Print header
    header = "True\\Pred  |"
    for name in class_names:
        header += f" {name[:8]:>8s} |"
    print_colored(header, Colors.BOLD)
    print_colored("-" * len(header), Colors.GRAY)
    
    # Print matrix
    for i, name in enumerate(class_names):
        if i < cm.shape[0]:
            row = f"{name:12s} |"
            for j in range(cm.shape[1]):
                row += f" {cm[i, j]:8d} |"
            print(row)
    
    print_colored("=" * 80, Colors.BOLD)


def plot_confusion_matrix(results: Dict, save_path: str = "confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    # Limit to actual classes present
    num_classes = min(len(class_names), cm.shape[0])
    cm = cm[:num_classes, :num_classes]
    class_names = class_names[:num_classes]
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_debug(f"Confusion matrix saved to: {save_path}", "SUCCESS")
    plt.close()


def plot_class_performance(results: Dict, save_path: str = "class_performance.png"):
    """Plot per-class performance metrics"""
    class_names = results['class_names']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print_debug(f"Class performance plot saved to: {save_path}", "SUCCESS")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Sleep Staging Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/latest_checkpoint.pt",
        help="Path to checkpoint file (default: checkpoints/latest_checkpoint.pt)"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=r"F:\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette",
        help="Path to sleep-cassette folder"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--use_best",
        action="store_true",
        help="Use best_model.pt instead of latest_checkpoint.pt"
    )
    
    args = parser.parse_args()
    
    # Override checkpoint path if using best model
    if args.use_best:
        checkpoint_dir = Path(args.checkpoint).parent
        args.checkpoint = str(checkpoint_dir / "best_model.pt")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_debug(f"Using device: {device}", "INFO")
    
    # Load model
    model, metadata, checkpoint_info = load_model_from_checkpoint(args.checkpoint, device)
    segment_size = checkpoint_info.get('segment_size', 10)
    
    # Load dataset
    print_colored("\nLoading Dataset", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.BOLD)
    
    full_dataset = SleepEDFDataset(
        folder_path=args.data_folder,
        segment_size=segment_size,
        use_spectrogram=True,
        filter_unscored=True
    )
    
    # Use same split as training (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print_debug(f"Validation samples: {len(val_dataset)}", "INFO")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Evaluate
    print_colored("\nEvaluating Model", Colors.HEADER, bold=True)
    print_colored("=" * 80, Colors.BOLD)
    
    results = evaluate_model(model, val_loader, device)
    
    # Print results
    print_results(results)
    
    # Save results to JSON
    results_json = {
        'checkpoint_info': checkpoint_info,
        'accuracy': float(results['accuracy']),
        'macro_precision': float(results['macro_precision']),
        'macro_recall': float(results['macro_recall']),
        'macro_f1': float(results['macro_f1']),
        'per_class_metrics': {
            name: {
                'precision': float(results['precision'][i]),
                'recall': float(results['recall'][i]),
                'f1': float(results['f1'][i]),
                'support': int(results['support'][i])
            }
            for i, name in enumerate(results['class_names'])
            if i < len(results['precision'])
        },
        'confusion_matrix': results['confusion_matrix'].tolist()
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print_debug(f"Results saved to: {results_path}", "SUCCESS")
    
    # Plot confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(results, str(cm_path))
    
    # Plot class performance
    perf_path = output_dir / "class_performance.png"
    plot_class_performance(results, str(perf_path))
    
    print_colored("\n" + "=" * 80, Colors.BOLD)
    print_colored("Evaluation Complete!", Colors.OKGREEN, bold=True)
    print_colored("=" * 80, Colors.BOLD)
    print_debug(f"All results saved to: {output_dir}", "SUCCESS")


if __name__ == "__main__":
    main()