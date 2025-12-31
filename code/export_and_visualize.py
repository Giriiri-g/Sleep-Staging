"""
Export MESA Transformer model to ONNX and create evaluation visualizations
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.onnx
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
from sklearn.metrics import confusion_matrix, classification_report

from mesa_transformer import MESATransformer


def export_to_onnx(model_path: str, output_path: str, seq_len: int = 20, 
                   num_channels: int = 3, time_steps: int = 3840, num_classes: int = 6,
                   device: str = "cpu"):
    """
    Export trained model to ONNX format.
    
    Args:
        model_path: Path to the saved model checkpoint
        output_path: Path to save the ONNX model
        seq_len: Sequence length (number of epochs)
        num_channels: Number of input channels
        time_steps: Number of time steps per epoch
        num_classes: Number of output classes
        device: Device to load model on
    """
    print(f"Loading model from {model_path}...")
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = MESATransformer(
        num_channels=num_channels,
        time_steps=time_steps,
        seq_len=seq_len,
        d_model=256,
        num_classes=num_classes,
        dropout=0.0,  # Disable dropout for inference
        return_attention=False
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, seq_len, num_channels, time_steps).to(device)
    
    # Export to ONNX
    print(f"Exporting to ONNX format...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,  # Increased to support scaled_dot_product_attention
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    return model


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, 
                         title: str = "Confusion Matrix", normalize: bool = False):
    """Plot confusion matrix"""
    if normalize:
        # Avoid division by zero for classes with no samples
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # Set zero sums to 1 to avoid division by zero
        cm = cm.astype('float') / row_sums
        fmt = '.2f'
        title += " (Normalized)"
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    
    if USE_SEABORN:
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    else:
        # Use matplotlib's imshow if seaborn not available
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, label='Count' if not normalize else 'Proportion')
        plt.xticks(range(len(class_names)), class_names)
        plt.yticks(range(len(class_names)), class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_class_metrics(metrics: Dict, class_names: list, output_path: str, stage: str = ""):
    """Plot per-class metrics (precision, recall, F1)"""
    precisions = [metrics['per_class'][name]['precision'] for name in class_names]
    recalls = [metrics['per_class'][name]['recall'] for name in class_names]
    f1_scores = [metrics['per_class'][name]['f1'] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Metrics{stage}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class metrics plot to {output_path}")


def plot_class_distribution(supports: list, class_names: list, output_path: str, stage: str = ""):
    """Plot class distribution (support counts)"""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(class_names, supports, color=colors, alpha=0.8, edgecolor='black')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Class Distribution{stage}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution plot to {output_path}")


def plot_summary_metrics(metrics: Dict, output_path: str, stage: str = ""):
    """Plot summary metrics (overall, macro, weighted)"""
    categories = ['Overall\nAccuracy', 'Macro\nPrecision', 'Macro\nRecall', 
                  'Macro\nF1', 'Weighted\nPrecision', 'Weighted\nRecall', 'Weighted\nF1']
    values = [
        metrics['overall_accuracy'],
        metrics['macro_precision'],
        metrics['macro_recall'],
        metrics['macro_f1'],
        metrics['weighted_precision'],
        metrics['weighted_recall'],
        metrics['weighted_f1']
    ]
    
    colors = ['#2ecc71', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Summary Metrics{stage}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add target line at 0.6
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Target (60%)', alpha=0.7)
    ax.legend()
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary metrics plot to {output_path}")


def create_visualizations(metrics: Dict, class_names: list, output_dir: str, stage: str = ""):
    """Create all visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage_suffix = f"_{stage.lower().replace(' ', '_')}" if stage else ""
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    plot_confusion_matrix(cm, class_names, 
                         output_dir / f"confusion_matrix{stage_suffix}.png", 
                         title=f"Confusion Matrix{stage}")
    
    # Normalized confusion matrix
    plot_confusion_matrix(cm, class_names, 
                         output_dir / f"confusion_matrix_normalized{stage_suffix}.png",
                         title=f"Confusion Matrix (Normalized){stage}", 
                         normalize=True)
    
    # Per-class metrics
    plot_class_metrics(metrics, class_names, 
                      output_dir / f"class_metrics{stage_suffix}.png", 
                      stage=stage)
    
    # Class distribution
    supports = [metrics['per_class'][name]['support'] for name in class_names]
    plot_class_distribution(supports, class_names, 
                           output_dir / f"class_distribution{stage_suffix}.png",
                           stage=stage)
    
    # Summary metrics
    plot_summary_metrics(metrics, 
                        output_dir / f"summary_metrics{stage_suffix}.png",
                        stage=stage)


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX and create visualizations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_mesa",
                       help="Directory containing model checkpoints")
    parser.add_argument("--model_file", type=str, default="best_model.pth",
                       help="Model checkpoint file name")
    parser.add_argument("--output_dir", type=str, default="model_exports",
                       help="Output directory for ONNX and visualizations")
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Sequence length")
    parser.add_argument("--num_channels", type=int, default=3,
                       help="Number of input channels")
    parser.add_argument("--time_steps", type=int, default=3840,
                       help="Time steps per epoch")
    parser.add_argument("--num_classes", type=int, default=6,
                       help="Number of classes")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_dir) / args.model_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    onnx_path = output_dir / "mesa_transformer.onnx"
    model = export_to_onnx(
        str(checkpoint_path),
        str(onnx_path),
        seq_len=args.seq_len,
        num_channels=args.num_channels,
        time_steps=args.time_steps,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Load metrics from checkpoint if available
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    class_names = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
    
    # If metrics are in checkpoint, use them; otherwise, note that evaluation is needed
    if 'test_metrics' in checkpoint:
        print("\nCreating visualizations from checkpoint metrics...")
        test_metrics = checkpoint['test_metrics']
        create_visualizations(test_metrics, class_names, output_dir, "Test Set")
        
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            create_visualizations(val_metrics, class_names, output_dir, "Validation Set")
    else:
        print("\nWarning: No metrics found in checkpoint. Run evaluation first to generate visualizations.")
        print("   You can create visualizations by running the training script evaluation,")
        print("   or by providing metrics directly to this script.")
    
    print(f"\nExport and visualization complete!")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Visualizations: {output_dir}")


if __name__ == "__main__":
    main()

