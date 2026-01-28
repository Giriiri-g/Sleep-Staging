"""
Plot Training Curves for Comorbidity Classifier
================================================
Plot loss curves and other metrics from training history with big fonts.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set big fonts globally
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.titlesize': 22
})


def parse_array_string(s):
    """Parse numpy array string representation"""
    if isinstance(s, str):
        # Remove brackets and split
        s = s.strip('[]')
        return np.array([float(x) for x in s.split()])
    return s


def load_training_history(json_path):
    """Load training history from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    train_history = data['train_history']
    val_history = data['val_history']
    
    # Parse string arrays if needed
    for h in train_history + val_history:
        if 'precision' in h and isinstance(h['precision'], str):
            h['precision'] = parse_array_string(h['precision'])
        if 'recall' in h and isinstance(h['recall'], str):
            h['recall'] = parse_array_string(h['recall'])
        if 'f1' in h and isinstance(h['f1'], str):
            h['f1'] = parse_array_string(h['f1'])
        if 'roc_auc' in h:
            if isinstance(h['roc_auc'], list):
                h['roc_auc'] = np.array(h['roc_auc'])
    
    return train_history, val_history


def plot_training_curves(train_history, val_history, output_path):
    """Plot training curves with big fonts, no captions, only axis titles and legends"""
    
    epochs = range(1, len(train_history) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h['loss'] for h in train_history], 'b-', label='Train', linewidth=3)
    ax.plot(epochs, [h['loss'] for h in val_history], 'r-', label='Val', linewidth=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [h['accuracy'] for h in train_history], 'b-', label='Train', linewidth=3)
    ax.plot(epochs, [h['accuracy'] for h in val_history], 'r-', label='Val', linewidth=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Macro F1
    ax = axes[0, 2]
    ax.plot(epochs, [h['macro_f1'] for h in train_history], 'b-', label='Train', linewidth=3)
    ax.plot(epochs, [h['macro_f1'] for h in val_history], 'r-', label='Val', linewidth=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Macro F1', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Macro Precision
    ax = axes[1, 0]
    ax.plot(epochs, [h['macro_precision'] for h in train_history], 'b-', label='Train', linewidth=3)
    ax.plot(epochs, [h['macro_precision'] for h in val_history], 'r-', label='Val', linewidth=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Macro Precision', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Macro Recall
    ax = axes[1, 1]
    ax.plot(epochs, [h['macro_recall'] for h in train_history], 'b-', label='Train', linewidth=3)
    ax.plot(epochs, [h['macro_recall'] for h in val_history], 'r-', label='Val', linewidth=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Macro Recall', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Mean ROC AUC
    train_aucs = [np.mean(h['roc_auc']) if isinstance(h['roc_auc'], (list, np.ndarray)) else h.get('roc_auc', 0) for h in train_history]
    val_aucs = [np.mean(h['roc_auc']) if isinstance(h['roc_auc'], (list, np.ndarray)) else h.get('roc_auc', 0) for h in val_history]
    ax = axes[1, 2]
    ax.plot(epochs, train_aucs, 'b-', label='Train', linewidth=3)
    ax.plot(epochs, val_aucs, 'r-', label='Val', linewidth=3)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Mean ROC AUC', fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_per_class_metrics(train_history, val_history, output_path, class_names=None):
    """Plot per-class F1, Precision, Recall, and ROC AUC curves"""
    
    if class_names is None:
        class_names = ['Insomnia', 'Restless Leg', 'Apnea']
    
    epochs = range(1, len(train_history) + 1)
    n_classes = len(class_names)
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Per-class F1
    ax = axes[0, 0]
    colors = ['b', 'g', 'orange']
    linestyles = ['-', '--']
    for i in range(n_classes):
        train_f1 = [h['f1'][i] if isinstance(h['f1'], (list, np.ndarray)) else 0 for h in train_history]
        val_f1 = [h['f1'][i] if isinstance(h['f1'], (list, np.ndarray)) else 0 for h in val_history]
        ax.plot(epochs, train_f1, color=colors[i], linestyle='-', label=f'Train {class_names[i]}', linewidth=2.5, alpha=0.7)
        ax.plot(epochs, val_f1, color=colors[i], linestyle='--', label=f'Val {class_names[i]}', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('F1 Score', fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Per-class Precision
    ax = axes[0, 1]
    for i in range(n_classes):
        train_prec = [h['precision'][i] if isinstance(h['precision'], (list, np.ndarray)) else 0 for h in train_history]
        val_prec = [h['precision'][i] if isinstance(h['precision'], (list, np.ndarray)) else 0 for h in val_history]
        ax.plot(epochs, train_prec, color=colors[i], linestyle='-', label=f'Train {class_names[i]}', linewidth=2.5, alpha=0.7)
        ax.plot(epochs, val_prec, color=colors[i], linestyle='--', label=f'Val {class_names[i]}', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Per-class Recall
    ax = axes[1, 0]
    for i in range(n_classes):
        train_rec = [h['recall'][i] if isinstance(h['recall'], (list, np.ndarray)) else 0 for h in train_history]
        val_rec = [h['recall'][i] if isinstance(h['recall'], (list, np.ndarray)) else 0 for h in val_history]
        ax.plot(epochs, train_rec, color=colors[i], linestyle='-', label=f'Train {class_names[i]}', linewidth=2.5, alpha=0.7)
        ax.plot(epochs, val_rec, color=colors[i], linestyle='--', label=f'Val {class_names[i]}', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Recall', fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Per-class ROC AUC
    ax = axes[1, 1]
    for i in range(n_classes):
        train_auc = [h['roc_auc'][i] if isinstance(h['roc_auc'], (list, np.ndarray)) and len(h['roc_auc']) > i else 0 for h in train_history]
        val_auc = [h['roc_auc'][i] if isinstance(h['roc_auc'], (list, np.ndarray)) and len(h['roc_auc']) > i else 0 for h in val_history]
        ax.plot(epochs, train_auc, color=colors[i], linestyle='-', label=f'Train {class_names[i]}', linewidth=2.5, alpha=0.7)
        ax.plot(epochs, val_auc, color=colors[i], linestyle='--', label=f'Val {class_names[i]}', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('ROC AUC', fontsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot Comorbidity Classifier Training Curves')
    parser.add_argument('--history_json', type=str, 
                        default='checkpoints/comorbidity_aggressive/training_history.json',
                        help='Path to training_history.json file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as history_json directory)')
    parser.add_argument('--class_names', type=str, nargs='+', 
                        default=['Insomnia', 'Restless Leg', 'Apnea'],
                        help='Class names for per-class plots')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.history_json)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training history
    print(f"Loading training history from {args.history_json}")
    train_history, val_history = load_training_history(args.history_json)
    print(f"Loaded {len(train_history)} epochs of training history")
    
    # Plot main training curves
    output_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(train_history, val_history, output_path)
    
    # Plot per-class metrics
    per_class_path = os.path.join(args.output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(train_history, val_history, per_class_path, args.class_names)
    
    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()

