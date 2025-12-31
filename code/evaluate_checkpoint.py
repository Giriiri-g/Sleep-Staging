"""
Evaluate Checkpoint Script
===========================
Load a trained checkpoint and generate comprehensive evaluation visualizations.
Can be used to evaluate existing checkpoints without retraining.
"""

import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

from comorbidity_classifier_dataset import create_dataloader
from comorbidity_classifier import ComorbidityClassifier
# Import evaluation functions from training script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_comorbidity_classifier import (
    evaluate, optimize_thresholds,
    plot_confusion_matrices, plot_roc_curves, plot_pr_curves,
    plot_class_performance
)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})
    
    # Create model with same architecture
    model = ComorbidityClassifier(
        num_sleep_stages=6,
        embedding_dim=args.get('embedding_dim', 32),
        rnn_hidden_dim=args.get('rnn_hidden_dim', 128),
        rnn_num_layers=args.get('rnn_num_layers', 2),
        rnn_type=args.get('rnn_type', 'LSTM'),
        num_features=args.get('num_features', 12),
        dropout=args.get('dropout', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate Comorbidity Classifier Checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (if None, uses latest in output_dir)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing checkpoints and data splits')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--optimize_thresholds', action='store_true',
                        help='Optimize thresholds on validation set')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                        choices=['f1', 'balanced_accuracy'],
                        help='Metric to optimize for threshold selection')
    parser.add_argument('--eval_splits', type=str, nargs='+', 
                        default=['test'],
                        choices=['train', 'val', 'test'],
                        help='Which splits to evaluate')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Try best_model.pt first, then latest checkpoint
        best_model_path = os.path.join(args.output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            checkpoint_files = glob.glob(os.path.join(args.output_dir, 'checkpoint_epoch_*.pt'))
            if checkpoint_files:
                checkpoint_path = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            else:
                raise FileNotFoundError(f"No checkpoints found in {args.output_dir}")
    
    # Load model
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    
    # Create dataloaders for requested splits
    splits_to_eval = {}
    for split_name in args.eval_splits:
        split_csv = os.path.join(args.output_dir, f'{split_name}_split.csv')
        if not os.path.exists(split_csv):
            print(f"Warning: {split_csv} not found, skipping {split_name}")
            continue
        
        # Load scaler from train split if available
        train_csv = os.path.join(args.output_dir, 'train_split.csv')
        scaler = None
        if os.path.exists(train_csv):
            _, train_dataset = create_dataloader(train_csv, batch_size=1, shuffle=False)
            scaler = train_dataset.get_scaler()
        
        loader, dataset = create_dataloader(
            split_csv,
            batch_size=args.batch_size,
            shuffle=False,
            scaler=scaler
        )
        splits_to_eval[split_name] = (loader, dataset)
    
    if not splits_to_eval:
        raise ValueError("No valid splits found to evaluate")
    
    # Optimize thresholds on validation set if requested
    optimal_thresholds = None
    if args.optimize_thresholds and 'val' in splits_to_eval:
        print("\nOptimizing thresholds on validation set...")
        val_loader, val_dataset = splits_to_eval['val']
        
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
                all_val_probs.append(probs.cpu().numpy())
                all_val_targets.append(targets.cpu().numpy())
        
        all_val_probs = np.vstack(all_val_probs)
        all_val_targets = np.vstack(all_val_targets)
        
        optimal_thresholds = optimize_thresholds(
            all_val_probs, all_val_targets, metric=args.threshold_metric
        )
        print(f"Optimal thresholds: {optimal_thresholds}")
    
    # Evaluate on each split
    criterion = nn.BCEWithLogitsLoss()
    all_results = {}
    
    for split_name, (loader, dataset) in splits_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name.upper()} set")
        print(f"{'='*60}")
        
        metrics = evaluate(
            model, loader, criterion, device,
            dataset.target_cols,
            thresholds=optimal_thresholds
        )
        
        all_results[split_name] = metrics
        
        # Print results
        print(f"\n{split_name.upper()} Results:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"\nPer-class metrics:")
        for i, name in enumerate(dataset.target_cols):
            print(f"  {name}:")
            print(f"    Precision: {metrics['precision'][i]:.4f}")
            print(f"    Recall: {metrics['recall'][i]:.4f}")
            print(f"    F1: {metrics['f1'][i]:.4f}")
            print(f"    ROC AUC: {metrics['roc_auc'][i]:.4f}")
            if optimal_thresholds is not None:
                print(f"    Threshold: {optimal_thresholds[i]:.4f}")
        
        # Generate visualizations for each split
        split_output_dir = os.path.join(args.output_dir, f'eval_{split_name}')
        os.makedirs(split_output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations for {split_name}...")
        plot_confusion_matrices(metrics, dataset.target_cols, split_output_dir)
        plot_roc_curves(metrics, dataset.target_cols, split_output_dir)
        plot_pr_curves(metrics, dataset.target_cols, split_output_dir)
        plot_class_performance(metrics, dataset.target_cols, split_output_dir)
        
        # Save results
        import json
        results_dict = {
            'split': split_name,
            'metrics': {
                'loss': float(metrics['loss']),
                'accuracy': float(metrics['accuracy']),
                'macro_precision': float(metrics['macro_precision']),
                'macro_recall': float(metrics['macro_recall']),
                'macro_f1': float(metrics['macro_f1']),
                'per_class': {
                    name: {
                        'precision': float(metrics['precision'][i]),
                        'recall': float(metrics['recall'][i]),
                        'f1': float(metrics['f1'][i]),
                        'roc_auc': float(metrics['roc_auc'][i])
                    }
                    for i, name in enumerate(dataset.target_cols)
                }
            },
            'optimal_thresholds': optimal_thresholds.tolist() if optimal_thresholds is not None else None,
            'checkpoint': checkpoint_path
        }
        
        with open(os.path.join(split_output_dir, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save predictions
        np.savez(
            os.path.join(split_output_dir, 'predictions.npz'),
            predictions=metrics['predictions'],
            probabilities=metrics['probabilities'],
            targets=metrics['targets']
        )
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

