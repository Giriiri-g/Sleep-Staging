"""
Compare All Models
=================
Compare neural network with traditional ML classifiers.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(output_dir):
    """Load results from different model directories"""
    results = {}
    
    # Neural network results
    nn_path = os.path.join(output_dir, 'test_results.json')
    if os.path.exists(nn_path):
        with open(nn_path, 'r') as f:
            nn_data = json.load(f)
            results['Neural Network'] = nn_data.get('test_metrics', {})
    
    # ML classifier results
    ml_dir = os.path.join(output_dir.replace('comorbidity', 'ml_classifiers'))
    if os.path.exists(ml_dir):
        for clf_name in ['svm', 'rf', 'xgb']:
            clf_path = os.path.join(ml_dir, clf_name, 'results.json')
            if os.path.exists(clf_path):
                with open(clf_path, 'r') as f:
                    ml_data = json.load(f)
                    results[clf_name.upper()] = ml_data.get('test_metrics', {})
    
    return results


def plot_comparison(results, output_dir):
    """Plot comparison of all models"""
    if not results:
        print("No results found to compare")
        return
    
    model_names = list(results.keys())
    target_cols = ['insomnia', 'restless leg', 'apnea']
    
    # Extract metrics
    macro_f1s = [results[m].get('macro_f1', 0) for m in model_names]
    macro_recalls = [results[m].get('macro_recall', 0) for m in model_names]
    macro_precisions = [results[m].get('macro_precision', 0) for m in model_names]
    mean_aucs = []
    for m in model_names:
        per_class = results[m].get('per_class', {})
        aucs = [per_class.get(col, {}).get('roc_auc', 0) for col in target_cols]
        mean_aucs.append(np.mean(aucs) if aucs else 0)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Macro F1
    axes[0, 0].bar(x, macro_f1s, width, label='Macro F1', alpha=0.8)
    axes[0, 0].set_xlabel('Model', fontsize=11)
    axes[0, 0].set_ylabel('Macro F1', fontsize=11)
    axes[0, 0].set_title('Macro F1 Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, max(macro_f1s) * 1.2 if max(macro_f1s) > 0 else 1])
    
    # Macro Recall
    axes[0, 1].bar(x, macro_recalls, width, label='Macro Recall', alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Model', fontsize=11)
    axes[0, 1].set_ylabel('Macro Recall', fontsize=11)
    axes[0, 1].set_title('Macro Recall Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, max(macro_recalls) * 1.2 if max(macro_recalls) > 0 else 1])
    
    # Mean ROC AUC
    axes[1, 0].bar(x, mean_aucs, width, label='Mean ROC AUC', alpha=0.8, color='green')
    axes[1, 0].set_xlabel('Model', fontsize=11)
    axes[1, 0].set_ylabel('Mean ROC AUC', fontsize=11)
    axes[1, 0].set_title('Mean ROC AUC Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, max(mean_aucs) * 1.2 if max(mean_aucs) > 0 else 1])
    
    # Per-class F1 comparison
    per_class_f1 = {col: [] for col in target_cols}
    for model_name in model_names:
        per_class = results[model_name].get('per_class', {})
        for col in target_cols:
            f1 = per_class.get(col, {}).get('f1', 0)
            per_class_f1[col].append(f1)
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    for i, col in enumerate(target_cols):
        axes[1, 1].bar(x_pos + i*width, per_class_f1[col], width, label=col, alpha=0.8)
    
    axes[1, 1].set_xlabel('Model', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title('Per-Class F1 Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos + width)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {os.path.join(output_dir, 'model_comparison.png')}")


def main():
    parser = argparse.ArgumentParser(description='Compare All Models')
    parser.add_argument('--nn_output_dir', type=str, 
                        default='checkpoints/comorbidity_aggressive',
                        help='Neural network output directory')
    parser.add_argument('--ml_output_dir', type=str,
                        default='checkpoints/ml_classifiers',
                        help='ML classifiers output directory')
    parser.add_argument('--comparison_dir', type=str,
                        default='checkpoints/comparison',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    os.makedirs(args.comparison_dir, exist_ok=True)
    
    # Load results
    results = {}
    
    # Neural network
    nn_path = os.path.join(args.nn_output_dir, 'test_results.json')
    if os.path.exists(nn_path):
        with open(nn_path, 'r') as f:
            nn_data = json.load(f)
            results['Neural Network'] = nn_data.get('test_metrics', {})
    
    # ML classifiers
    if os.path.exists(args.ml_output_dir):
        for clf_name in ['svm', 'rf', 'xgb']:
            clf_path = os.path.join(args.ml_output_dir, clf_name, 'results.json')
            if os.path.exists(clf_path):
                with open(clf_path, 'r') as f:
                    ml_data = json.load(f)
                    results[clf_name.upper()] = ml_data.get('test_metrics', {})
    
    if not results:
        print("No results found. Please train models first.")
        return
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for model_name, metrics in results.items():
        per_class = metrics.get('per_class', {})
        mean_auc = np.mean([
            per_class.get(col, {}).get('roc_auc', 0) 
            for col in ['insomnia', 'restless leg', 'apnea']
        ])
        
        comparison_data.append({
            'Model': model_name,
            'Macro F1': metrics.get('macro_f1', 0),
            'Macro Recall': metrics.get('macro_recall', 0),
            'Macro Precision': metrics.get('macro_precision', 0),
            'Mean ROC AUC': mean_auc
        })
        
        print(f"\n{model_name}:")
        print(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"  Macro Recall: {metrics.get('macro_recall', 0):.4f}")
        print(f"  Macro Precision: {metrics.get('macro_precision', 0):.4f}")
        print(f"  Mean ROC AUC: {mean_auc:.4f}")
        
        print(f"  Per-class F1:")
        for col in ['insomnia', 'restless leg', 'apnea']:
            f1 = per_class.get(col, {}).get('f1', 0)
            print(f"    {col}: {f1:.4f}")
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(os.path.join(args.comparison_dir, 'comparison.csv'), index=False)
    
    # Plot comparison
    plot_comparison(results, args.comparison_dir)
    
    print(f"\nComparison saved to {args.comparison_dir}")


if __name__ == '__main__':
    main()

