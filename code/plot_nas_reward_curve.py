"""
Generate reward curve plot for NAS (Neural Architecture Search)
Plots validation accuracy (reward) over epochs during architecture search
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_nas_results(json_path: str) -> dict:
    """Load NAS results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def plot_reward_curve(data: dict, output_path: str, show_efficiency: bool = False):
    """
    Plot reward curve (validation accuracy) over epochs
    
    Args:
        data: Dictionary containing NAS results with history
        output_path: Path to save the plot
        show_efficiency: If True, also plot efficiency metrics on secondary axis
    """
    history = data.get('history', {})
    val_acc = history.get('val_acc', [])
    train_acc = history.get('train_acc', [])
    best_val_acc = data.get('best_val_acc', None)
    
    if not val_acc:
        raise ValueError("No validation accuracy data found in history")
    
    epochs = list(range(1, len(val_acc) + 1))
    
    # Create figure
    if show_efficiency:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax1 = axes[0]
        ax2 = axes[1]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot validation accuracy (reward)
    ax1.plot(epochs, val_acc, 'o-', label='Validation Accuracy (Reward)', 
             linewidth=2, markersize=6, color='#2ecc71', markerfacecolor='#27ae60')
    
    # Plot training accuracy for reference
    if train_acc:
        ax1.plot(epochs, train_acc, 's--', label='Training Accuracy', 
                 linewidth=1.5, markersize=4, color='#3498db', alpha=0.7)
    
    # Mark best validation accuracy
    if best_val_acc is not None:
        best_epoch = epochs[np.argmax(val_acc)]
        ax1.axhline(y=best_val_acc, color='r', linestyle=':', linewidth=2, 
                   label=f'Best Val Acc: {best_val_acc:.2f}%', alpha=0.8)
        ax1.scatter([best_epoch], [best_val_acc], color='red', s=150, 
                   zorder=5, marker='*', edgecolors='darkred', linewidths=1.5,
                   label=f'Best at Epoch {best_epoch}')
    
    # Formatting
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('NAS Reward Curve: Validation Accuracy During Architecture Search', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, max(epochs) + 1])
    
    # Set y-axis to show reasonable range
    y_min = min(min(val_acc), min(train_acc) if train_acc else min(val_acc)) * 0.98
    y_max = max(max(val_acc), max(train_acc) if train_acc else max(val_acc)) * 1.02
    ax1.set_ylim([y_min, y_max])
    
    # Add text annotation with statistics
    stats_text = f"Final Val Acc: {val_acc[-1]:.2f}%\n"
    if best_val_acc is not None:
        stats_text += f"Best Val Acc: {best_val_acc:.2f}%\n"
    stats_text += f"Epochs: {len(epochs)}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot efficiency metrics on secondary axis if requested
    if show_efficiency and 'efficiency' in history:
        efficiency = history['efficiency']
        flops = [e.get('flops', 0) for e in efficiency]
        macs = [e.get('macs', 0) for e in efficiency]
        
        # Convert to millions for readability
        flops_m = [f / 1e6 for f in flops]
        macs_m = [m / 1e6 for m in macs]
        
        ax2.plot(epochs, flops_m, 'o-', label='FLOPs (M)', 
                linewidth=2, markersize=5, color='#e74c3c')
        ax2.plot(epochs, macs_m, 's--', label='MACs (M)', 
                linewidth=1.5, markersize=4, color='#9b59b6', alpha=0.7)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Computational Cost (Millions)', fontsize=12, fontweight='bold')
        ax2.set_title('Architecture Efficiency Evolution', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim([0, max(epochs) + 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reward curve plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot NAS reward curve from results JSON")
    parser.add_argument("--input_json", type=str, 
                       default="efficient_darts_results/efficient_darts_results.json",
                       help="Path to NAS results JSON file")
    parser.add_argument("--output_path", type=str, 
                       default="plots/nas_reward_curve.png",
                       help="Path to save the plot")
    parser.add_argument("--show_efficiency", action="store_true",
                       help="Also plot efficiency metrics (FLOPs/MACs)")
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")
    
    print(f"Loading NAS results from: {input_path}")
    data = load_nas_results(str(input_path))
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Plot reward curve
    print(f"Generating reward curve plot...")
    plot_reward_curve(data, str(output_path), show_efficiency=args.show_efficiency)
    print(f"Plot saved successfully!")


if __name__ == "__main__":
    main()

