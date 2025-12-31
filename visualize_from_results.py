"""
Create visualizations from the terminal results you've already obtained.
This script recreates the metrics dictionary from your terminal output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from export_and_visualize import create_visualizations, export_to_onnx
import numpy as np
import torch

# Metrics from your terminal output (Test Set)
test_metrics_dict = {
    'overall_accuracy': 0.4730,
    'macro_precision': 0.3908,
    'macro_recall': 0.4558,
    'macro_f1': 0.3514,
    'weighted_precision': 0.6479,
    'weighted_recall': 0.4730,
    'weighted_f1': 0.4828,
    'per_class': {
        'W': {'precision': 0.8499, 'recall': 0.6703, 'f1': 0.7495, 'support': 5615},
        'N1': {'precision': 0.2030, 'recall': 0.5027, 'f1': 0.2893, 'support': 1273},
        'N2': {'precision': 0.7081, 'recall': 0.1911, 'f1': 0.3009, 'support': 5485},
        'N3': {'precision': 0.2370, 'recall': 0.9202, 'f1': 0.3770, 'support': 978},
        'N4': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0},
        'REM': {'precision': 0.3469, 'recall': 0.4506, 'f1': 0.3920, 'support': 1649}
    },
    'confusion_matrix': np.array([
        [3764, 1121, 115, 124, 0, 491],
        [155, 640, 140, 110, 0, 228],
        [281, 913, 1048, 2600, 0, 643],
        [9, 6, 26, 900, 0, 37],
        [0, 0, 0, 0, 0, 0],
        [220, 472, 151, 63, 0, 743]
    ])
}

# Metrics from Validation Set
val_metrics_dict = {
    'overall_accuracy': 0.4913,
    'macro_precision': 0.3924,
    'macro_recall': 0.4541,
    'macro_f1': 0.3541,
    'weighted_precision': 0.6535,
    'weighted_recall': 0.4913,
    'weighted_f1': 0.4997,
    'per_class': {
        'W': {'precision': 0.8369, 'recall': 0.7063, 'f1': 0.7660, 'support': 5876},
        'N1': {'precision': 0.2049, 'recall': 0.4902, 'f1': 0.2890, 'support': 1275},
        'N2': {'precision': 0.7391, 'recall': 0.1995, 'f1': 0.3141, 'support': 5169},
        'N3': {'precision': 0.2474, 'recall': 0.9100, 'f1': 0.3890, 'support': 900},
        'N4': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0},
        'REM': {'precision': 0.3262, 'recall': 0.4185, 'f1': 0.3666, 'support': 1780}
    },
    'confusion_matrix': np.array([
        [4150, 878, 97, 108, 0, 643],
        [187, 625, 126, 102, 0, 235],
        [404, 867, 1031, 2229, 0, 638],
        [21, 10, 27, 819, 0, 23],
        [0, 0, 0, 0, 0, 0],
        [197, 671, 114, 53, 0, 745]
    ])
}

if __name__ == "__main__":
    output_dir = "model_exports"
    class_names = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
    
    # Export model to ONNX if checkpoint exists
    checkpoint_path = Path("checkpoints_mesa/best_model.pth")
    if checkpoint_path.exists():
        print("="*70)
        print("Exporting Model to ONNX")
        print("="*70)
        try:
            export_to_onnx(
                str(checkpoint_path),
                str(Path(output_dir) / "mesa_transformer.onnx"),
                seq_len=20,
                num_channels=3,
                time_steps=3840,
                num_classes=6,
                device="cpu"
            )
        except Exception as e:
            print(f"Warning: Could not export to ONNX: {e}")
            print("   Continuing with visualizations only...")
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("   Creating visualizations from provided metrics...")
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    create_visualizations(test_metrics_dict, class_names, output_dir, "Test Set")
    create_visualizations(val_metrics_dict, class_names, output_dir, "Validation Set")
    
    print(f"\nComplete! Visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - confusion_matrix_test_set.png")
    print("  - confusion_matrix_normalized_test_set.png")
    print("  - class_metrics_test_set.png")
    print("  - class_distribution_test_set.png")
    print("  - summary_metrics_test_set.png")
    print("  - (same for validation_set)")
    if checkpoint_path.exists():
        print("  - mesa_transformer.onnx")

