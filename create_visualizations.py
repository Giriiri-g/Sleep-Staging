"""
Quick script to export model and create visualizations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from export_and_visualize import export_to_onnx, create_visualizations, compute_class_metrics
import torch
import numpy as np

if __name__ == "__main__":
    # Configuration
    checkpoint_dir = "checkpoints_mesa"
    output_dir = "model_exports"
    model_file = "best_model.pth"
    
    checkpoint_path = Path(checkpoint_dir) / model_file
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run training first to generate the model.")
        sys.exit(1)
    
    # Export to ONNX
    print("="*70)
    print("Exporting Model to ONNX")
    print("="*70)
    model = export_to_onnx(
        str(checkpoint_path),
        str(Path(output_dir) / "mesa_transformer.onnx"),
        seq_len=20,
        num_channels=3,
        time_steps=3840,
        num_classes=6,
        device="cpu"
    )
    
    # Load checkpoint and create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
    
    # Try to get metrics from checkpoint
    if 'test_metrics' in checkpoint:
        print("\nUsing metrics from checkpoint...")
        test_metrics = checkpoint['test_metrics']
        create_visualizations(test_metrics, class_names, output_dir, "Test Set")
        
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            create_visualizations(val_metrics, class_names, output_dir, "Validation Set")
    else:
        print("\n⚠ No metrics in checkpoint. Please re-run training to save metrics,")
        print("   or run evaluation separately to generate metrics.")
    
    print(f"\n✓ Complete! Check {output_dir} for ONNX model and visualizations.")

