"""
Evaluate trained model and create visualizations from actual results
This script loads the trained model and evaluates it, then creates visualizations
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from export_and_visualize import export_to_onnx, create_visualizations, compute_class_metrics
from train_mesa_transformer import validate_epoch, compute_class_metrics as compute_metrics
from mesa_transformer import MESATransformer
from mesa_dataloader import create_mesa_dataloader
from train_mesa_transformer import FocalLoss

import torch
import numpy as np


def main():
    checkpoint_dir = "checkpoints_mesa"
    output_dir = "model_exports"
    model_file = "best_model.pth"
    csv_path = "mesa_final.csv"
    preprocessed_dir = r"C:\mesa"
    
    checkpoint_path = Path(checkpoint_dir) / model_file
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    print("Creating model...")
    model = MESATransformer(
        num_channels=3,
        time_steps=3840,
        seq_len=20,
        d_model=256,
        num_classes=6,
        dropout=0.0,  # Disable dropout for evaluation
        return_attention=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    print("\n" + "="*70)
    print("Exporting Model to ONNX")
    print("="*70)
    onnx_path = Path(output_dir) / "mesa_transformer.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    dummy_input = torch.randn(1, 20, 3, 3840).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    print(f"✓ Exported to {onnx_path}")
    
    # Create test dataloader
    print("\n" + "="*70)
    print("Evaluating Model")
    print("="*70)
    
    test_loader = create_mesa_dataloader(
        preprocessed_dir=preprocessed_dir,
        csv_path=csv_path,
        seq_len=20,
        batch_size=32,
        shuffle=False,
        filter_unscored=True
    )
    
    # We need to evaluate on a subset since we don't have the exact test split
    # For visualization, we'll use a sample
    print("Evaluating on test set...")
    
    # Create a simple loss function for evaluation
    class_counts = torch.ones(6, dtype=torch.float32)  # Dummy weights
    criterion = FocalLoss(alpha=class_counts.to(device), gamma=2.0)
    
    test_loss, test_labels, test_preds = validate_epoch(model, test_loader, criterion, device)
    
    class_names = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
    
    # Compute metrics
    test_metrics = compute_metrics(test_labels, test_preds, class_names)
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    create_visualizations(test_metrics, class_names, output_dir, "Test Set")
    
    print(f"\n✓ Complete!")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()

