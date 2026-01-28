"""
Quick script to visualize attention heatmaps from MESA Transformer
Run this from the project root directory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from visualize_attention_heatmaps import (
    load_model, visualize_sample_attention
)
from mesa_dataloader import create_mesa_dataloader
import torch

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "checkpoints_mesa/best_model.pth"
    preprocessed_dir = r"C:\mesa"
    csv_path = "mesa_final.csv"
    output_dir = "attention_visualizations"
    num_samples = 3
    channel_names = ["EEG1", "EEG2", "EEG3"]
    seq_len = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("MESA Transformer Attention Visualization")
    print("="*70)
    
    # Load model
    model = load_model(
        checkpoint_path,
        num_channels=3,
        time_steps=3840,
        seq_len=seq_len,
        num_classes=6,
        device=device
    )
    
    # Create dataloader
    print(f"\nLoading data from {preprocessed_dir}...")
    dataloader = create_mesa_dataloader(
        preprocessed_dir=preprocessed_dir,
        csv_path=csv_path,
        seq_len=seq_len,
        batch_size=1,
        shuffle=False,
        filter_unscored=True
    )
    
    # Visualize attention for first N samples
    print(f"\nVisualizing attention for {num_samples} samples...")
    for batch_idx, (features, labels) in enumerate(dataloader):
        if batch_idx >= num_samples:
            break
        
        print(f"\nProcessing sample {batch_idx + 1}/{num_samples}...")
        visualize_sample_attention(
            model, features, labels, channel_names,
            output_dir, sample_idx=batch_idx, seq_len=seq_len
        )
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to {output_dir}/")
    print(f"{'='*70}")

