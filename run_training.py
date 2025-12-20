"""
Simple script to run MESA Transformer training
"""
import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

from train_mesa_transformer import train
import os

if __name__ == "__main__":
    csv_path = os.path.abspath("mesa_final.csv")
    print(f"Using CSV: {csv_path}")
    print("Starting training...")
    
    # Optimized for ~4 hour training time
    # Calculation: 4 hours = 240 minutes
    # If batches take 2.5 min average: 240 / 2.5 = 96 batches max
    # With 8 epochs: 96 / 8 = 12 batches per epoch max
    # With batch_size=512: 12 * 512 = ~6,144 train samples max
    # Using max_samples=5000: Train ~3.5K -> ~7 batches/epoch -> ~17.5 min/epoch -> ~2.3 hours total
    
    train(
        preprocessed_dir=r"C:\mesa",
        csv_path=csv_path,
        num_epochs=8,  # Reduced epochs
        batch_size=512,  # Much larger batch size
        learning_rate=3e-4,
        seq_len=20,
        max_samples=5000,  # Limit to 5K samples to ensure <4 hours
        device=None,
        checkpoint_dir="checkpoints_mesa"
    )

