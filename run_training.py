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
    # With batch_size=256 and max_samples=50000:
    # - Train: ~35K samples -> ~137 batches per epoch
    # - At 2-3 min per batch: ~4.5-7 hours per epoch
    # - With 10 epochs: ~45-70 hours total (still too long)
    # 
    # Better: batch_size=512, max_samples=30000, num_epochs=10
    # - Train: ~21K samples -> ~41 batches per epoch  
    # - At 2-3 min per batch: ~1.5-2 hours per epoch
    # - With 10 epochs: ~15-20 hours total (still too long)
    #
    # Best: batch_size=512, max_samples=15000, num_epochs=15
    # - Train: ~10.5K samples -> ~21 batches per epoch
    # - At 2-3 min per batch: ~42-63 min per epoch
    # - With 15 epochs: ~10.5-15.75 hours total
    #
    # Target 4 hours: Need ~20 batches per epoch * 15 epochs = 300 total batches
    # At 2-3 min: 600-900 min = 10-15 hours still...
    #
    # Let's try: batch_size=512, max_samples=10000, num_epochs=10
    # - Train: ~7K samples -> ~14 batches per epoch
    # - At 2-3 min per batch: ~28-42 min per epoch
    # - With 10 epochs: ~4.7-7 hours (close to target!)
    
    train(
        preprocessed_dir=r"C:\mesa",
        csv_path=csv_path,
        num_epochs=10,  # Reduced epochs
        batch_size=512,  # Much larger batch size
        learning_rate=3e-4,
        seq_len=20,
        max_samples=10000,  # Limit to 10K samples
        device=None,
        checkpoint_dir="checkpoints_mesa"
    )

