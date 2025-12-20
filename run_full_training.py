"""
Run full training for MESA Transformer with class-wise metrics
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
    print("="*70)
    print("Starting FULL training for MESA Transformer")
    print("="*70)
    print(f"Preprocessed data: C:\\mesa")
    print(f"CSV file: {csv_path}")
    print(f"Training parameters (optimized for ~4 hour training):")
    print(f"  - Epochs: 8")
    print(f"  - Batch size: 512 (increased for efficiency)")
    print(f"  - Max samples: 5,000 (subset for speed)")
    print(f"  - Learning rate: 3e-4")
    print(f"  - Sequence length: 20 epochs")
    print(f"  - Expected: ~7 batches/epoch * 2.5 min = ~17.5 min/epoch")
    print(f"  - Total: ~2.3 hours for 8 epochs")
    print("="*70)
    print()
    
    train(
        preprocessed_dir=r"C:\mesa",
        csv_path=csv_path,
        num_epochs=8,  # Reduced for 4-hour target
        batch_size=512,  # Much larger batch size
        learning_rate=3e-4,
        seq_len=20,
        train_split=0.7,
        val_split=0.15,
        max_samples=5000,  # Limit dataset size to ensure <4 hours
        device=None,  # Auto-detect
        checkpoint_dir="checkpoints_mesa"
    )

