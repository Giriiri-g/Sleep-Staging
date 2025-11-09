"""Run NAS with a smaller subset for testing"""
import sys
import os
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import LazyPSGDataset
from torch.utils.data import DataLoader, Subset, random_split
from nas_search_space import Network, DARTS_OPS
from darts import DARTSTrainer
import torch.nn as nn

def run_darts_small():
    """Run DARTS with a smaller dataset subset"""
    print("=" * 80)
    print("DARTS: Differentiable Architecture Search (Small Test)")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    data_path = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette"
    print(f"Loading dataset from: {data_path}")
    
    dataset = LazyPSGDataset(folder_path=data_path, window_size=30)
    print(f"Total dataset size: {len(dataset)}")
    
    # Use a smaller subset for faster testing
    # Take first 10000 samples for training
    subset_size = min(10000, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    dataset_subset = Subset(dataset, indices)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset_subset))
    val_size = len(dataset_subset) - train_size
    train_dataset, val_dataset = random_split(
        dataset_subset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    # Create supernet with smaller architecture
    print("\nCreating supernet...")
    model = Network(
        input_channels=7,
        input_length=3000,
        num_classes=7,
        init_channels=32,  # Smaller for faster training
        num_cells=4,  # Fewer cells
        num_nodes=3,  # Fewer nodes
        search_space=DARTS_OPS
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Supernet parameters: {num_params:,}")
    
    # Create DARTS trainer
    print("\nCreating DARTS trainer...")
    trainer = DARTSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        w_lr=0.025,
        w_momentum=0.9,
        w_weight_decay=3e-4,
        alpha_lr=3e-4,
        alpha_weight_decay=1e-3,
        unrolled=False
    )
    
    # Train for a few epochs
    print("\nStarting DARTS training...")
    print("=" * 80)
    results = trainer.train(num_epochs=5, print_freq=1)
    
    # Get final architecture
    arch = trainer.get_architecture()
    print("\n" + "=" * 80)
    print("Final Architecture:")
    print("=" * 80)
    for cell_idx, cell_arch in arch.items():
        print(f"\nCell {cell_idx}:")
        for node_idx, edge_arch in enumerate(cell_arch):
            print(f"  Node {node_idx}:")
            for j, op_name in edge_arch:
                print(f"    Edge from node {j} -> {op_name}")
    
    print("\n" + "=" * 80)
    print(f"Best Val Accuracy: {results['best_val_acc']:.2f}%")
    print("=" * 80)
    
    return results, arch

if __name__ == "__main__":
    try:
        results, arch = run_darts_small()
        print("\nNAS completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

