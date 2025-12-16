"""Test the dataloader before running NAS"""
import sys
from dataloader import LazyPSGDataset
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    data_path = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette"
    
    print("Loading dataset...")
    try:
        dataset = LazyPSGDataset(folder_path=data_path, window_size=30)
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        x, y = next(iter(loader))
        print(f"Batch shape: {x.shape}")
        print(f"Labels: {y}")
        print(f"Data type: {x.dtype}")
        print(f"Label type: {y.dtype}")
        print(f"Data range: [{x.min():.2f}, {x.max():.2f}]")
        
        # Check channel count
        print(f"Number of channels: {x.shape[1]}")
        print(f"Time steps: {x.shape[2]}")
        
        print("\nDataset loaded successfully!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

