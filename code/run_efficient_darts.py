"""
Quick run script for efficient DARTS search on CFS dataset
"""
import sys
import os
from pathlib import Path

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from efficient_darts_search import run_efficient_darts_search
import argparse

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    
    class Args:
        # Data configuration
        data_path = str(project_root / "csv-docs" / "cfs_visit5_selected.csv")
        input_channels = 8
        input_length = 3000
        batch_size = 1  # Very small batch for 4GB GPU
        num_workers = 0
        val_split = 100  # Absolute count
        test_split = 0.2  # Ratio
        channel_names = 'C3,M2,LOC,ECG1,EMG1,THOR EFFORT,ABDO EFFORT,SaO2'
        target_sample_rate = 100.0
        split_seed = 42
        normalization = "zscore"
        
        # Small architecture for efficiency (optimized for 4GB GPU)
        num_cells = 3  # Further reduced for memory
        num_nodes = 2  # Reduced nodes
        init_channels = 24  # Smaller initial channels
        
        # Efficiency constraints (set to None to disable)
        max_flops = 50_000_000  # 50M FLOPs max (None to disable)
        max_macs = 25_000_000   # 25M MACs max (None to disable)
        max_params = 500_000    # 500K parameters max (None to disable)
        efficiency_weight = 0.0  # Weight for efficiency penalty (0.0 = tracking only)
        
        # DARTS configuration
        w_lr = 0.025
        w_momentum = 0.9
        w_weight_decay = 3e-4
        alpha_lr = 3e-4
        alpha_weight_decay = 1e-3
        unrolled = False
        
        # Training configuration (quick results)
        num_epochs = 20  # Quick search
        print_freq = 2   # Print every 2 epochs
        log_freq = 2     # Log every 2 epochs
        checkpoint_freq = 5  # Checkpoint every 5 epochs
        prediction_threshold = 0.5
        
        # Memory management
        force_cpu_on_oom = False  # Set to True to switch to CPU if GPU OOM
        
        # Save directory
        save_dir = "efficient_darts_results"
    
    args = Args()
    
    print("=" * 80)
    print("Efficient DARTS Search for CFS Health Risk Prediction")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Architecture: {args.num_cells} cells, {args.num_nodes} nodes, {args.init_channels} channels")
    print(f"Constraints: FLOPs<{args.max_flops:,}, MACs<{args.max_macs:,}, Params<{args.max_params:,}")
    print(f"Training: {args.num_epochs} epochs")
    print("=" * 80)
    
    try:
        results, arch = run_efficient_darts_search(args)
        print("\n" + "=" * 80)
        print("Search Completed Successfully!")
        print("=" * 80)
        print(f"Best Val Accuracy: {results['best_val_acc']:.2f}%")
        print(f"Final FLOPs: {results['final_efficiency']['flops']:,}")
        print(f"Final MACs: {results['final_efficiency']['macs']:,}")
        print(f"Final Params: {results['final_efficiency']['params']:,}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

