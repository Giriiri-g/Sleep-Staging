"""Run NAS with the sleep staging data"""
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_nas import run_darts_search, run_rl_search
import argparse

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    class Args:
        data_path = str(project_root / "csv-docs" / "cfs_visit5_selected.csv")
        input_channels = 7
        input_length = 3000
        num_classes = 20
        batch_size = 4  # EDF loading is memory intensive
        num_workers = 0
        val_split = 100  # absolute count for validation
        test_split = 0.2  # ratio applied to remaining samples
        channel_names = None
        target_sample_rate = 100.0
        split_seed = 42
        normalization = "zscore"
        num_cells = 6  # Fewer cells for faster search
        num_nodes = 3  # Fewer nodes for faster search
        init_channels = 32  # Smaller initial channels
        strategy = "darts"  # Start with DARTS
        task_type = "multi_label"
        prediction_threshold = 0.5
        w_lr = 0.025
        w_momentum = 0.9
        w_weight_decay = 3e-4
        alpha_lr = 3e-4
        alpha_weight_decay = 1e-3
        unrolled = False
        policy_hidden_dim = 64
        reward_type = "accuracy"
        baseline_type = "moving_average"
        baseline_decay = 0.9
        temperature = 1.0
        entropy_coeff = 0.0001
        rl_lr = 0.00035
        num_iterations = 50
        eval_epochs = 3
        num_epochs = 30  # Reduced for faster initial search
        final_train_epochs = 15
        print_freq = 5
        save_dir = "nas_results_darts"
    
    args = Args()
    
    print("=" * 80)
    print("Neural Architecture Search for Sleep Stage Classification")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Input: {args.input_channels} channels × {args.input_length} time steps")
    print(f"Output: {args.num_classes} sleep stages")
    print(f"Strategy: {args.strategy}")
    print(f"Architecture: {args.num_cells} cells, {args.num_nodes} nodes per cell")
    print("=" * 80)
    
    if args.strategy == "darts":
        print("\nRunning DARTS search...")
        results, arch = run_darts_search(args)
        print("\n" + "=" * 80)
        print("DARTS Search Completed!")
        print("=" * 80)
    elif args.strategy == "rl":
        print("\nRunning RL-based search...")
        results, arch = run_rl_search(args)
        print("\n" + "=" * 80)
        print("RL Search Completed!")
        print("=" * 80)
    else:
        print(f"Unknown strategy: {args.strategy}")

