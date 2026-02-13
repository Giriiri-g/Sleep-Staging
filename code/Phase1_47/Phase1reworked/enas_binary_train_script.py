"""
Complete ENAS Training Script for Binary Disease Classification

Usage:
    python enas_binary_train_script.py \
        --data_path csv-docs/cfs_binary_disease.csv \
        --disease_column sleep_apnea \
        --save_dir enas_binary_results \
        --num_epochs 50
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import ENAS components
from enas_binary_classification import SharedModel, Controller, OP_NAMES
from enas_trainer_binary import ENASTrainer, compute_class_weight
from cfs_dataset import load_cfs_dataframe, CFSAilmentDataset
from torch.utils.data import DataLoader
from utils import print_header, print_section, print_info, print_success, print_key_value
from cfs_dataset import create_binary_dataloaders  # Add this import


# At top, add:
import sys
from datetime import datetime



def run_enas_search(args):
    """Run ENAS search for binary disease classification"""
    
    print_header("ENAS for Binary Disease Classification")
    print_header(f"Disease: {args.disease_column}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("Configuration")
    print_key_value("Device", device)
    print_key_value("Disease", args.disease_column)
    print_key_value("Data path", args.data_path)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print_section("Loading Data")
    train_loader, val_loader, test_loader, stats = create_binary_dataloaders(
        csv_path=args.data_path,
        disease_column=args.disease_column,
        batch_size=args.batch_size,
        input_channels=args.input_channels,
        input_length=args.input_length,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
        channel_names=args.channel_names.split(',') if args.channel_names else None,
        target_sample_rate=args.target_sample_rate,
        normalization=args.normalization,
    )
    
    print_key_value("Train samples", stats['train'])
    print_key_value("Val samples", stats['val'])
    print_key_value("Test samples", stats['test'])
    
    # Compute class weight
    pos_weight = compute_class_weight(train_loader)
    
    # Create models
    print_section("Creating Models")
    
    shared_model = SharedModel(
        input_channels=args.input_channels,
        input_length=args.input_length,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes
    )
    
    controller = Controller(
        num_nodes=args.num_nodes,
        num_ops=len(OP_NAMES),
        hidden_dim=args.controller_hidden_dim,
        temperature=args.controller_temperature
    )
    
    shared_params = sum(p.numel() for p in shared_model.parameters() if p.requires_grad)
    controller_params = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    
    print_key_value("Shared model parameters", f"{shared_params:,}")
    print_key_value("Controller parameters", f"{controller_params:,}")
    print_key_value("Hidden dimension", args.hidden_dim)
    print_key_value("Number of nodes", args.num_nodes)
    print_key_value("Number of operations", len(OP_NAMES))
    
    print_info("\nAvailable operations:")
    for i, op in enumerate(OP_NAMES):
        print(f"  {i}: {op}")
    
    # Create trainer
    print_section("Creating Trainer")
    
    trainer = ENASTrainer(
        shared_model=shared_model,
        controller=controller,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        # Loss function
        loss_type=args.loss_type,
        pos_weight=pos_weight if args.auto_pos_weight else args.pos_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        # Shared model optimization
        shared_optimizer=args.shared_optimizer,
        shared_lr=args.shared_lr,
        shared_momentum=args.shared_momentum,
        shared_weight_decay=args.shared_weight_decay,
        shared_grad_clip=args.shared_grad_clip,
        # Controller optimization
        controller_optimizer=args.controller_optimizer,
        controller_lr=args.controller_lr,
        controller_entropy_weight=args.controller_entropy_weight,
        controller_baseline_decay=args.controller_baseline_decay,
        controller_grad_clip=args.controller_grad_clip,
        # Training schedule
        shared_num_steps=args.shared_num_steps,
        controller_num_steps=args.controller_num_steps,
        controller_num_aggregate=args.controller_num_aggregate,
    )
    
    print_key_value("Loss type", args.loss_type)
    print_key_value("Shared optimizer", args.shared_optimizer)
    print_key_value("Shared LR", args.shared_lr)
    print_key_value("Controller optimizer", args.controller_optimizer)
    print_key_value("Controller LR", args.controller_lr)
    print_key_value("Entropy weight", args.controller_entropy_weight)
    
    # Run search
    print_section("Running Search")
    
    results = trainer.search(
        num_epochs=args.num_epochs,
        print_freq=args.print_freq
    )
    
    # Derive final architecture
    print_section("Deriving Final Architecture")
    best_dag, best_reward = trainer.derive_architecture(num_samples=args.derive_num_samples)
    
    print_success(f"Best architecture reward: {best_reward:.4f}")
    
    # Save results
    print_section("Saving Results")
    
    results_dict = {
        'config': {
            'disease': args.disease_column,
            'input_channels': args.input_channels,
            'input_length': args.input_length,
            'hidden_dim': args.hidden_dim,
            'num_nodes': args.num_nodes,
            'loss_type': args.loss_type,
            'pos_weight': pos_weight if args.auto_pos_weight else args.pos_weight,
        },
        'stats': stats,
        'best_dag': results['best_dag'],
        "best_reward": results["best_reward"],
        'derived_dag': best_dag,
        'derived_reward': best_reward,
        'history': results['history'],
    }
    
    # Save as JSON
    results_path = save_dir / f"enas_{args.disease_column}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    # Save model checkpoint
    checkpoint = {
        'shared_model': shared_model.state_dict(),
        'controller': controller.state_dict(),
        'best_dag': best_dag,
        'config': results_dict['config'],
    }
    checkpoint_path = save_dir / f"enas_{args.disease_column}_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    
    print_success(f"Results saved to {save_dir}")
    print_key_value("Results file", results_path)
    print_key_value("Checkpoint file", checkpoint_path)
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description="ENAS for Binary Disease Classification"
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to CSV file")
    parser.add_argument("--disease_column", type=str, required=True,
                       help="Name of disease column for binary classification")
    parser.add_argument("--input_channels", type=int, default=7,
                       help="Number of input channels")
    parser.add_argument("--input_length", type=int, default=3000,
                       help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.15,
                       help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15,
                       help="Test split ratio")
    parser.add_argument("--channel_names", type=str, default=None,
                       help="Comma-separated channel names")
    parser.add_argument("--target_sample_rate", type=float, default=128.0,
                       help="Target sample rate")
    parser.add_argument("--normalization", type=str, default="zscore",
                       choices=["zscore", "minmax", "none"],
                       help="Normalization method")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=64,
                       help="Hidden dimension for shared model")
    parser.add_argument("--num_nodes", type=int, default=5,
                       help="Number of nodes in DAG")
    parser.add_argument("--controller_hidden_dim", type=int, default=100,
                       help="Hidden dimension for controller LSTM")
    parser.add_argument("--controller_temperature", type=float, default=2.0,
                       help="Temperature for controller sampling")
    
    # Loss function
    parser.add_argument("--loss_type", type=str, default="focal",
                       choices=["focal", "weighted_bce", "bce"],
                       help="Loss function type")
    parser.add_argument("--auto_pos_weight", action="store_true",
                       help="Automatically compute pos_weight from data")
    parser.add_argument("--pos_weight", type=float, default=5.0,
                       help="Positive class weight (if not auto)")
    parser.add_argument("--focal_alpha", type=float, default=0.75,
                       help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=0.8,
                       help="Focal loss gamma parameter")
    
    # Shared model optimization
    parser.add_argument("--shared_optimizer", type=str, default="adam",
                       choices=["sgd", "adam"],
                       help="Optimizer for shared model")
    parser.add_argument("--shared_lr", type=float, default=0.01,
                       help="Learning rate for shared model")
    parser.add_argument("--shared_momentum", type=float, default=0.9,
                       help="Momentum for SGD")
    parser.add_argument("--shared_weight_decay", type=float, default=1e-4,
                       help="Weight decay for shared model")
    parser.add_argument("--shared_grad_clip", type=float, default=5.0,
                       help="Gradient clipping for shared model")
    
    # Controller optimization
    parser.add_argument("--controller_optimizer", type=str, default="adam",
                       choices=["adam", "sgd"],
                       help="Optimizer for controller")
    parser.add_argument("--controller_lr", type=float, default=3.5e-4,
                       help="Learning rate for controller")
    parser.add_argument("--controller_entropy_weight", type=float, default=1e-4,
                       help="Entropy regularization weight")
    parser.add_argument("--controller_baseline_decay", type=float, default=0.999,
                       help="Baseline decay rate")
    parser.add_argument("--controller_grad_clip", type=float, default=5.0,
                       help="Gradient clipping for controller")
    
    # Training schedule
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of search epochs")
    parser.add_argument("--shared_num_steps", type=int, default=100,
                       help="Steps to train shared model per epoch")
    parser.add_argument("--controller_num_steps", type=int, default=50,
                       help="Steps to train controller per epoch")
    parser.add_argument("--controller_num_aggregate", type=int, default=10,
                       help="Number of samples to aggregate controller gradients")
    parser.add_argument("--derive_num_samples", type=int, default=100,
                       help="Number of samples for deriving final architecture")
    
    # Save arguments
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Save directory")
    parser.add_argument("--print_freq", type=int, default=1,
                       help="Print frequency")
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"enas_{args.disease_column}_{timestamp}"

    try:
        results = run_enas_search(args)
        print_success("\nENAS search completed successfully!")
    except KeyboardInterrupt:
        print("\n[WARNING] Search interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
