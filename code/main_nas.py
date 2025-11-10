"""
Main script for Neural Architecture Search for Sleep Stage Classification
Supports both DARTS and RL-based search strategies
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import warnings

# Suppress all warnings at the start
warnings.filterwarnings('ignore')
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

from nas_search_space import Network, DARTS_OPS, ALL_OPS
from darts import DARTSTrainer
from rl_search import RLSearchTrainer, PolicyNetwork, actions_to_architecture
from nas_evaluator import ArchitectureEvaluator, PerformanceEstimator, print_architecture
from dataloader import LazyPSGDataset
from utils import (print_header, print_section, print_info, print_success, 
                   print_warning, print_error, print_key_value, print_metric,
                   suppress_warnings)

# Ensure warnings are suppressed
suppress_warnings()


def get_data_loaders(data_path: str, batch_size: int = 32, val_split: float = 0.2,
                    test_split: float = 0.1, num_workers: int = 0):
    """
    Get data loaders for training, validation, and testing
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of worker processes
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print_info("Loading dataset...")
    # Load dataset
    dataset = LazyPSGDataset(folder_path=data_path, window_size=30)
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print_section("Dataset Information")
    print_key_value("Total samples", f"{total_size:,}")
    print_key_value("Train samples", f"{train_size:,}")
    print_key_value("Validation samples", f"{val_size:,}")
    print_key_value("Test samples", f"{test_size:,}")
    print_key_value("Batch size", batch_size)
    
    return train_loader, val_loader, test_loader


def run_darts_search(args):
    """Run DARTS search"""
    print_header("DARTS: Differentiable Architecture Search")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_section("Configuration")
    print_key_value("Device", device)
    print_key_value("Data path", args.data_path)
    print_key_value("Strategy", "DARTS")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_path, args.batch_size, args.val_split, args.test_split
    )
    
    # Create supernet
    print_info("Creating supernet...")
    model = Network(
        input_channels=args.input_channels,
        input_length=args.input_length,
        num_classes=args.num_classes,
        init_channels=args.init_channels,
        num_cells=args.num_cells,
        num_nodes=args.num_nodes,
        search_space=DARTS_OPS
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print_key_value("Supernet parameters", f"{num_params:,}")
    
    # Setup checkpoint directory
    checkpoint_dir = None
    if args.save_dir:
        checkpoint_dir = Path(args.save_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DARTS trainer
    trainer = DARTSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        w_lr=args.w_lr,
        w_momentum=args.w_momentum,
        w_weight_decay=args.w_weight_decay,
        alpha_lr=args.alpha_lr,
        alpha_weight_decay=args.alpha_weight_decay,
        unrolled=args.unrolled,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None
    )
    
    # Determine checkpoint to resume from
    resume_from = None
    if args.resume_from:
        resume_from = args.resume_from
    elif checkpoint_dir:
        # Auto-detect latest checkpoint
        latest_checkpoint = checkpoint_dir / "checkpoint_latest.pth"
        if latest_checkpoint.exists():
            resume_from = str(latest_checkpoint)
            print_info(f"Found checkpoint: {resume_from}")
    
    # Train
    print_section("Training")
    results = trainer.train(
        num_epochs=args.num_epochs,
        print_freq=args.print_freq,
        resume_from=resume_from,
        save_freq=args.checkpoint_freq
    )
    
    # Get final architecture
    final_arch = trainer.get_architecture()
    print_section("Final Architecture")
    print_architecture(final_arch)
    
    # Evaluate final architecture
    print_section("Evaluation")
    print_info("Evaluating final architecture...")
    evaluator = ArchitectureEvaluator(train_loader, val_loader, device, args.num_classes)
    eval_results = evaluator.evaluate(final_arch, num_epochs=args.final_train_epochs, verbose=True,
                                     num_cells=args.num_cells, num_nodes=args.num_nodes)
    
    print_section("Final Results")
    print_metric("Best Val Accuracy", eval_results['best_val_acc'], "%")
    print_metric("Final Val Accuracy", eval_results['final_val_acc'], "%")
    
    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save architecture
        with open(save_dir / "darts_architecture.json", "w") as f:
            json.dump(final_arch, f, indent=2)
        
        # Save results
        results['final_evaluation'] = eval_results
        results['architecture'] = final_arch
        with open(save_dir / "darts_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        torch.save(model.state_dict(), save_dir / "darts_model.pth")
        
        print_success(f"Results saved to {save_dir}")
    
    return results, final_arch


def run_rl_search(args):
    """Run RL-based search"""
    print("=" * 60)
    print("RL-based Neural Architecture Search")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_path, args.batch_size, args.val_split, args.test_split
    )
    
    # Create policy network
    policy = PolicyNetwork(
        num_cells=args.num_cells,
        num_nodes=args.num_nodes,
        num_ops=len(ALL_OPS),
        hidden_dim=args.policy_hidden_dim
    )
    
    print(f"Policy network parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Create RL trainer
    trainer = RLSearchTrainer(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        reward_type=args.reward_type,
        baseline_type=args.baseline_type,
        baseline_decay=args.baseline_decay,
        temperature=args.temperature,
        entropy_coeff=args.entropy_coeff,
        lr=args.rl_lr
    )
    
    # Search
    results = trainer.search(
        num_iterations=args.num_iterations,
        eval_epochs=args.eval_epochs,
        print_freq=args.print_freq
    )
    
    # Get best architecture
    best_arch = results['best_architecture']
    print("\nBest Architecture:")
    print_architecture(best_arch)
    
    # Evaluate best architecture
    print("\nEvaluating best architecture...")
    evaluator = ArchitectureEvaluator(train_loader, val_loader, device, args.num_classes)
    eval_results = evaluator.evaluate(best_arch, num_epochs=args.final_train_epochs, verbose=True,
                                     num_cells=args.num_cells, num_nodes=args.num_nodes)
    
    print(f"\nFinal Evaluation Results:")
    print(f"Best Val Accuracy: {eval_results['best_val_acc']:.2f}%")
    print(f"Final Val Accuracy: {eval_results['final_val_acc']:.2f}%")
    
    # Estimate performance metrics
    estimator = PerformanceEstimator(device)
    num_params = estimator.estimate_params(best_arch, args.input_channels, args.input_length, args.num_classes,
                                          num_cells=args.num_cells, num_nodes=args.num_nodes)
    flops = estimator.estimate_flops(best_arch, args.input_channels, args.input_length)
    
    print(f"\nArchitecture Metrics:")
    print(f"Parameters: {num_params:,}")
    print(f"Estimated FLOPs: {flops:,}")
    
    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save architecture
        with open(save_dir / "rl_architecture.json", "w") as f:
            json.dump(best_arch, f, indent=2)
        
        # Save results
        results['final_evaluation'] = eval_results
        results['architecture'] = best_arch
        results['num_params'] = num_params
        results['flops'] = flops
        with open(save_dir / "rl_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save policy
        torch.save(policy.state_dict(), save_dir / "rl_policy.pth")
        
        print(f"\nResults saved to {save_dir}")
    
    return results, best_arch


def main():
    parser = argparse.ArgumentParser(description="Neural Architecture Search for Sleep Stage Classification")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to sleep data directory")
    parser.add_argument("--input_channels", type=int, default=7,
                       help="Number of input channels (default: 7)")
    parser.add_argument("--input_length", type=int, default=3000,
                       help="Input sequence length (default: 3000 for 30s at 100Hz)")
    parser.add_argument("--num_classes", type=int, default=7,
                       help="Number of classes (default: 7)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument("--test_split", type=float, default=0.1,
                       help="Test split ratio (default: 0.1)")
    
    # Architecture arguments
    parser.add_argument("--num_cells", type=int, default=8,
                       help="Number of cells in network (default: 8)")
    parser.add_argument("--num_nodes", type=int, default=4,
                       help="Number of nodes per cell (default: 4)")
    parser.add_argument("--init_channels", type=int, default=64,
                       help="Initial number of channels (default: 64)")
    
    # Search strategy
    parser.add_argument("--strategy", type=str, choices=["darts", "rl", "both"],
                       default="darts", help="Search strategy (default: darts)")
    
    # DARTS arguments
    parser.add_argument("--w_lr", type=float, default=0.025,
                       help="Learning rate for model weights (default: 0.025)")
    parser.add_argument("--w_momentum", type=float, default=0.9,
                       help="Momentum for model weights (default: 0.9)")
    parser.add_argument("--w_weight_decay", type=float, default=3e-4,
                       help="Weight decay for model weights (default: 3e-4)")
    parser.add_argument("--alpha_lr", type=float, default=3e-4,
                       help="Learning rate for architecture parameters (default: 3e-4)")
    parser.add_argument("--alpha_weight_decay", type=float, default=1e-3,
                       help="Weight decay for architecture parameters (default: 1e-3)")
    parser.add_argument("--unrolled", action="store_true",
                       help="Use unrolled optimization (second-order DARTS)")
    
    # RL arguments
    parser.add_argument("--policy_hidden_dim", type=int, default=128,
                       help="Hidden dimension of policy network (default: 128)")
    parser.add_argument("--reward_type", type=str, choices=["accuracy", "loss"],
                       default="accuracy", help="Reward type (default: accuracy)")
    parser.add_argument("--baseline_type", type=str, choices=["moving_average", "none"],
                       default="moving_average", help="Baseline type (default: moving_average)")
    parser.add_argument("--baseline_decay", type=float, default=0.9,
                       help="Baseline decay factor (default: 0.9)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for policy sampling (default: 1.0)")
    parser.add_argument("--entropy_coeff", type=float, default=0.0001,
                       help="Entropy coefficient (default: 0.0001)")
    parser.add_argument("--rl_lr", type=float, default=0.00035,
                       help="Learning rate for policy network (default: 0.00035)")
    parser.add_argument("--num_iterations", type=int, default=100,
                       help="Number of search iterations for RL (default: 100)")
    parser.add_argument("--eval_epochs", type=int, default=5,
                       help="Number of epochs to train each architecture in RL (default: 5)")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs for DARTS (default: 50)")
    parser.add_argument("--final_train_epochs", type=int, default=20,
                       help="Number of epochs for final architecture training (default: 20)")
    parser.add_argument("--print_freq", type=int, default=10,
                       help="Print frequency (default: 10)")
    
    # Save arguments
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save results (default: None)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint file to resume from (default: None)")
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                       help="Frequency of checkpoint saving (default: 5 epochs)")
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"nas_results_{args.strategy}_{timestamp}"
    
    print_header("Neural Architecture Search for Sleep Stage Classification")
    print_key_value("Save directory", args.save_dir)
    if args.resume_from:
        print_key_value("Resume from", args.resume_from)
    
    # Run search
    try:
        if args.strategy == "darts":
            run_darts_search(args)
        elif args.strategy == "rl":
            run_rl_search(args)
        elif args.strategy == "both":
            print_info("Running both DARTS and RL search...")
            darts_results, darts_arch = run_darts_search(args)
            print_section("RL Search")
            rl_results, rl_arch = run_rl_search(args)
            
            # Compare results
            print_header("Comparison Results")
            print_metric("DARTS Best Val Acc", darts_results['best_val_acc'], "%")
            print_metric("RL Best Reward", rl_results['best_reward'], "")
        print_success("Training completed successfully!")
    except KeyboardInterrupt:
        print_warning("Training interrupted by user")
        print_info("Checkpoint should be available for resume")
    except Exception as e:
        print_error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

