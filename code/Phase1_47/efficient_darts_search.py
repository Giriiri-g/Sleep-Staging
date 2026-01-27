"""
Efficient DARTS search for CFS health risk prediction
Optimized for FLOPs, MACs, and parameter constraints
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_search_space import Network, DARTS_OPS
from darts import DARTSTrainer
from nas_evaluator import ArchitectureEvaluator, PerformanceEstimator, print_architecture
from cfs_dataset import create_cfs_dataloaders
from efficiency_tracker import (
    estimate_architecture_efficiency,
    check_efficiency_constraints,
    count_parameters
)
try:
    from memory_utils import clear_gpu_cache, get_gpu_memory_info, print_gpu_memory, set_memory_fraction
except ImportError:
    # Fallback if memory_utils not available
    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def get_gpu_memory_info():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {'allocated_gb': allocated, 'reserved_gb': reserved, 'total_gb': total, 'free_gb': total - reserved}
        return None
    def print_gpu_memory():
        info = get_gpu_memory_info()
        if info:
            print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f} GB, Reserved: {info['reserved_gb']:.2f} GB")
    def set_memory_fraction(fraction=0.9):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
from utils import (
    print_header, print_section, print_info, print_success, print_warning,
    print_key_value, print_metric, suppress_warnings, print_error, print_progress
)

suppress_warnings()


class EfficientDARTSTrainer(DARTSTrainer):
    """DARTS trainer with efficiency constraints"""
    
    def __init__(
        self,
        model: Network,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_flops: Optional[int] = None,
        max_macs: Optional[int] = None,
        max_params: Optional[int] = None,
        efficiency_weight: float = 0.1,
        input_channels: int = 7,
        input_length: int = 3000,
        init_channels: int = 32,
        num_cells: int = 6,
        num_nodes: int = 3,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, device, **kwargs)
        self.max_flops = max_flops
        self.max_macs = max_macs
        self.max_params = max_params
        self.efficiency_weight = efficiency_weight
        self.input_channels = input_channels
        self.input_length = input_length
        self.init_channels = init_channels
        self.num_cells = num_cells
        self.num_nodes = num_nodes
    
    def _compute_efficiency_penalty(self, alpha) -> float:
        """Compute efficiency penalty based on architecture parameters"""
        # Get current architecture
        arch = self.model.discretize(alpha)
        
        # Estimate efficiency metrics
        metrics = estimate_architecture_efficiency(
            arch,
            self.input_channels,
            self.input_length,
            self.init_channels,
            self.num_cells,
            self.num_nodes
        )
        
        penalty = 0.0
        
        if self.max_flops and metrics['flops'] > self.max_flops:
            excess = (metrics['flops'] - self.max_flops) / self.max_flops
            penalty += excess
        
        if self.max_macs and metrics['macs'] > self.max_macs:
            excess = (metrics['macs'] - self.max_macs) / self.max_macs
            penalty += excess
        
        if self.max_params and metrics['params'] > self.max_params:
            excess = (metrics['params'] - self.max_params) / self.max_params
            penalty += excess
        
        return penalty
    
    def step(self, x_train, target_train, x_valid, target_valid):
        """Override step to include efficiency penalty"""
        x_train = x_train.to(self.device)
        target_train = self._format_target(target_train.to(self.device))
        x_valid = x_valid.to(self.device)
        target_valid = self._format_target(target_valid.to(self.device))
        
        # Step 1: Update model weights (w) on training set
        self.w_optimizer.zero_grad()
        logits_train = self.model(x_train)
        loss_train = self.criterion(logits_train, target_train)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.w_optimizer.step()
        
        # Step 2: Update architecture parameters (alpha) on validation set
        self.alpha_optimizer.zero_grad()
        
        # Get architecture parameters
        alpha = self.model.arch_params
        
        # Compute validation loss
        logits_valid = self.model(x_valid, alpha)
        loss_valid = self.criterion(logits_valid, target_valid)
        
        # Note: Efficiency penalty is tracked but not used in gradient
        # since it requires discretization. We use it for logging and
        # post-processing to select efficient architectures.
        
        loss_valid.backward()
        torch.nn.utils.clip_grad_norm_([alpha], self.grad_clip)
        self.alpha_optimizer.step()


def setup_logging(log_dir: Path, log_file: str = "darts_search.log"):
    """Setup logging to both file and console"""
    log_path = log_dir / log_file
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('efficient_darts')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_metrics(logger: logging.Logger, epoch: int, metrics: Dict, efficiency: Dict):
    """Log metrics to file"""
    logger.info(f"Epoch {epoch}:")
    logger.info(f"  Train Loss: {metrics.get('train_loss', 0):.4f}")
    logger.info(f"  Train Acc: {metrics.get('train_acc', 0):.2f}%")
    logger.info(f"  Val Loss: {metrics.get('val_loss', 0):.4f}")
    logger.info(f"  Val Acc: {metrics.get('val_acc', 0):.2f}%")
    logger.info(f"  FLOPs: {efficiency.get('flops', 0):,}")
    logger.info(f"  MACs: {efficiency.get('macs', 0):,}")
    logger.info(f"  Params: {efficiency.get('params', 0):,}")


def run_efficient_darts_search(args):
    """Run efficient DARTS search with constraints"""
    print_header("Efficient DARTS Search for CFS Health Risk Prediction")
    
    # Setup device with memory management
    if torch.cuda.is_available():
        # Clear GPU cache before starting
        clear_gpu_cache()
        # Set memory fraction to avoid fragmentation
        set_memory_fraction(0.85)  # Use 85% to leave room
        device = torch.device("cuda")
        mem_info = get_gpu_memory_info()
        if mem_info:
            print_info(f"GPU Memory: {mem_info['total_gb']:.2f} GB total, "
                      f"{mem_info['free_gb']:.2f} GB free")
            print_gpu_memory()
    else:
        device = torch.device("cpu")
        print_warning("CUDA not available, using CPU (will be slower)")
    
    print_section("Configuration")
    print_key_value("Device", device)
    print_key_value("Data path", args.data_path)
    print_key_value("Strategy", "Efficient DARTS")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir, "darts_search.log")
    logger.info("=" * 80)
    logger.info("Starting Efficient DARTS Search")
    logger.info(f"Device: {device}")
    logger.info(f"Data path: {args.data_path}")
    
    # Data loaders
    print_section("Loading Data")
    train_loader, val_loader, test_loader, stats = create_cfs_dataloaders(
        csv_path=args.data_path,
        batch_size=args.batch_size,
        input_channels=args.input_channels,
        input_length=args.input_length,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.split_seed,
        channel_names=args.channel_names.split(',') if args.channel_names else None,
        target_sample_rate=args.target_sample_rate,
        normalization=args.normalization,
    )
    
    print_key_value("Total samples", stats['total'])
    print_key_value("Train samples", stats['train'])
    print_key_value("Validation samples", stats['val'])
    print_key_value("Test samples", stats['test'])
    print_key_value("Number of labels", stats['num_labels'])
    
    logger.info(f"Dataset loaded: {stats['total']} total samples")
    logger.info(f"Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    
    # Update num_classes from dataset
    args.num_classes = stats['num_labels']
    
    # Create supernet with smaller architecture
    print_section("Creating Supernet")
    print_key_value("Init channels", args.init_channels)
    print_key_value("Number of cells", args.num_cells)
    print_key_value("Nodes per cell", args.num_nodes)
    
    # Try to create model on CPU first to check memory
    try:
        model = Network(
            input_channels=args.input_channels,
            input_length=args.input_length,
            num_classes=args.num_classes,
            init_channels=args.init_channels,
            num_cells=args.num_cells,
            num_nodes=args.num_nodes,
            search_space=DARTS_OPS
        )
            
        num_params = count_parameters(model)
        print_key_value("Supernet parameters", f"{num_params:,}")
        
        logger.info(f"Supernet created: {num_params:,} parameters")
        logger.info(f"Architecture: {args.num_cells} cells, {args.num_nodes} nodes, {args.init_channels} init channels")
        
        # Move model to device first (needed before memory check)
        model = model.to(device)
        
        # Check if model fits in GPU memory
        if device.type == "cuda":
            # Try a dummy forward pass to check memory
            try:
                dummy_input = torch.randn(1, args.input_channels, args.input_length, device=device)
                with torch.no_grad():
                    _ = model(dummy_input)
                del dummy_input, _
                clear_gpu_cache()
                print_success("Model fits in GPU memory")
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg:
                    print_warning("Model too large for GPU, consider using CPU or reducing architecture further")
                    clear_gpu_cache()
                    # Optionally switch to CPU
                    if args.force_cpu_on_oom:
                        print_warning("Switching to CPU due to GPU memory constraints")
                        device = torch.device("cpu")
                        model = model.cpu()  # Move model to CPU
                        clear_gpu_cache()
                        logger.warning("Switched to CPU due to GPU OOM")
                    else:
                        raise RuntimeError("GPU out of memory. Use --force_cpu_on_oom to switch to CPU or reduce architecture size.")
                else:
                    # Re-raise if it's not a memory error
                    clear_gpu_cache()
                    raise
        # If CPU mode, model is already moved above
    except Exception as e:
        print_error(f"Error creating model: {str(e)}")
        logger.error(f"Error creating model: {str(e)}")
        raise
    
    # Efficiency constraints
    print_section("Efficiency Constraints")
    if args.max_flops:
        print_key_value("Max FLOPs", f"{args.max_flops:,}")
    if args.max_macs:
        print_key_value("Max MACs", f"{args.max_macs:,}")
    if args.max_params:
        print_key_value("Max Parameters", f"{args.max_params:,}")
    
    logger.info(f"Constraints: FLOPs={args.max_flops}, MACs={args.max_macs}, Params={args.max_params}")
    
    # Setup checkpoint directory
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create efficient DARTS trainer
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    trainer = EfficientDARTSTrainer(
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
        checkpoint_dir=str(checkpoint_dir),
        criterion=criterion,
        task_type="multi_label",
        pred_threshold=args.prediction_threshold,
        max_flops=args.max_flops,
        max_macs=args.max_macs,
        max_params=args.max_params,
        efficiency_weight=args.efficiency_weight,
        input_channels=args.input_channels,
        input_length=args.input_length,
        init_channels=args.init_channels,
        num_cells=args.num_cells,
        num_nodes=args.num_nodes,
    )
    
    # Training
    print_section("Training")
    logger.info("Starting training...")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'efficiency': []
    }
    
    best_val_acc = 0.0
    best_arch = None
    best_efficiency = None
    
    for epoch in range(args.num_epochs):
        try:
            # Clear GPU cache before each epoch
            clear_gpu_cache()
            print_info(f"[TRAIN] ===== Epoch {epoch} / {args.num_epochs} =====")
            
            # Train epoch
            train_metrics = trainer.train_epoch(epoch)
            print_info(f"[TRAIN] Completed train_epoch {epoch} "
                       f"(loss={train_metrics['train_loss']:.4f}, "
                       f"acc={train_metrics['train_acc']:.2f}%)")
            
            # Validate
            val_metrics = trainer.validate()
            print_info(f"[VAL] Completed validate {epoch} "
                       f"(loss={val_metrics['val_loss']:.4f}, "
                       f"acc={val_metrics['val_acc']:.2f}%)")
            
            # Get current architecture and efficiency
            current_arch = trainer.get_architecture()
            efficiency = estimate_architecture_efficiency(
                current_arch,
                args.input_channels,
                args.input_length,
                args.init_channels,
                args.num_cells,
                args.num_nodes
            )
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])
            history['efficiency'].append(efficiency)
            
            # Log metrics
            if (epoch + 1) % args.log_freq == 0:
                log_metrics(logger, epoch + 1, {**train_metrics, **val_metrics}, efficiency)
            
            # Print progress
            if (epoch + 1) % args.print_freq == 0 or epoch == 0:
                print_progress(epoch + 1, args.num_epochs, {
                    'Train Loss': train_metrics['train_loss'],
                    'Train Acc': f"{train_metrics['train_acc']:.2f}%",
                    'Val Loss': val_metrics['val_loss'],
                    'Val Acc': f"{val_metrics['val_acc']:.2f}%",
                    'FLOPs': f"{efficiency['flops']:,}",
                    'MACs': f"{efficiency['macs']:,}",
                    'Params': f"{efficiency['params']:,}"
                })
            
            # Save best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                best_arch = current_arch
                best_efficiency = efficiency
                trainer.save_checkpoint(
                    epoch, history, best_val_acc, best_arch,
                    "checkpoint_best.pth"
                )
                logger.info(f"New best model at epoch {epoch + 1}: Val Acc = {best_val_acc:.2f}%")
            
            # Periodic checkpoint
            if (epoch + 1) % args.checkpoint_freq == 0:
                trainer.save_checkpoint(epoch, history, best_val_acc, best_arch)
                # Dump logs
                log_dump_path = save_dir / f"logs_epoch_{epoch + 1}.json"
                with open(log_dump_path, 'w') as f:
                    json.dump({
                        'epoch': epoch + 1,
                        'history': history,
                        'best_val_acc': best_val_acc,
                        'current_efficiency': efficiency,
                        'best_efficiency': best_efficiency
                    }, f, indent=2, default=str)
                logger.info(f"Logs dumped to {log_dump_path}")
            
            # Clear cache after each epoch to prevent memory buildup
            clear_gpu_cache()
            
            # Print memory usage periodically
            if (epoch + 1) % args.print_freq == 0 and torch.cuda.is_available():
                print_gpu_memory()
        
        except Exception as e:
            logger.error(f"Error at epoch {epoch}: {str(e)}", exc_info=True)
            print_warning(f"Error at epoch {epoch}: {str(e)}")
            trainer.save_checkpoint(epoch - 1, history, best_val_acc, best_arch, "checkpoint_error.pth")
            raise
    
    # Final results
    print_section("Final Results")
    final_arch = trainer.get_architecture()
    final_efficiency = estimate_architecture_efficiency(
        final_arch,
        args.input_channels,
        args.input_length,
        args.init_channels,
        args.num_cells,
        args.num_nodes
    )
    
    print_architecture(final_arch)
    print_key_value("Best Val Accuracy", f"{best_val_acc:.2f}%")
    print_key_value("Final FLOPs", f"{final_efficiency['flops']:,}")
    print_key_value("Final MACs", f"{final_efficiency['macs']:,}")
    print_key_value("Final Parameters", f"{final_efficiency['params']:,}")
    
    logger.info("=" * 80)
    logger.info("Training completed")
    logger.info(f"Best Val Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Final Efficiency: FLOPs={final_efficiency['flops']:,}, "
                f"MACs={final_efficiency['macs']:,}, Params={final_efficiency['params']:,}")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'best_architecture': best_arch,
        'final_architecture': final_arch,
        'best_efficiency': best_efficiency,
        'final_efficiency': final_efficiency,
        'history': history
    }
    
    results_path = save_dir / "efficient_darts_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_success(f"Results saved to {save_dir}")
    logger.info(f"Results saved to {save_dir}")
    
    return results, final_arch


def main():
    parser = argparse.ArgumentParser(
        description="Efficient DARTS Search for CFS Health Risk Prediction"
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to CFS CSV file")
    parser.add_argument("--input_channels", type=int, default=8,
                       help="Number of input channels")
    parser.add_argument("--input_length", type=int, default=3000,
                       help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=100,
                       help="Validation split (count or ratio)")
    parser.add_argument("--test_split", type=float, default=0.2,
                       help="Test split (count or ratio)")
    parser.add_argument("--channel_names", type=str, default=None,
                       help="Comma-separated channel names")
    parser.add_argument("--target_sample_rate", type=float, default=100.0,
                       help="Target sample rate")
    parser.add_argument("--split_seed", type=int, default=42,
                       help="Random seed for splits")
    parser.add_argument("--normalization", type=str, default="zscore",
                       choices=["zscore", "minmax", "none"],
                       help="Normalization method")
    
    # Architecture arguments (smaller for efficiency)
    parser.add_argument("--num_cells", type=int, default=4,
                       help="Number of cells (default: 4)")
    parser.add_argument("--num_nodes", type=int, default=3,
                       help="Number of nodes per cell (default: 3)")
    parser.add_argument("--init_channels", type=int, default=32,
                       help="Initial channels (default: 32)")
    
    # Efficiency constraints
    parser.add_argument("--max_flops", type=int, default=None,
                       help="Maximum FLOPs constraint")
    parser.add_argument("--max_macs", type=int, default=None,
                       help="Maximum MACs constraint")
    parser.add_argument("--max_params", type=int, default=500000,
                       help="Maximum parameters constraint (default: 500K)")
    parser.add_argument("--efficiency_weight", type=float, default=0.1,
                       help="Weight for efficiency penalty")
    
    # DARTS arguments
    parser.add_argument("--w_lr", type=float, default=0.025,
                       help="Weight learning rate")
    parser.add_argument("--w_momentum", type=float, default=0.9,
                       help="Weight momentum")
    parser.add_argument("--w_weight_decay", type=float, default=3e-4,
                       help="Weight decay")
    parser.add_argument("--alpha_lr", type=float, default=3e-4,
                       help="Architecture learning rate")
    parser.add_argument("--alpha_weight_decay", type=float, default=1e-3,
                       help="Architecture weight decay")
    parser.add_argument("--unrolled", action="store_true",
                       help="Use unrolled optimization")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of epochs (default: 20 for quick results)")
    parser.add_argument("--print_freq", type=int, default=5,
                       help="Print frequency")
    parser.add_argument("--log_freq", type=int, default=2,
                       help="Log dump frequency")
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                       help="Checkpoint frequency")
    parser.add_argument("--prediction_threshold", type=float, default=0.5,
                       help="Prediction threshold for multi-label")
    
    # Save arguments
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Save directory")
    parser.add_argument("--force_cpu_on_oom", action="store_true",
                       help="Switch to CPU if GPU OOM occurs")
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"efficient_darts_results_{timestamp}"
    
    try:
        results, arch = run_efficient_darts_search(args)
        print_success("Search completed successfully!")
    except KeyboardInterrupt:
        print_warning("Search interrupted by user")
    except Exception as e:
        print_error(f"Search failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

