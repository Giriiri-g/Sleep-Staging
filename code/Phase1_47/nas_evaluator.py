"""
Evaluation utilities for Neural Architecture Search
Provides methods for evaluating architectures and estimating performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from nas_search_space import Network, ALL_OPS

def architecture_to_model(arch: Dict, input_channels: int = 7, 
                         input_length: int = 3000, num_classes: int = 7,
                         init_channels: int = 64, num_cells: int = 8,
                         num_nodes: int = 4) -> nn.Module:
    """
    Build a model from architecture dictionary
    
    Note: This is a simplified version. In practice, you'd want to
    build the actual model structure based on the architecture.
    """
    # For now, we'll use the Network class and set architecture parameters
    # to match the selected operations
    model = Network(
        input_channels=input_channels,
        input_length=input_length,
        num_classes=num_classes,
        init_channels=init_channels,
        num_cells=num_cells,
        num_nodes=num_nodes,
        search_space=ALL_OPS
    )
    
    # Set architecture parameters to one-hot encoding of selected operations
    alpha = torch.zeros_like(model.arch_params)
    
    for cell_idx in range(num_cells):
        if cell_idx not in arch:
            continue
        cell_arch = arch[cell_idx]
        
        # The edges are indexed as: for node i, there are i+2 edges (from nodes 0 to i+1)
        # Total edges per cell: sum(i+2 for i in range(num_nodes))
        edge_idx = 0
        for node_idx in range(num_nodes):
            if node_idx < len(cell_arch):
                node_arch = cell_arch[node_idx]
                for prev_node, op_name in node_arch:
                    if op_name in ALL_OPS:
                        op_idx = ALL_OPS.index(op_name)
                        # Set large value for selected operation to make softmax sharp
                        alpha[cell_idx, edge_idx, op_idx] = 100.0
                    edge_idx += 1
            else:
                # If node_arch doesn't exist, skip edges for this node
                edge_idx += node_idx + 2
    
    # Set alpha parameters
    model._arch_params.data = alpha
    
    return model



class PerformanceEstimator:
    """Estimate architecture performance without full training"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def estimate_params(self, arch: Dict, input_channels: int = 7,
                       input_length: int = 3000, num_classes: int = 7,
                       init_channels: int = 64, num_cells: int = 8,
                       num_nodes: int = 4) -> int:
        """Estimate number of parameters in architecture"""
        if len(arch) == 0:
            return 0
        num_cells = len(arch)
        num_nodes = len(arch[0]) if 0 in arch else num_nodes
        model = architecture_to_model(
            arch, input_channels, input_length, num_classes, init_channels,
            num_cells=num_cells, num_nodes=num_nodes
        )
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def estimate_flops(self, arch: Dict, input_channels: int = 7,
                      input_length: int = 3000, batch_size: int = 1) -> int:
        """Estimate FLOPs for architecture (simplified)"""
        # This is a simplified FLOPs estimation
        # In practice, you'd want to use tools like fvcore or thop
        total_flops = 0
        
        # Estimate based on operations in architecture
        for cell_arch in arch.values():
            for edge_arch in cell_arch:
                for j, op_name in edge_arch:
                    if 'conv' in op_name:
                        # Rough estimate for conv operations
                        kernel_size = int(op_name.split('_')[1].split('x')[0])
                        total_flops += input_length * kernel_size * input_channels
                    elif 'lstm' in op_name:
                        # Rough estimate for LSTM
                        total_flops += input_length * input_channels * 4
                    elif 'attention' in op_name:
                        # Rough estimate for attention
                        total_flops += input_length * input_length * input_channels
        
        return total_flops * batch_size
    
    def zero_cost_score(self, model: nn.Module, train_loader: DataLoader,
                       method: str = 'grad_norm') -> float:
        """
        Compute zero-cost proxy score
        
        Methods:
            - grad_norm: Gradient norm after one forward-backward pass
            - snip: SNIP score (connection sensitivity)
            - grasp: GRASP score
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Get a batch
        x, target = next(iter(train_loader))
        x = x.to(self.device)
        target = target.to(self.device)
        
        if method == 'grad_norm':
            # Forward and backward
            model.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            return total_norm
        
        elif method == 'snip':
            # SNIP: Connection sensitivity
            model.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            
            # Compute SNIP score
            score = 0
            for p in model.parameters():
                if p.grad is not None:
                    score += torch.abs(p * p.grad).sum().item()
            
            return score
        
        else:
            raise ValueError(f"Unknown method: {method}")


class ArchitectureEvaluator:
    """Evaluate architectures with full or partial training"""
    
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, num_classes: int = 7,
                 input_channels: int = 7, input_length: int = 3000,
                 init_channels: int = 64, task_type: str = "single_label",
                 criterion: Optional[nn.Module] = None, threshold: float = 0.5):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_length = input_length
        self.init_channels = init_channels
        self.task_type = task_type
        self.threshold = threshold
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss() if task_type == "multi_label" else nn.CrossEntropyLoss()
        self.criterion = criterion
    
    def _format_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.task_type == "multi_label":
            return target.float()
        return target.long()
    
    def _batch_accuracy(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, int]:
        if self.task_type == "multi_label":
            preds = (torch.sigmoid(logits) >= self.threshold).float()
            per_sample = (preds == target).float().mean(dim=1)
            return per_sample.sum().item(), per_sample.shape[0]
        _, predicted = logits.max(1)
        return predicted.eq(target).sum().item(), target.size(0)
    
    def evaluate(self, arch: Dict, num_epochs: int = 10, lr: float = 0.025,
                momentum: float = 0.9, weight_decay: float = 3e-4,
                verbose: bool = False, num_cells: int = 8, num_nodes: int = 4) -> Dict[str, float]:
        """
        Fully evaluate an architecture by training it
        
        Args:
            arch: Architecture dictionary
            num_epochs: Number of training epochs
            lr: Learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay
            verbose: Whether to print progress
            num_cells: Number of cells
            num_nodes: Number of nodes per cell
        
        Returns:
            Dictionary with evaluation metrics
        """
        if len(arch) == 0:
            return {'best_val_acc': 0.0, 'final_val_acc': 0.0}
        num_cells = len(arch)
        num_nodes = len(arch[0]) if 0 in arch else num_nodes
        model = architecture_to_model(
            arch,
            input_channels=self.input_channels,
            input_length=self.input_length,
            num_classes=self.num_classes,
            init_channels=self.init_channels,
            num_cells=num_cells,
            num_nodes=num_nodes
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for x, target in self.train_loader:
                x = x.to(self.device)
                target = self._format_target(target.to(self.device))
                
                optimizer.zero_grad()
                logits = model(x)
                loss = self.criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                correct, total = self._batch_accuracy(logits, target)
                train_total += total
                train_correct += correct
            
            scheduler.step()
            
            # Validate
            val_metrics = self.validate(model)
            
            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_metrics['val_loss'])
            effective_total = max(1, train_total)
            train_accs.append(100. * train_correct / effective_total)
            val_accs.append(val_metrics['val_acc'])
            
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]: '
                      f'Train Acc: {train_accs[-1]:.2f}%, '
                      f'Val Acc: {val_accs[-1]:.2f}%')
        
        return {
            'best_val_acc': best_val_acc,
            'final_val_acc': val_accs[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def quick_evaluate(self, arch: Dict, num_epochs: int = 3,
                      lr: float = 0.025, num_cells: int = 8, num_nodes: int = 4) -> float:
        """
        Quick evaluation with limited training
        
        Args:
            arch: Architecture dictionary
            num_epochs: Number of training epochs
            lr: Learning rate
            num_cells: Number of cells
            num_nodes: Number of nodes per cell
        
        Returns:
            Validation accuracy
        """
        if len(arch) == 0:
            return 0.0
        num_cells = len(arch)
        num_nodes = len(arch[0]) if 0 in arch else num_nodes
        model = architecture_to_model(
            arch,
            input_channels=self.input_channels,
            input_length=self.input_length,
            num_classes=self.num_classes,
            init_channels=self.init_channels,
            num_cells=num_cells,
            num_nodes=num_nodes
        ).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
        
        # Train for a few epochs
        model.train()
        for epoch in range(num_epochs):
            for batch_idx, (x, target) in enumerate(self.train_loader):
                if batch_idx >= 20:  # Limit batches for speed
                    break
                
                x = x.to(self.device)
                target = self._format_target(target.to(self.device))
                
                optimizer.zero_grad()
                logits = model(x)
                loss = self.criterion(logits, target)
                loss.backward()
                optimizer.step()
        
        # Validate
        val_metrics = self.validate(model)
        return val_metrics['val_acc']
    
    def validate(self, model: nn.Module) -> Dict[str, float]:
        """Validate a model"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, target in self.val_loader:
                x = x.to(self.device)
                target = self._format_target(target.to(self.device))
                
                logits = model(x)
                loss = self.criterion(logits, target)
                val_loss += loss.item()
                
                correct, total = self._batch_accuracy(logits, target)
                val_total += total
                val_correct += correct
        
        effective_total = max(1, val_total)
        return {
            'val_loss': val_loss / len(self.val_loader),
            'val_acc': 100. * val_correct / effective_total
        }


def print_architecture(arch: Dict):
    """Print architecture in a readable format"""
    print("=" * 60)
    print("Architecture:")
    print("=" * 60)
    for cell_idx, cell_arch in arch.items():
        print(f"\nCell {cell_idx}:")
        for node_idx, edge_arch in enumerate(cell_arch):
            print(f"  Node {node_idx}:")
            for j, op_name in edge_arch:
                print(f"    Edge from node {j} -> {op_name}")
    print("=" * 60)

