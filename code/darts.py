"""
DARTS: Differentiable Architecture Search
Implementation of DARTS for neural architecture search
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
import copy
import numpy as np
from nas_search_space import Network, DARTS_OPS


class DARTSTrainer:
    """DARTS trainer for differentiable architecture search"""
    
    def __init__(self, model: Network, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, w_lr: float = 0.025, w_momentum: float = 0.9,
                 w_weight_decay: float = 3e-4, alpha_lr: float = 3e-4,
                 alpha_weight_decay: float = 1e-3, grad_clip: float = 5.0,
                 unrolled: bool = False):
        """
        Initialize DARTS trainer
        
        Args:
            model: Supernet model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            w_lr: Learning rate for model weights
            w_momentum: Momentum for model weights
            w_weight_decay: Weight decay for model weights
            alpha_lr: Learning rate for architecture parameters
            alpha_weight_decay: Weight decay for architecture parameters
            grad_clip: Gradient clipping value
            unrolled: Whether to use unrolled optimization (computationally expensive)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.unrolled = unrolled
        self.grad_clip = grad_clip
        
        # Optimizer for model weights
        self.w_optimizer = optim.SGD(
            self.model.parameters(),
            lr=w_lr,
            momentum=w_momentum,
            weight_decay=w_weight_decay
        )
        
        # Optimizer for architecture parameters (only alpha)
        self.alpha_optimizer = optim.Adam(
            [self.model.arch_params],  # Wrap in list for optimizer
            lr=alpha_lr,
            weight_decay=alpha_weight_decay,
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.w_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer, T_max=50, eta_min=0.001
        )
        self.alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.alpha_optimizer, T_max=50, eta_min=0.0001
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def _compute_unrolled_model(self, x, target, eta, w_optimizer):
        """
        Compute unrolled model for second-order approximation
        This is computationally expensive, so we use first-order approximation by default
        """
        # Forward pass
        loss = self._loss(x, target)
        
        # Compute gradients
        gradients = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        # Update weights
        with torch.no_grad():
            unrolled_model = copy.deepcopy(self.model)
            for (name, param), grad in zip(unrolled_model.named_parameters(), gradients):
                if 'arch_params' not in name:
                    param.data = param.data - eta * grad
        
        return unrolled_model
    
    def _loss(self, x, target):
        """Compute loss"""
        logits = self.model(x)
        return self.criterion(logits, target)
    
    def _backward_step_unrolled(self, x_train, target_train, x_valid, target_valid,
                                eta, w_optimizer):
        """
        Backward step using unrolled optimization (second-order)
        """
        unrolled_model = self._compute_unrolled_model(x_train, target_train, eta, w_optimizer)
        unrolled_loss = self._loss(unrolled_model(x_valid), target_valid)
        
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_params]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        
        # Compute implicit gradients
        implicit_grads = self._hessian_vector_product(vector, x_train, target_train)
        
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        
        for v, g in zip(self.model.arch_params, dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)
    
    def _hessian_vector_product(self, vector, x, target, r=1e-2):
        """
        Compute hessian-vector product approximation
        """
        R = r / torch.cat([v.view(-1) for v in vector]).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self._loss(x, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_params)
        
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self._loss(x, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_params)
        
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    
    def _backward_step_first_order(self, x_train, target_train, x_valid, target_valid):
        """
        Backward step using first-order approximation (faster)
        """
        # Update architecture parameters
        alpha = self.model.arch_params
        logits = self.model(x_valid, alpha)
        loss = self.criterion(logits, target_valid)
        loss.backward()
    
    def step(self, x_train, target_train, x_valid, target_valid):
        """
        Perform one step of DARTS training
        
        Args:
            x_train: Training input
            target_train: Training targets
            x_valid: Validation input
            target_valid: Validation targets
        """
        # Step 1: Update model weights (w) on training set
        self.w_optimizer.zero_grad()
        logits_train = self.model(x_train)
        loss_train = self.criterion(logits_train, target_train)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.w_optimizer.step()
        
        # Step 2: Update architecture parameters (alpha) on validation set
        self.alpha_optimizer.zero_grad()
        if self.unrolled:
            self._backward_step_unrolled(
                x_train, target_train, x_valid, target_valid,
                self.w_optimizer.param_groups[0]['lr'], self.w_optimizer
            )
        else:
            self._backward_step_first_order(x_train, target_train, x_valid, target_valid)
        
        torch.nn.utils.clip_grad_norm_(self.model.arch_params, self.grad_clip)
        self.alpha_optimizer.step()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Split training data into train and val for DARTS
        # In practice, you might want to use a separate validation set
        for batch_idx, (x_train, target_train) in enumerate(self.train_loader):
            x_train = x_train.to(self.device)
            target_train = target_train.to(self.device)
            
            # Get validation batch (use next batch or cycle through)
            try:
                x_valid, target_valid = next(iter(self.val_loader))
            except:
                val_iter = iter(self.val_loader)
                x_valid, target_valid = next(val_iter)
            
            x_valid = x_valid.to(self.device)
            target_valid = target_valid.to(self.device)
            
            # DARTS step
            self.step(x_train, target_train, x_valid, target_valid)
            
            # Compute metrics
            with torch.no_grad():
                logits = self.model(x_train)
                loss = self.criterion(logits, target_train)
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += target_train.size(0)
                train_correct += predicted.eq(target_train).sum().item()
        
        # Update learning rates
        self.w_scheduler.step()
        self.alpha_scheduler.step()
        
        return {
            'train_loss': train_loss / len(self.train_loader),
            'train_acc': 100. * train_correct / train_total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, target in self.val_loader:
                x = x.to(self.device)
                target = target.to(self.device)
                
                logits = self.model(x)
                loss = self.criterion(logits, target)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        return {
            'val_loss': val_loss / len(self.val_loader),
            'val_acc': 100. * val_correct / val_total
        }
    
    def get_architecture(self) -> Dict:
        """Get the current architecture by discretizing alpha"""
        self.model.eval()
        with torch.no_grad():
            alpha = self.model.arch_params
            arch = self.model.discretize(alpha)
        return arch
    
    def train(self, num_epochs: int, print_freq: int = 10) -> Dict:
        """Train DARTS for specified number of epochs"""
        best_val_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])
            
            # Print progress
            if (epoch + 1) % print_freq == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Train Loss: {train_metrics["train_loss"]:.4f}, '
                      f'Train Acc: {train_metrics["train_acc"]:.2f}%')
                print(f'Val Loss: {val_metrics["val_loss"]:.4f}, '
                      f'Val Acc: {val_metrics["val_acc"]:.2f}%')
                
                # Print architecture
                alpha = self.model.arch_params
                alpha_softmax = torch.softmax(alpha, dim=-1)
                print(f'Architecture weights (sample): {alpha_softmax[0, 0, :5].cpu().numpy()}')
                print('-' * 50)
            
            # Save best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                best_arch = self.get_architecture()
        
        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_architecture': best_arch,
            'final_architecture': self.get_architecture()
        }

