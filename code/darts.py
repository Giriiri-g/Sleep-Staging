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
from pathlib import Path
from nas_search_space import Network, DARTS_OPS
from utils import print_info, print_success, print_warning, print_progress, print_key_value, print_metric


class DARTSTrainer:
    """DARTS trainer for differentiable architecture search"""
    
    def __init__(self, model: Network, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, w_lr: float = 0.025, w_momentum: float = 0.9,
                 w_weight_decay: float = 3e-4, alpha_lr: float = 3e-4,
                 alpha_weight_decay: float = 1e-3, grad_clip: float = 5.0,
                 unrolled: bool = False, checkpoint_dir: Optional[str] = None,
                 criterion: Optional[nn.Module] = None, task_type: str = "single_label",
                 pred_threshold: float = 0.5):
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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.task_type = task_type
        self.pred_threshold = pred_threshold
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Learning rate schedulers (T_max will be updated in train method)
        self.w_scheduler = None
        self.alpha_scheduler = None
        self.num_epochs = None
        
        # Loss function
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss() if task_type == "multi_label" else nn.CrossEntropyLoss()
        self.criterion = criterion
    
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
    
    def _loss(self, x, target, model=None):
        """Compute loss"""
        model = model or self.model
        logits = model(x)
        return self.criterion(logits, target)
    
    def _format_target(self, target):
        if self.task_type == "multi_label":
            return target.float()
        return target.long()
    
    def _compute_batch_accuracy(self, logits, target):
        if self.task_type == "multi_label":
            preds = (torch.sigmoid(logits) >= self.pred_threshold).float()
            sample_acc = (preds == target).float().mean(dim=1)
            return sample_acc.sum().item(), sample_acc.shape[0]
        _, predicted = logits.max(1)
        return predicted.eq(target).sum().item(), target.size(0)
    
    def _backward_step_unrolled(self, x_train, target_train, x_valid, target_valid,
                                eta, w_optimizer):
        """
        Backward step using unrolled optimization (second-order)
        """
        unrolled_model = self._compute_unrolled_model(x_train, target_train, eta, w_optimizer)
        unrolled_loss = self._loss(x_valid, target_valid, model=unrolled_model)
        
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
        if self.unrolled:
            self._backward_step_unrolled(
                x_train, target_train, x_valid, target_valid,
                self.w_optimizer.param_groups[0]['lr'], self.w_optimizer
            )
        else:
            self._backward_step_first_order(x_train, target_train, x_valid, target_valid)
        
        torch.nn.utils.clip_grad_norm_([self.model.arch_params], self.grad_clip)
        self.alpha_optimizer.step()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print_info(f"[DARTS] Starting train_epoch {epoch} "
                   f"with {len(self.train_loader)} train batches and "
                   f"{len(self.val_loader)} val batches")
        
        # Split training data into train and val for DARTS
        # In practice, you might want to use a separate validation set
        val_iter = iter(self.val_loader)
        for batch_idx, (x_train, target_train) in enumerate(self.train_loader):
            # Clear cache periodically for memory management
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print_info(f"[DARTS] Epoch {epoch} batch {batch_idx}: cleared CUDA cache")

            if batch_idx % 5 == 0:
                print_info(f"[DARTS] Epoch {epoch} batch {batch_idx}/{len(self.train_loader)} "
                           f"- got batch of shape {tuple(x_train.shape)}")

            x_train = x_train.to(self.device)
            target_train = self._format_target(target_train.to(self.device))
            
            try:
                x_valid, target_valid = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                x_valid, target_valid = next(val_iter)
            x_valid = x_valid.to(self.device)
            target_valid = self._format_target(target_valid.to(self.device))
            
            self.step(x_train, target_train, x_valid, target_valid)
            
            with torch.no_grad():
                logits = self.model(x_train)
                loss = self.criterion(logits, target_train)
                train_loss += loss.item()
                correct, total = self._compute_batch_accuracy(logits, target_train)
                train_total += total
                train_correct += correct
            
            # Delete intermediate tensors to free memory
            del x_train, target_train, x_valid, target_valid, logits, loss

        print_info(f"[DARTS] Finished train_epoch {epoch}: "
                   f"processed {train_total} samples")
        
        # Update learning rates (only if schedulers are initialized)
        if self.w_scheduler is not None:
            self.w_scheduler.step()
            self.alpha_scheduler.step()
        
        effective_total = max(1, train_total)
        return {
            'train_loss': train_loss / len(self.train_loader),
            'train_acc': 100. * train_correct / effective_total
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
                target = self._format_target(target.to(self.device))
                
                logits = self.model(x)
                loss = self.criterion(logits, target)
                val_loss += loss.item()
                
                correct, total = self._compute_batch_accuracy(logits, target)
                val_total += total
                val_correct += correct
        
        effective_total = max(1, val_total)
        return {
            'val_loss': val_loss / len(self.val_loader),
            'val_acc': 100. * val_correct / effective_total
        }
    
    def get_architecture(self) -> Dict:
        """Get the current architecture by discretizing alpha"""
        self.model.eval()
        with torch.no_grad():
            alpha = self.model.arch_params
            arch = self.model.discretize(alpha)
        return arch
    
    def save_checkpoint(self, epoch: int, history: Dict, best_val_acc: float, 
                       best_arch: Dict, checkpoint_name: str = "checkpoint_latest.pth"):
        """Save training checkpoint"""
        if self.checkpoint_dir is None:
            return
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'w_optimizer_state_dict': self.w_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'arch_params': self.model.arch_params.data,
            'history': history,
            'best_val_acc': best_val_acc,
            'best_architecture': best_arch,
        }
        # Only save scheduler states if they exist
        if self.w_scheduler is not None:
            checkpoint['w_scheduler_state_dict'] = self.w_scheduler.state_dict()
            checkpoint['alpha_scheduler_state_dict'] = self.alpha_scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print_success(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.w_optimizer.load_state_dict(checkpoint['w_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.model._arch_params.data = checkpoint['arch_params']
        
        # Only load scheduler states if they exist in checkpoint and schedulers are initialized
        if 'w_scheduler_state_dict' in checkpoint and self.w_scheduler is not None:
            self.w_scheduler.load_state_dict(checkpoint['w_scheduler_state_dict'])
            self.alpha_scheduler.load_state_dict(checkpoint['alpha_scheduler_state_dict'])
        
        print_success(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def train(self, num_epochs: int, print_freq: int = 10, 
             resume_from: Optional[str] = None, save_freq: int = 5) -> Dict:
        """Train DARTS for specified number of epochs"""
        self.num_epochs = num_epochs
        
        # Initialize schedulers if not already done
        if self.w_scheduler is None:
            self.w_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.w_optimizer, T_max=num_epochs, eta_min=0.001
            )
            self.alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.alpha_optimizer, T_max=num_epochs, eta_min=0.0001
            )
        
        start_epoch = 0
        best_val_acc = 0.0
        best_arch = None
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Load checkpoint if resuming (must initialize schedulers first)
        if resume_from and Path(resume_from).exists():
            # Initialize schedulers before loading checkpoint
            if self.w_scheduler is None:
                self.w_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.w_optimizer, T_max=num_epochs, eta_min=0.001
                )
                self.alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.alpha_optimizer, T_max=num_epochs, eta_min=0.0001
                )
            checkpoint = self.load_checkpoint(resume_from)
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_val_acc = checkpoint['best_val_acc']
            best_arch = checkpoint['best_architecture']
            print_info(f"Resuming from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            try:
                # Train
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate()
                
                # Update history
                history['train_loss'].append(train_metrics['train_loss'])
                history['train_acc'].append(train_metrics['train_acc'])
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_acc'].append(val_metrics['val_acc'])
                
                # Print progress with colored output
                if (epoch + 1) % print_freq == 0 or epoch == 0:
                    print_progress(epoch + 1, num_epochs, {
                        'Train Loss': train_metrics['train_loss'],
                        'Train Acc': f"{train_metrics['train_acc']:.2f}%",
                        'Val Loss': val_metrics['val_loss'],
                        'Val Acc': f"{val_metrics['val_acc']:.2f}%"
                    })
                
                # Save best model
                if val_metrics['val_acc'] > best_val_acc:
                    best_val_acc = val_metrics['val_acc']
                    best_arch = self.get_architecture()
                    if self.checkpoint_dir:
                        self.save_checkpoint(epoch, history, best_val_acc, best_arch, 
                                           "checkpoint_best.pth")
                
                # Save periodic checkpoint
                if self.checkpoint_dir and (epoch + 1) % save_freq == 0:
                    self.save_checkpoint(epoch, history, best_val_acc, best_arch)
                    
            except Exception as e:
                print_warning(f"Error at epoch {epoch}: {str(e)}")
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch - 1, history, best_val_acc, best_arch, 
                                       "checkpoint_error.pth")
                raise
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(num_epochs - 1, history, best_val_acc, best_arch, 
                               "checkpoint_final.pth")
        
        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_architecture': best_arch,
            'final_architecture': self.get_architecture()
        }

