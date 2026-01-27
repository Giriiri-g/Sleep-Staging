"""
ENAS Trainer for Binary Disease Classification
Based on carpedm20/ENAS-pytorch with adaptations for:
- Binary classification with class imbalance
- Time series input
- Multiple loss functions and optimizers
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from enas_binary_classification import (
    SharedModel, Controller, FocalLoss, WeightedBCEWithLogitsLoss,
    print_dag, OP_NAMES
)



def pr_auc_and_best_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    pr_auc = auc(recall, precision)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[max(best_idx - 1, 0)] if len(thresholds) else 0.5

    return pr_auc, best_f1, best_threshold


class ENASTrainer:
    """
    ENAS Trainer following carpedm20's implementation
    
    Training Procedure (from paper):
    1. Train shared model for N steps with sampled architectures
    2. Train controller for M steps using REINFORCE
    3. Repeat alternating training
    
    Args:
        shared_model: The child network with all operations
        controller: LSTM controller that samples architectures
        train_loader: Training data
        val_loader: Validation data for computing rewards
        device: torch.device
        
        # Loss function selection
        loss_type: 'focal', 'weighted_bce', or 'bce'
        pos_weight: Weight for positive class (for weighted_bce)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        
        # Shared model optimization
        shared_optimizer: 'sgd' or 'adam'
        shared_lr: Learning rate for shared model
        shared_momentum: Momentum for SGD
        shared_weight_decay: Weight decay
        
        # Controller optimization  
        controller_optimizer: 'adam' or 'sgd'
        controller_lr: Learning rate for controller
        controller_entropy_weight: Weight for entropy regularization
        controller_baseline_decay: Decay rate for baseline (0.999 typical)
        
        # Training schedule
        shared_num_steps: Steps to train shared model per epoch
        controller_num_steps: Steps to train controller per epoch
        controller_num_aggregate: Number of samples to aggregate gradients
    """
    def __init__(self,
                 shared_model: SharedModel,
                 controller: Controller,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 # Loss function
                 loss_type: str = 'focal',
                 pos_weight: float = 5.0,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0,
                 # Shared model optimization
                 shared_optimizer: str = 'sgd',
                 shared_lr: float = 0.1,
                 shared_momentum: float = 0.9,
                 shared_weight_decay: float = 1e-4,
                 shared_grad_clip: float = 5.0,
                 # Controller optimization
                 controller_optimizer: str = 'adam',
                 controller_lr: float = 3.5e-4,
                 controller_entropy_weight: float = 1e-4,
                 controller_baseline_decay: float = 0.999,
                 controller_grad_clip: float = 5.0,
                 # Training schedule
                 shared_num_steps: int = 400,
                 controller_num_steps: int = 50,
                 controller_num_aggregate: int = 10):
        
        self.shared = shared_model.to(device)
        self.controller = controller.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create loss function
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif loss_type == 'weighted_bce':
            self.criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using Weighted BCE Loss (pos_weight={pos_weight})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print("Using standard BCE Loss")
        
        # Shared model optimizer
        if shared_optimizer == 'sgd':
            self.shared_optim = optim.SGD(
                self.shared.parameters(),
                lr=shared_lr,
                momentum=shared_momentum,
                weight_decay=shared_weight_decay,
                nesterov=True
            )
        else:
            self.shared_optim = optim.Adam(
                self.shared.parameters(),
                lr=shared_lr,
                weight_decay=shared_weight_decay
            )
        
        # Controller optimizer
        if controller_optimizer == 'adam':
            self.controller_optim = optim.Adam(
                self.controller.parameters(),
                lr=controller_lr,
                betas=(0.0, 0.999),  # Following carpedm20
                eps=1e-3
            )
        else:
            self.controller_optim = optim.SGD(
                self.controller.parameters(),
                lr=controller_lr
            )
        
        # Training parameters
        self.shared_grad_clip = shared_grad_clip
        self.controller_grad_clip = controller_grad_clip
        self.entropy_weight = controller_entropy_weight
        self.baseline_decay = controller_baseline_decay
        self.shared_num_steps = shared_num_steps
        self.controller_num_steps = controller_num_steps
        self.controller_num_aggregate = controller_num_aggregate
        
        # Baseline for REINFORCE
        self.baseline = None
        
        # Metrics tracking
        self.epoch = 0
        self.shared_step = 0
        self.controller_step = 0
    
    def train_shared(self) -> Dict[str, float]:
        """
        Train shared model for specified number of steps
        
        Following carpedm20:
        - Sample architecture from controller
        - Train shared model with sampled architecture
        - Use SGD with momentum (typical for child network)
        
        Returns:
            Dictionary with training metrics
        """
        self.shared.train()
        self.controller.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        train_iter = iter(self.train_loader)
        
        for step in range(self.shared_num_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)
            
            x = x.to(self.device)
            y = y.float().to(self.device).squeeze(-1)
            batch_size = x.size(0)
            
            # Sample architecture from controller
            with torch.no_grad():
                dag, _, _ = self.controller(batch_size=1)
            
            # Forward pass
            self.shared_optim.zero_grad()
            logits = self.shared(x, dag)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.shared.parameters(), self.shared_grad_clip)
            self.shared_optim.step()
            
            # Track metrics
            total_loss += loss.item() * batch_size
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct = (preds == y).float()
                acc = correct.mean().item() * 100  # True % accuracy
                total_acc += acc * batch_size
            total_samples += batch_size
            self.shared_step += 1
        
        return {
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples
        }
    
    def get_reward(self, dag, num_batches=10):
        self.shared.eval()

        probs, targets = [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= num_batches:
                    break

                x = x.to(self.device)
                y = y.float().to(self.device)

                logits = self.shared(x, dag)
                prob = torch.sigmoid(logits).view(-1)

                probs.append(prob.cpu().numpy())
                targets.append(y.view(-1).cpu().numpy())

        y_prob = np.concatenate(probs)
        y_true = np.concatenate(targets)

        pr_auc, best_f1, _ = pr_auc_and_best_f1(y_true, y_prob)

        reward = 0.6 * pr_auc + 0.4 * best_f1
        return reward


    def train_controller(self) -> Dict[str, float]:
        """
        Train controller using REINFORCE
        
        Following carpedm20:
        - Sample multiple architectures
        - Evaluate each to get reward (validation accuracy)
        - Update controller to maximize expected reward
        - Use baseline to reduce variance
        - Add entropy regularization for exploration
        
        Returns:
            Dictionary with controller training metrics
        """
        self.shared.eval()
        self.controller.train()
        
        total_loss = 0.0
        total_reward = 0.0
        total_entropy = 0.0
        
        for step in range(self.controller_num_steps):
            # Sample architecture
            dag, log_prob, entropy = self.controller(batch_size=1)
            
            # Get reward (validation accuracy)
            reward = self.get_reward(dag, num_batches=5)
            
            # Update baseline (exponential moving average)
            if self.baseline is None:
                self.baseline = reward
            else:
                self.baseline = self.baseline_decay * self.baseline + \
                               (1 - self.baseline_decay) * reward
            
            # Compute advantage
            advantage = reward - self.baseline
            
            # Policy gradient loss
            # Loss = -log_prob * advantage - entropy_weight * entropy
            # We want to maximize reward and entropy, so minimize negative
            loss = -log_prob * advantage - self.entropy_weight * entropy
            
            # Accumulate gradients
            if step % self.controller_num_aggregate == 0:
                self.controller_optim.zero_grad()
            
            loss.backward()
            
            # Update controller
            if (step + 1) % self.controller_num_aggregate == 0:
                nn.utils.clip_grad_norm_(
                    self.controller.parameters(), 
                    self.controller_grad_clip
                )
                self.controller_optim.step()
            
            # Track metrics
            total_loss += loss.item()
            total_reward += reward
            total_entropy += entropy.item()
            self.controller_step += 1
        
        return {
            'loss': total_loss / self.controller_num_steps,
            'reward': total_reward / self.controller_num_steps,
            'baseline': self.baseline,
            'entropy': total_entropy / self.controller_num_steps
        }
    
    def validate(self, dag: Optional[Dict] = None) -> Dict[str, float]:
        """
        Validate on full validation set
        
        Args:
            dag: Architecture to validate (None = sample from controller)
        
        Returns:
            Dictionary with validation metrics
        """
        self.shared.eval()
        
        if dag is None:
            with torch.no_grad():
                dag, _, _ = self.controller(batch_size=1)
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.float().to(self.device).squeeze(-1)
                
                logits = self.shared(x, dag)
                loss = self.criterion(logits, y)
                
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc = (preds == y).float().sum()
                
                total_loss += loss.item() * y.size(0)
                total_acc += acc.item()
                total_samples += y.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Compute additional metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        tp = np.sum((all_preds == 1) & (all_targets == 1))
        fp = np.sum((all_preds == 1) & (all_targets == 0))
        tn = np.sum((all_preds == 0) & (all_targets == 0))
        fn = np.sum((all_preds == 0) & (all_targets == 1))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'dag': dag
        }
    
    def derive_architecture(self, num_samples: int = 10) -> Tuple[Dict, float]:
        """
        Derive best architecture by sampling multiple times
        
        Following carpedm20:
        - Sample multiple architectures
        - Evaluate each on validation set
        - Return architecture with highest reward
        
        Args:
            num_samples: Number of architectures to sample
        
        Returns:
            (best_dag, best_reward)
        """
        self.shared.eval()
        self.controller.eval()
        
        best_dag = None
        best_reward = 0.0
        
        with torch.no_grad():
            for _ in range(num_samples):
                dag, _, _ = self.controller(batch_size=1)
                reward = self.get_reward(dag, num_batches=len(self.val_loader))
                
                if reward > best_reward:
                    best_reward = reward
                    best_dag = dag
        
        return best_dag, best_reward
    def search(self, num_epochs: int, print_freq: int = 1) -> Dict:
        """
        Run ENAS search

        Args:
            num_epochs: Number of search epochs
            print_freq: Print frequency

        Returns:
            Dictionary with search history
        """
        history = defaultdict(list)

        # Track best architecture by REWARD (not accuracy)
        best_reward = -float("inf")
        best_dag = None
        patience = 15
        epochs_no_improve = 0

        print("\n" + "="*60)
        print("Starting ENAS Search")
        print("="*60)

        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # 1. Train shared model
            shared_metrics = self.train_shared()

            # 2. Train controller (reward computed here)
            controller_metrics = self.train_controller()

            # 3. Validate (for logging only)
            val_metrics = self.validate()

            # Track history
            history['train_loss'].append(shared_metrics['loss'])
            history['train_acc'].append(shared_metrics['acc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['acc'])
            history['val_f1'].append(val_metrics['f1'])
            history['controller_loss'].append(controller_metrics['loss'])
            history['controller_reward'].append(controller_metrics['reward'])
            history['baseline'].append(controller_metrics['baseline'])
            history['entropy'].append(controller_metrics['entropy'])

            # ✅ FIX: track best architecture by reward (AUC)
            current_reward = controller_metrics['reward']
            if current_reward > best_reward:
                best_reward = current_reward
                best_dag = val_metrics['dag']
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            epoch_time = time.time() - start_time

            # Print progress
            if (epoch + 1) % print_freq == 0 or epoch == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
                print(f"  Shared  - Loss: {shared_metrics['loss']:.4f}, "
                    f"Acc: {shared_metrics['acc']:.4f}")
                print(f"  Val     - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
                print(f"  Val     - Prec: {val_metrics['precision']:.4f}, "
                    f"Rec: {val_metrics['recall']:.4f}")
                print(f"  Controller - Loss: {controller_metrics['loss']:.4f}, "
                    f"Reward (AUC): {current_reward:.4f}")
                print(f"  Best Reward (AUC): {best_reward:.4f}")
            if epochs_no_improve >= patience:
                print("[EARLY STOP] PR-AUC did not improve")
                break

        print("\n" + "="*60)
        print("Search Complete")
        print("="*60)
        print(f"Best Architecture Reward (AUC): {best_reward:.4f}")
        print("\nBest Architecture:")
        print_dag(best_dag)

        return {
            'history': dict(history),
            'best_reward': best_reward,
            'best_dag': best_dag
        }

def compute_class_weight(dataloader: DataLoader) -> float:
    """
    Compute class weight for imbalanced dataset
    
    Returns:
        pos_weight: num_negatives / num_positives
    """
    num_pos = 0
    num_neg = 0
    
    for _, y in dataloader:
        num_pos += y.sum().item()
        num_neg += (1 - y).sum().item()
    
    pos_weight = num_neg / max(num_pos, 1)
    
    print(f"\nClass Distribution:")
    print(f"  Positive samples: {int(num_pos)}")
    print(f"  Negative samples: {int(num_neg)}")
    print(f"  Ratio (neg/pos): {pos_weight:.2f}")
    print(f"  Recommended pos_weight: {pos_weight:.2f}")
    
    return pos_weight


if __name__ == "__main__":
    print("ENAS Trainer for Binary Disease Classification")
