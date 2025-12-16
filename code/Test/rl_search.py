"""
Reinforcement Learning based Neural Architecture Search
Uses a policy network to generate architectures and REINFORCE algorithm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from nas_search_space import Network, ALL_OPS, SEARCH_SPACE


class PolicyNetwork(nn.Module):
    """Policy network for generating architectures"""
    
    def __init__(self, num_cells: int = 8, num_nodes: int = 4, 
                 num_ops: int = None, hidden_dim: int = 128):
        """
        Initialize policy network
        
        Args:
            num_cells: Number of cells in the architecture
            num_nodes: Number of nodes in each cell
            num_ops: Number of operations in search space
            hidden_dim: Hidden dimension of policy network
        """
        super().__init__()
        if num_ops is None:
            num_ops = len(ALL_OPS)
        
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        
        # Calculate number of edges
        self.num_edges_per_cell = sum(i + 2 for i in range(num_nodes))
        self.total_edges = num_cells * self.num_edges_per_cell
        
        # Policy network: LSTM-based controller
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Embedding for operation indices
        self.op_embedding = nn.Embedding(num_ops, hidden_dim)
        
        # Output layers for each decision
        self.controllers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_ops)
            ) for _ in range(self.total_edges)
        ])
        
        # Initial hidden state
        self.h0 = nn.Parameter(torch.randn(2, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.randn(2, 1, hidden_dim))
    
    def forward(self, temperature: float = 1.0, deterministic: bool = False):
        """
        Sample an architecture from the policy
        
        Args:
            temperature: Temperature for sampling (higher = more exploration)
            deterministic: If True, return most likely actions
        
        Returns:
            actions: List of operation indices for each edge
            log_probs: Log probabilities of the actions
            entropy: Entropy of the policy
        """
        batch_size = 1
        hidden = (self.h0.expand(2, batch_size, -1).contiguous(),
                  self.c0.expand(2, batch_size, -1).contiguous())
        
        actions = []
        log_probs = []
        entropies = []
        
        # Sample operations for each edge
        for i, controller in enumerate(self.controllers):
            # Get LSTM output
            if i == 0:
                input_embed = torch.zeros(batch_size, 1, self.lstm.input_size).to(self.h0.device)
            else:
                prev_op = actions[-1]
                input_embed = self.op_embedding(prev_op).unsqueeze(1)
            
            lstm_out, hidden = self.lstm(input_embed, hidden)
            lstm_out = lstm_out.squeeze(1)
            
            # Get logits from controller
            logits = controller(lstm_out) / temperature
            
            # Sample action
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
                entropy = torch.zeros(batch_size).to(self.h0.device)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            
            actions.append(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return actions, torch.stack(log_probs), torch.stack(entropies)
    
    def get_log_probs(self, actions: List[int], temperature: float = 1.0):
        """
        Get log probabilities for given actions
        
        Args:
            actions: List of operation indices
            temperature: Temperature for logits
        
        Returns:
            log_probs: Log probabilities of the actions
            entropy: Entropy of the policy
        """
        batch_size = 1
        hidden = (self.h0.expand(2, batch_size, -1).contiguous(),
                  self.c0.expand(2, batch_size, -1).contiguous())
        
        log_probs = []
        entropies = []
        
        for i, (action, controller) in enumerate(zip(actions, self.controllers)):
            if i == 0:
                input_embed = torch.zeros(batch_size, 1, self.lstm.input_size).to(self.h0.device)
            else:
                prev_op = actions[i-1]
                input_embed = self.op_embedding(torch.tensor([prev_op]).to(self.h0.device)).unsqueeze(1)
            
            lstm_out, hidden = self.lstm(input_embed, hidden)
            lstm_out = lstm_out.squeeze(1)
            
            logits = controller(lstm_out) / temperature
            dist = torch.distributions.Categorical(logits=logits)
            
            log_prob = dist.log_prob(torch.tensor([action]).to(self.h0.device))
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return torch.stack(log_probs), torch.stack(entropies)


def actions_to_architecture(actions: List[int], num_cells: int = 8, 
                           num_nodes: int = 4) -> Dict:
    """
    Convert action sequence to architecture dictionary
    
    Args:
        actions: List of operation indices
        num_cells: Number of cells
        num_nodes: Number of nodes per cell
    
    Returns:
        Architecture dictionary
    """
    arch = {}
    idx = 0
    
    # Calculate edges per cell
    edges_per_cell = sum(i + 2 for i in range(num_nodes))
    
    for cell_idx in range(num_cells):
        cell_arch = []
        edge_idx = 0
        for node_idx in range(num_nodes):
            edge_arch = []
            # Each node connects to previous nodes (0 to node_idx+1)
            for prev_node in range(node_idx + 2):
                if idx < len(actions):
                    op_idx = actions[idx]
                    op_name = ALL_OPS[op_idx] if op_idx < len(ALL_OPS) else ALL_OPS[0]
                    edge_arch.append((prev_node, op_name))
                    idx += 1
                    edge_idx += 1
            cell_arch.append(edge_arch)
        arch[cell_idx] = cell_arch
    
    return arch


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


class RLSearchTrainer:
    """RL-based NAS trainer using REINFORCE algorithm"""
    
    def __init__(self, policy: PolicyNetwork, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 reward_type: str = 'accuracy', baseline_type: str = 'moving_average',
                 baseline_decay: float = 0.9, temperature: float = 1.0,
                 entropy_coeff: float = 0.0001, lr: float = 0.00035):
        """
        Initialize RL search trainer
        
        Args:
            policy: Policy network
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            reward_type: Type of reward ('accuracy' or 'loss')
            baseline_type: Type of baseline ('moving_average' or 'none')
            baseline_decay: Decay factor for moving average baseline
            temperature: Temperature for policy sampling
            entropy_coeff: Coefficient for entropy regularization
            lr: Learning rate for policy network
        """
        self.policy = policy.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.reward_type = reward_type
        self.baseline_type = baseline_type
        self.baseline_decay = baseline_decay
        self.temperature = temperature
        self.entropy_coeff = entropy_coeff
        
        # Optimizer for policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Baseline
        self.baseline = 0.0
        self.baseline_history = []
        
        # Training history
        self.history = {
            'rewards': [],
            'baselines': [],
            'entropies': [],
            'architectures': []
        }
    
    def evaluate_architecture(self, arch: Dict, num_epochs: int = 5,
                             lr: float = 0.025, batch_size: int = 32) -> float:
        """
        Evaluate an architecture by training it briefly
        
        Args:
            arch: Architecture dictionary
            num_epochs: Number of epochs to train
            lr: Learning rate for training
            batch_size: Batch size for training
        
        Returns:
            Validation accuracy or negative loss (depending on reward_type)
        """
        # Get number of cells and nodes from architecture
        num_cells = len(arch)
        if num_cells == 0:
            return 0.0
        num_nodes = len(arch[0]) if 0 in arch else 4
        
        # Build model from architecture
        model = architecture_to_model(
            arch, num_cells=num_cells, num_nodes=num_nodes
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        model.train()
        for epoch in range(num_epochs):
            for batch_idx, (x, target) in enumerate(self.train_loader):
                if batch_idx >= 10:  # Limit training steps for efficiency
                    break
                
                x = x.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for x, target in self.val_loader:
                x = x.to(self.device)
                target = target.to(self.device)
                
                logits = model(x)
                loss = criterion(logits, target)
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        # Return reward
        if self.reward_type == 'accuracy':
            return accuracy
        else:  # loss
            return -avg_loss
    
    def update_baseline(self, reward: float):
        """Update baseline using moving average"""
        if self.baseline_type == 'moving_average':
            if len(self.baseline_history) == 0:
                self.baseline = reward
            else:
                self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
            self.baseline_history.append(self.baseline)
        elif self.baseline_type == 'none':
            self.baseline = 0.0
    
    def train_step(self, actions: List[int], reward: float) -> Dict[str, float]:
        """
        Perform one training step using REINFORCE
        
        Args:
            actions: Action sequence (architecture)
            reward: Reward for this architecture
        
        Returns:
            Dictionary with training metrics
        """
        # Update baseline
        self.update_baseline(reward)
        
        # Compute advantage
        advantage = reward - self.baseline
        
        # Get log probabilities and entropy
        log_probs, entropies = self.policy.get_log_probs(actions, self.temperature)
        
        # Compute policy loss (REINFORCE)
        policy_loss = -(log_probs.sum() * advantage)
        
        # Entropy regularization
        entropy_loss = -self.entropy_coeff * entropies.sum()
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item(),
            'reward': reward,
            'baseline': self.baseline,
            'advantage': advantage,
            'entropy': entropies.sum().item()
        }
    
    def search(self, num_iterations: int = 100, eval_epochs: int = 5,
              print_freq: int = 10) -> Dict:
        """
        Perform architecture search
        
        Args:
            num_iterations: Number of search iterations
            eval_epochs: Number of epochs to train each architecture
            print_freq: Frequency of printing progress
        
        Returns:
            Dictionary with search results
        """
        best_reward = float('-inf')
        best_architecture = None
        
        for iteration in range(num_iterations):
            # Sample architecture from policy
            actions, log_probs, entropies = self.policy.forward(
                temperature=self.temperature,
                deterministic=False
            )
            
            # Convert to architecture
            arch = actions_to_architecture(actions)
            
            # Evaluate architecture
            reward = self.evaluate_architecture(arch, num_epochs=eval_epochs)
            
            # Update policy
            metrics = self.train_step(actions, reward)
            
            # Update history
            self.history['rewards'].append(reward)
            self.history['baselines'].append(self.baseline)
            self.history['entropies'].append(entropies.sum().item())
            self.history['architectures'].append(arch)
            
            # Update best
            if reward > best_reward:
                best_reward = reward
                best_architecture = arch
            
            # Print progress
            if (iteration + 1) % print_freq == 0 or iteration == 0:
                print(f'Iteration [{iteration+1}/{num_iterations}]')
                print(f'Reward: {reward:.4f}, Baseline: {self.baseline:.4f}, '
                      f'Advantage: {metrics["advantage"]:.4f}')
                print(f'Policy Loss: {metrics["policy_loss"]:.4f}, '
                      f'Entropy: {metrics["entropy"]:.4f}')
                print(f'Best Reward: {best_reward:.4f}')
                print('-' * 50)
        
        return {
            'best_reward': best_reward,
            'best_architecture': best_architecture,
            'history': self.history
        }

