"""
ENAS for Binary Disease Classification with Class Imbalance Handling

This implementation is based on carpedm20/ENAS-pytorch adapted for:
1. Binary classification (1 disease)
2. Time series input (7 channels)
3. Class imbalance handling (Focal Loss, class weights)
4. Detailed MDP formulation for RL controller
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


# ==================== FOCAL LOSS FOR CLASS IMBALANCE ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (0-1)
               Higher alpha gives more weight to positive class
        gamma: Focusing parameter (typically 2.0)
               Higher gamma focuses more on hard examples
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model (batch_size,)
            targets: Binary labels (batch_size,) with values 0 or 1
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCE Loss with class weights for imbalanced data
    
    Args:
        pos_weight: Weight for positive class
                   Set to (num_negatives / num_positives) for balance
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.device != self.pos_weight.device:
            self.pos_weight = self.pos_weight.to(inputs.device)
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight
        )


# ==================== OPERATIONS FOR TIME SERIES ====================

class SepConv1d(nn.Module):
    """Depthwise Separable Convolution - Efficient for time series"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.op = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel, stride, pad, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.op(x)


class DilConv1d(nn.Module):
    """Dilated Convolution - Larger receptive field"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 2):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.op = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.op(x)


class StdConv1d(nn.Module):
    """Standard Convolution"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        pad = (kernel - 1) // 2
        self.op = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.op(x)


class PoolBN(nn.Module):
    """Pooling with Batch Normalization"""
    def __init__(self, in_ch: int, pool_type: str = 'max'):
        super().__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        else:
            self.pool = nn.AvgPool1d(3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(in_ch)
    
    def forward(self, x):
        return self.bn(self.pool(x))


# ==================== SEARCH SPACE ====================

# Following carpedm20 design: operations for time series
OPS = {
    'sep_conv_3': lambda C: SepConv1d(C, C, 3),
    'sep_conv_5': lambda C: SepConv1d(C, C, 5),
    'sep_conv_7': lambda C: SepConv1d(C, C, 7),
    'dil_conv_3': lambda C: DilConv1d(C, C, 3, dilation=2),
    'dil_conv_5': lambda C: DilConv1d(C, C, 5, dilation=2),
    'conv_3': lambda C: StdConv1d(C, C, 3),
    'conv_5': lambda C: StdConv1d(C, C, 5),
    'max_pool': lambda C: PoolBN(C, 'max'),
    'avg_pool': lambda C: PoolBN(C, 'avg'),
    'identity': lambda C: nn.Identity(),
}

OP_NAMES = list(OPS.keys())
NUM_OPS = len(OP_NAMES)


# ==================== SHARED MODEL (CHILD NETWORK) ====================

class DAGNode(nn.Module):
    """
    A single node in the DAG
    Can take inputs from multiple previous nodes with different operations
    """
    def __init__(self, node_id: int, num_prev_nodes: int, channels: int):
        super().__init__()
        self.node_id = node_id
        self.num_prev_nodes = num_prev_nodes
        
        # Operations from each previous node
        self.ops = nn.ModuleList([
            nn.ModuleList([OPS[name](channels) for name in OP_NAMES])
            for _ in range(num_prev_nodes)
        ])
    
    def forward(self, prev_nodes: List[torch.Tensor], 
                selected_edges: List[int], selected_ops: List[int]) -> torch.Tensor:
        """
        Args:
            prev_nodes: List of previous node outputs
            selected_edges: Which previous nodes to connect from
            selected_ops: Which operations to use for each connection
        
        Returns:
            Aggregated output from selected connections
        """
        outputs = []
        for edge_idx, (prev_idx, op_idx) in enumerate(zip(selected_edges, selected_ops)):
            if prev_idx < len(prev_nodes):
                h = self.ops[edge_idx][op_idx](prev_nodes[prev_idx])
                outputs.append(h)
        
        # Average the outputs
        return sum(outputs) / len(outputs) if outputs else prev_nodes[0]


class SharedModel(nn.Module):
    """
    Shared model (child network) for ENAS
    
    Architecture:
    - Stem: Projects 7-channel input to hidden dimension
    - DAG: num_nodes nodes, each can connect to previous nodes
    - Classifier: Global pooling + linear for binary classification
    """
    def __init__(self, 
                 input_channels: int = 7,
                 input_length: int = 3000,
                 hidden_dim: int = 64,
                 num_nodes: int = 5):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Stem: 7 channels -> hidden_dim
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # DAG nodes
        # Node i can connect to nodes 0, 1, ..., i-1 (and two input nodes)
        self.nodes = nn.ModuleList([
            DAGNode(i, i + 2, hidden_dim)  # +2 for two input states
            for i in range(num_nodes)
        ])
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * num_nodes, 1)  # Binary classification
    
    def forward(self, x: torch.Tensor, dag: Dict) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, 7, length)
            dag: Architecture sampled by controller
                 dag[node_id] = {'edges': [prev_idx, ...], 'ops': [op_idx, ...]}
        
        Returns:
            Logits for binary classification (batch, 1)
        """
        # Stem
        s0 = s1 = self.stem(x)
        states = [s0, s1]
        
        # Process each node in DAG
        for node_id in range(self.num_nodes):
            node_info = dag[node_id]
            h = self.nodes[node_id](
                states,
                node_info['edges'],
                node_info['ops']
            )
            states.append(h)
        
        # Concatenate outputs from all nodes
        # Use only intermediate nodes (skip s0, s1)
        out = torch.cat(states[2:], dim=1)  # (batch, hidden_dim * num_nodes, length)
        
        # Global pooling
        out = self.global_pool(out)  # (batch, hidden_dim * num_nodes, 1)
        out = out.squeeze(-1)  # (batch, hidden_dim * num_nodes)
        
        # Classifier
        out = self.dropout(out)
        logits = self.classifier(out).squeeze(-1)  # (batch,)
        
        return logits


# ==================== RL CONTROLLER ====================

class Controller(nn.Module):
    """
    LSTM-based RL Controller for ENAS
    
    MDP Formulation:
    ================
    
    STATE SPACE (s_t):
    - Hidden state h_t of LSTM at time t
    - Embeddings of previous architecture decisions
    - For node i: we've already decided connections for nodes 0...i-1
    
    ACTION SPACE (a_t):
    For each node i in the DAG:
      1. Select previous node to connect from: {0, 1, ..., i+1}
      2. Select operation: {sep_conv_3, sep_conv_5, ..., identity}
    
    Total decisions per architecture: 
      - Node 0: 2 edges (from s0, s1) × 2 ops = 4 decisions
      - Node 1: 3 edges (from s0, s1, node0) × 2 ops = 6 decisions
      - Node i: (i+2) edges × 2 ops = 2(i+2) decisions
    
    REWARD (R):
    - Validation accuracy after training with sampled architecture
    - Higher accuracy = higher reward
    
    TRANSITION:
    - Deterministic: s_{t+1} = LSTM(s_t, a_t)
    - Next state is LSTM hidden state after processing current action
    
    POLICY π(a|s; θ):
    - Parameterized by LSTM parameters θ
    - π(prev_node | s) = softmax(W_attn * h_t)
    - π(op | s) = softmax(W_op * h_t)
    
    OBJECTIVE:
    - Maximize: J(θ) = E_{a~π(·|s;θ)}[R(a)]
    - Using REINFORCE: ∇J(θ) = E[(R - b) * ∇log π(a|s;θ)]
    - Baseline b: exponential moving average of rewards
    """
    def __init__(self,
                 num_nodes: int = 5,
                 num_ops: int = NUM_OPS,
                 hidden_dim: int = 100,
                 temperature: float = 5.0,
                 tanh_c: float = 2.5):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.tanh_c = tanh_c
        
        # LSTM controller
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Embeddings
        self.g_emb = nn.Embedding(1, hidden_dim)  # Starting token
        
        # For selecting previous nodes (varies per node)
        self.w_emb = nn.ModuleList([
            nn.Embedding(i + 2, hidden_dim)  # Node i can connect to i+2 nodes
            for i in range(num_nodes)
        ])
        
        # For selecting operations
        self.op_emb = nn.Embedding(num_ops, hidden_dim)
        
        # Attention for selecting previous nodes
        self.w_attn = nn.Linear(hidden_dim, 1, bias=False)
        
        # Linear for selecting operations
        self.w_soft = nn.Linear(hidden_dim, num_ops, bias=False)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def forward(self, batch_size: int = 1) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Sample an architecture from the controller
        
        Returns:
            dag: Sampled architecture
            log_prob: Log probability of sampled architecture
            entropy: Entropy of the sampling distribution
        """
        device = next(self.parameters()).device
        
        # Initialize LSTM state
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Starting input
        inputs = self.g_emb.weight.unsqueeze(0)  # (1, hidden_dim)
        
        dag = {}
        log_probs = []
        entropies = []
        
        # Sample architecture for each node
        for node_id in range(self.num_nodes):
            num_prev_nodes = node_id + 2  # Can connect to node_id+2 previous nodes
            
            # Each node connects to 2 previous nodes (following carpedm20)
            num_edges = 2
            edges = []
            ops = []
            
            for edge_id in range(num_edges):
                # Step 1: Select which previous node to connect from
                h, c = self.lstm(inputs.squeeze(0), (h, c))
                
                # Compute logits for previous node selection
                logits = self.w_attn(h.unsqueeze(1))  # (batch, 1, 1)
                logits = logits.squeeze(-1) / self.temperature  # (batch, 1)
                logits = self.tanh_c * torch.tanh(logits)
                
                # Only consider valid previous nodes
                if num_prev_nodes == 1:
                    prev_node = torch.zeros(batch_size, dtype=torch.long, device=device)
                    log_prob_prev = torch.zeros(batch_size, device=device)
                    ent_prev = torch.zeros(batch_size, device=device)
                else:
                    # Sample previous node
                    logits_prev = logits.expand(batch_size, num_prev_nodes)
                    probs_prev = F.softmax(logits_prev, dim=-1)
                    dist_prev = torch.distributions.Categorical(probs_prev)
                    prev_node = dist_prev.sample()
                    log_prob_prev = dist_prev.log_prob(prev_node)
                    ent_prev = dist_prev.entropy()
                
                edges.append(prev_node.item())
                log_probs.append(log_prob_prev)
                entropies.append(ent_prev)
                
                # Embed selected previous node
                inputs = self.w_emb[node_id](prev_node).unsqueeze(1)
                
                # Step 2: Select which operation to use
                h, c = self.lstm(inputs.squeeze(0), (h, c))
                
                # Compute operation logits
                logits_op = self.w_soft(h) / self.temperature
                logits_op = self.tanh_c * torch.tanh(logits_op)
                
                # Sample operation
                probs_op = F.softmax(logits_op, dim=-1)
                dist_op = torch.distributions.Categorical(probs_op)
                op = dist_op.sample()
                log_prob_op = dist_op.log_prob(op)
                ent_op = dist_op.entropy()
                
                ops.append(op.item())
                log_probs.append(log_prob_op)
                entropies.append(ent_op)
                
                # Embed selected operation for next step
                inputs = self.op_emb(op).unsqueeze(1)
            
            dag[node_id] = {'edges': edges, 'ops': ops}
        
        # Aggregate log probabilities and entropies
        total_log_prob = torch.stack(log_probs).sum()
        total_entropy = torch.stack(entropies).sum()
        
        return dag, total_log_prob, total_entropy


def print_dag(dag: Dict):
    """Print sampled architecture in readable format"""
    print("\n" + "="*60)
    print("Sampled Architecture (DAG):")
    print("="*60)
    for node_id, node_info in dag.items():
        print(f"Node {node_id}:")
        for edge_idx, (prev_idx, op_idx) in enumerate(
            zip(node_info['edges'], node_info['ops'])
        ):
            op_name = OP_NAMES[op_idx]
            print(f"  Edge {edge_idx}: from node {prev_idx} via {op_name}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("ENAS for Binary Disease Classification")
    print(f"Number of operations: {NUM_OPS}")
    print(f"Operations: {OP_NAMES}")
    
    # Create models
    shared_model = SharedModel(
        input_channels=7,
        hidden_dim=64,
        num_nodes=5
    )
    
    controller = Controller(
        num_nodes=5,
        num_ops=NUM_OPS,
        hidden_dim=100
    )
    
    print(f"\nShared Model parameters: {sum(p.numel() for p in shared_model.parameters()):,}")
    print(f"Controller parameters: {sum(p.numel() for p in controller.parameters()):,}")
    
    # Sample architecture
    dag, log_prob, entropy = controller(batch_size=1)
    print_dag(dag)
    print(f"Log probability: {log_prob.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    
    # Test forward pass
    x = torch.randn(2, 7, 3000)
    logits = shared_model(x, dag)
    print(f"\nInput shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits: {logits}")
