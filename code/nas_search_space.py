
"""
Neural Architecture Search Space for Time Series Classification
Defines operations and search space for sleep stage classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class Conv1dOperation(nn.Module):
    """1D Convolution operation with configurable parameters"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: Optional[int] = None, dilation: int = 1,
                 groups: int = 1, use_bn: bool = True, activation: str = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, dilation=dilation,
                             groups=groups, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DisjointCNNOperation(nn.Module):
    """1+1D Convolution operation (Disjoint CNN) with configurable parameters
    
    DisjointCNN applies convolutions in two phases:
    1. Temporal convolution: (kernel_size, 1) - learns temporal patterns
    2. Spatial convolution: (1, n_channels) - learns channel interactions
    
    This separates temporal and spatial feature learning for multivariate time series.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: Optional[int] = None, dilation: int = 1,
                 use_bn: bool = True, activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        # Temporal Convolution: (kernel_size, 1)
        # Operates on time dimension, treats each channel independently
        self.temporal_conv = nn.Conv2d(
            in_channels=1,  # Process one channel at a time across time
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            bias=not use_bn
        )
        
        self.temporal_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Activation after temporal convolution
        if activation == 'relu':
            self.temporal_activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.temporal_activation = nn.GELU()
        elif activation == 'swish':
            self.temporal_activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.temporal_activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.temporal_activation = nn.ELU(inplace=True)
        else:
            self.temporal_activation = nn.Identity()
        
        # Spatial convolution will be applied dynamically based on input channels
        # It's created in forward pass to handle variable channel dimensions
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.activation_type = activation
        
        # Spatial BN and activation
        self.spatial_bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.spatial_activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.spatial_activation = nn.GELU()
        elif activation == 'swish':
            self.spatial_activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.spatial_activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.spatial_activation = nn.ELU(inplace=True)
        else:
            self.spatial_activation = nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Cache for spatial conv layer
        self.spatial_conv = None
        self._last_spatial_dim = None
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, in_channels, length)
        
        Returns:
            Output tensor of shape (batch, out_channels, length')
        """
        # Reshape from (B, C, L) to (B, 1, L, C) for 2D convolution
        # This allows temporal conv to operate on time dimension
        batch_size, in_channels, length = x.shape
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = x.unsqueeze(1)  # (B, 1, L, C)
        
        # Phase 1: Temporal Convolution (kernel_size, 1)
        x = self.temporal_conv(x)  # (B, out_channels, L', C)
        x = self.temporal_bn(x)
        x = self.temporal_activation(x)
        
        # Get current spatial dimension (number of channels)
        current_spatial_dim = x.shape[3]
        
        # Phase 2: Spatial Convolution (1, n_channels)
        # Create spatial conv layer if needed or if channel dimension changed
        if self.spatial_conv is None or self._last_spatial_dim != current_spatial_dim:
            self.spatial_conv = nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=(1, current_spatial_dim),
                padding='valid',
                bias=not self.use_bn
            ).to(x.device)
            self._last_spatial_dim = current_spatial_dim
        
        x = self.spatial_conv(x)  # (B, out_channels, L', 1)
        x = self.spatial_bn(x)
        x = self.spatial_activation(x)
        x = self.dropout(x)
        
        # Reshape back to (B, out_channels, L')
        x = x.squeeze(3)  # Remove spatial dimension
        
        return x


class LSTMOperation(nn.Module):
    """LSTM operation for sequence modeling"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # First convert to sequence format (B, C, T) -> (B, T, C)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.projection = nn.Linear(output_size, input_size)
    
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.lstm(x)
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        return x


class AttentionOperation(nn.Module):
    """Multi-head self-attention operation"""
    def __init__(self, in_channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Ensure num_heads divides in_channels, adjust if needed
        if in_channels < num_heads:
            num_heads = 1
        while in_channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.permute(0, 2, 1)  # (B, T, C)
        
        residual = x
        x = self.norm(x)
        
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = out + residual
        
        out = out.permute(0, 2, 1)  # (B, C, T)
        return out


class FullyConnectedOperation(nn.Module):
    """Fully Connected (Dense) operation with configurable parameters
    
    Can be used for:
    - Classification heads
    - Projection layers
    - Feature transformation
    - Dimensionality reduction/expansion
    """
    def __init__(self, in_features: int, out_features: int, 
                 use_bn: bool = True, activation: str = 'relu',
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        
        self.fc = nn.Linear(in_features, out_features, bias=bias if not use_bn else False)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, in_features) or (batch, in_features, length)
               If 3D, applies global average pooling first
        
        Returns:
            Output tensor of shape (batch, out_features)
        """
        # Handle 3D input from conv layers
        if x.dim() == 3:
            x = x.mean(dim=2)  # Global average pooling: (B, C, L) -> (B, C)
        
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: Optional[int] = None, activation: str = 'relu'):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TemporalPooling(nn.Module):
    """Temporal pooling operation with padding to maintain size"""
    def __init__(self, pool_type: str = 'avg', kernel_size: int = 2):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        if pool_type == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride=1, padding=(kernel_size-1)//2)
        elif pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride=1, padding=(kernel_size-1)//2)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
    
    def forward(self, x):
        return self.pool(x)


class IdentityOperation(nn.Module):
    """Identity operation (skip connection)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class ZeroOperation(nn.Module):
    """Zero operation (no connection)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.zeros_like(x)


# Search Space Configuration
SEARCH_SPACE = {
    'conv_1x1': lambda C: Conv1dOperation(C, C, kernel_size=1, activation='relu'),
    'conv_3x1': lambda C: Conv1dOperation(C, C, kernel_size=3, activation='relu'),
    'conv_5x1': lambda C: Conv1dOperation(C, C, kernel_size=5, activation='relu'),
    'conv_7x1': lambda C: Conv1dOperation(C, C, kernel_size=7, activation='relu'),
    'dil_conv_3x1': lambda C: Conv1dOperation(C, C, kernel_size=3, dilation=2, activation='relu'),
    'dil_conv_5x1': lambda C: Conv1dOperation(C, C, kernel_size=5, dilation=2, activation='relu'),
    'sep_conv_3x1': lambda C: DepthwiseSeparableConv(C, C, kernel_size=3, activation='relu'),
    'sep_conv_5x1': lambda C: DepthwiseSeparableConv(C, C, kernel_size=5, activation='relu'),
    'max_pool_3x1': lambda C: TemporalPooling('max', kernel_size=3),
    'avg_pool_3x1': lambda C: TemporalPooling('avg', kernel_size=3),
    'lstm': lambda C: LSTMOperation(C, C // 2, num_layers=1, bidirectional=True),
    'attention': lambda C: AttentionOperation(C, num_heads=4),  # Reduced heads for efficiency
    'attention_light': lambda C: AttentionOperation(C, num_heads=2),  # Lightweight attention
    'identity': lambda C: IdentityOperation(),
    'zero': lambda C: ZeroOperation(),
    'disjoint_cnn_3x1': lambda C: DisjointCNNOperation(C, C, kernel_size=3, activation='relu'),
    'disjoint_cnn_5x1': lambda C: DisjointCNNOperation(C, C, kernel_size=5, activation='relu'),
    'fc': lambda C: FullyConnectedOperation(C, C, use_bn=True, activation='relu'),
}

# Operations for DARTS (excluding zero for intermediate nodes)
# Prioritize efficient operations, exclude memory-intensive ones for small GPUs
DARTS_OPS = [
    'conv_1x1', 'conv_3x1', 'conv_5x1',
    'sep_conv_3x1', 'sep_conv_5x1',  # Efficient separable convolutions
    'dil_conv_3x1', 'dil_conv_5x1',  # Dilated convolutions
    # 'disjoint_cnn_3x1', 'disjoint_cnn_5x1',  # Disjoint CNN - memory intensive, disabled for 4GB GPU
    'attention_light',  # Lightweight attention only (2 heads)
    'max_pool_3x1', 'avg_pool_3x1',  # Pooling
    # 'lstm',  # LSTM - memory intensive, disabled for 4GB GPU
    'identity'  # Skip connection
]

# All operations for RL search
ALL_OPS = list(SEARCH_SPACE.keys())


class Cell(nn.Module):
    """Cell structure for NAS (DARTS cell)"""
    def __init__(self, num_nodes: int, channels: int, search_space: List[str],
                 is_reduction: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.search_space = search_space
        self.is_reduction = is_reduction
        
        # Preprocessing layers
        self.preprocess0 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels)
        )
        self.preprocess1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels)
        )
        
        # Operations between nodes
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i + 2):
                op_dict = nn.ModuleDict()
                for op_name in search_space:
                    op_dict[op_name] = SEARCH_SPACE[op_name](channels)
                self.ops.append(op_dict)
    
    def forward(self, s0, s1, alpha=None):
        """
        Forward pass through cell
        Args:
            s0, s1: Input states
            alpha: Architecture parameters (for DARTS). If None, use uniform weights
        """
        states = [self.preprocess0(s0), self.preprocess1(s1)]
        
        offset = 0
        for i in range(self.num_nodes):
            s = []
            for j in range(i + 2):
                idx = offset + j
                if alpha is not None:
                    # DARTS: weighted sum of operations
                    weights = F.softmax(alpha[idx], dim=-1)
                    # Apply each operation and ensure same temporal dimension
                    op_outputs = []
                    for w, op in zip(weights, self.ops[idx].values()):
                        op_out = op(states[j])
                        # Handle 2D outputs (e.g., from FullyConnectedOperation)
                        if op_out.dim() == 2:
                            # Reshape to 3D: (B, C) -> (B, C, 1) then expand to match temporal dim
                            op_out = op_out.unsqueeze(2)  # (B, C, 1)
                            op_out = op_out.expand(-1, -1, states[j].shape[2])  # (B, C, T)
                        # Ensure temporal dimension matches input
                        elif op_out.dim() == 3 and op_out.shape[2] != states[j].shape[2]:
                            # Use adaptive pooling or interpolation to match size
                            op_out = F.interpolate(
                                op_out, 
                                size=states[j].shape[2], 
                                mode='linear', 
                                align_corners=False
                            )
                        op_outputs.append(w * op_out)
                    x = sum(op_outputs)
                else:
                    # Uniform average (for initialization)
                    op_outputs = []
                    for op in self.ops[idx].values():
                        op_out = op(states[j])
                        # Handle 2D outputs (e.g., from FullyConnectedOperation)
                        if op_out.dim() == 2:
                            # Reshape to 3D: (B, C) -> (B, C, 1) then expand to match temporal dim
                            op_out = op_out.unsqueeze(2)  # (B, C, 1)
                            op_out = op_out.expand(-1, -1, states[j].shape[2])  # (B, C, T)
                        # Ensure temporal dimension matches
                        elif op_out.dim() == 3 and op_out.shape[2] != states[j].shape[2]:
                            op_out = F.interpolate(
                                op_out,
                                size=states[j].shape[2],
                                mode='linear',
                                align_corners=False
                            )
                        op_outputs.append(op_out)
                    x = sum(op_outputs) / len(op_outputs)
                s.append(x)
            states.append(sum(s))
            offset += i + 2
        
        return torch.cat(states[-self.num_nodes:], dim=1)


class Network(nn.Module):
    """Supernet for NAS"""
    def __init__(self, input_channels: int = 7, input_length: int = 3000,
                 num_classes: int = 7, init_channels: int = 64, num_cells: int = 8,
                 num_nodes: int = 4, search_space: List[str] = None):
        super().__init__()
        if search_space is None:
            search_space = DARTS_OPS
        
        self.input_channels = input_channels
        self.input_length = input_length
        self.num_classes = num_classes
        self.init_channels = init_channels
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.search_space = search_space
        
        # Stem network
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, init_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        # Calculate channels after stem
        channels = init_channels
        current_length = input_length // 2
        
        # Normal cells
        self.cells = nn.ModuleList()
        self.reduce_indices = [num_cells // 3, 2 * num_cells // 3]
        
        # Input and output projections for cells
        # We need separate projections for s0 and s1 as they may have different channel counts
        self.input_projections_s0 = nn.ModuleList()
        self.input_projections_s1 = nn.ModuleList()
        self.output_projections = nn.ModuleList()
        
        # Track channel counts for s0 and s1
        # Initially both come from stem
        s0_channels = init_channels
        s1_channels = init_channels
        
        for i in range(num_cells):
            if i in self.reduce_indices:
                # Reduction cell - doubles channels
                cell_channels = channels * 2
                channels = cell_channels
                current_length = current_length // 2
                cell = Cell(num_nodes, cell_channels, search_space, is_reduction=True)
            else:
                # Normal cell
                cell_channels = channels
                cell = Cell(num_nodes, channels, search_space, is_reduction=False)
            
            self.cells.append(cell)
            
            # Input projections for s0 and s1
            # Both need to be projected to cell_channels
            if s0_channels != cell_channels:
                input_proj_s0 = nn.Sequential(
                    nn.Conv1d(s0_channels, cell_channels, 1),
                    nn.BatchNorm1d(cell_channels)
                )
            else:
                input_proj_s0 = nn.Identity()
                
            if s1_channels != cell_channels:
                input_proj_s1 = nn.Sequential(
                    nn.Conv1d(s1_channels, cell_channels, 1),
                    nn.BatchNorm1d(cell_channels)
                )
            else:
                input_proj_s1 = nn.Identity()
            
            self.input_projections_s0.append(input_proj_s0)
            self.input_projections_s1.append(input_proj_s1)
            
            # Update channel counts for next iteration
            # After this cell: s0 = old s1, s1 = cell output
            s0_channels = s1_channels
            s1_channels = cell_channels * num_nodes  # Cell outputs concatenated features
            
            # Output projection: map from cell output to next cell's expected input
            # Cell outputs cell_channels * num_nodes, but projection is handled by next cell's input projection
            output_proj = nn.Identity()
            self.output_projections.append(output_proj)
        
        # Global pooling and classifier
        # After last cell, we have channels * num_nodes features (from concatenation)
        final_channels = channels * num_nodes
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(final_channels, num_classes)
        
        # Initialize architecture parameters for DARTS
        num_edges = sum(i + 2 for i in range(num_nodes))
        self._arch_params = nn.Parameter(
            1e-3 * torch.randn(num_cells, num_edges, len(search_space))
        )
    
    @property
    def arch_params(self):
        return self._arch_params
    
    def forward(self, x, alpha=None):
        """
        Forward pass
        Args:
            x: Input tensor (B, C, T)
            alpha: Architecture parameters. If None, use self._arch_params
        """
        x = self.stem(x)
        s0 = s1 = x
        
        if alpha is None:
            alpha = self.arch_params
        
        for i, cell in enumerate(self.cells):
            # Project inputs to match cell's expected channel size
            s0_proj = self.input_projections_s0[i](s0)
            s1_proj = self.input_projections_s1[i](s1)
            
            # Cell takes two inputs and outputs concatenated features
            cell_output = cell(s0_proj, s1_proj, alpha[i])
            
            # Apply output projection if needed
            cell_output = self.output_projections[i](cell_output)
            
            # Update states for next iteration
            s0, s1 = s1, cell_output
        
        x = s1
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def discretize(self, alpha=None):
        """Discretize architecture by selecting top operations"""
        if alpha is None:
            alpha = self.arch_params
        
        arch = {}
        for cell_idx in range(self.num_cells):
            cell_arch = []
            edge_idx = 0
            for i in range(self.num_nodes):
                edge_arch = []
                for j in range(i + 2):
                    weights = F.softmax(alpha[cell_idx, edge_idx], dim=-1)
                    top_op_idx = weights.argmax().item()
                    top_op = self.search_space[top_op_idx]
                    edge_arch.append((j, top_op))
                    edge_idx += 1
                cell_arch.append(edge_arch)
            arch[cell_idx] = cell_arch
        
        return arch

