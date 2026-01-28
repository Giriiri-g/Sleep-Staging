"""
Efficiency tracking utilities for FLOPs, MACs, and parameter counting
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_conv1d_flops(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    length: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> int:
    """Estimate FLOPs for 1D convolution"""
    # Effective kernel size with dilation
    effective_kernel = (kernel_size - 1) * dilation + 1
    output_length = (length - effective_kernel) // stride + 1
    kernel_flops = kernel_size * in_channels // groups
    flops = output_length * out_channels * kernel_flops
    return flops


def estimate_linear_flops(in_features: int, out_features: int, batch_size: int = 1) -> int:
    """Estimate FLOPs for linear layer"""
    return batch_size * in_features * out_features


def estimate_attention_flops(
    seq_length: int,
    hidden_dim: int,
    num_heads: int = 8,
    batch_size: int = 1
) -> int:
    """Estimate FLOPs for multi-head self-attention"""
    head_dim = hidden_dim // num_heads
    # Q, K, V projections: 3 * batch * seq * hidden * hidden
    proj_flops = 3 * batch_size * seq_length * hidden_dim * hidden_dim
    # Attention scores: batch * num_heads * seq * seq * head_dim
    attn_flops = batch_size * num_heads * seq_length * seq_length * head_dim
    # Output projection: batch * seq * hidden * hidden
    out_proj_flops = batch_size * seq_length * hidden_dim * hidden_dim
    return proj_flops + attn_flops + out_proj_flops


def estimate_lstm_flops(
    input_size: int,
    hidden_size: int,
    seq_length: int,
    num_layers: int = 1,
    bidirectional: bool = True,
    batch_size: int = 1
) -> int:
    """Estimate FLOPs for LSTM"""
    # LSTM has 4 gates, each with input and hidden transformations
    # Each gate: (input_size + hidden_size) * hidden_size
    gates_flops = 4 * (input_size + hidden_size) * hidden_size
    # Per timestep
    per_step = batch_size * seq_length * gates_flops
    # Multiply by layers and direction
    multiplier = num_layers * (2 if bidirectional else 1)
    return per_step * multiplier


def estimate_operation_flops(
    op_name: str,
    channels: int,
    length: int,
    batch_size: int = 1
) -> Tuple[int, int]:
    """
    Estimate FLOPs and MACs for an operation
    
    Returns:
        (flops, macs) tuple
    """
    flops = 0
    macs = 0
    
    if 'conv_1x1' in op_name:
        flops = estimate_conv1d_flops(channels, channels, 1, length)
    elif 'conv_3x1' in op_name:
        flops = estimate_conv1d_flops(channels, channels, 3, length)
    elif 'conv_5x1' in op_name:
        flops = estimate_conv1d_flops(channels, channels, 5, length)
    elif 'conv_7x1' in op_name:
        flops = estimate_conv1d_flops(channels, channels, 7, length)
    elif 'dil_conv_3x1' in op_name:
        # Dilated conv with dilation=2
        flops = estimate_conv1d_flops(channels, channels, 3, length, stride=1, groups=1, dilation=2)
    elif 'dil_conv_5x1' in op_name:
        flops = estimate_conv1d_flops(channels, channels, 5, length, stride=1, groups=1, dilation=2)
    elif 'sep_conv_3x1' in op_name:
        # Depthwise separable: depthwise + pointwise
        depthwise = estimate_conv1d_flops(channels, channels, 3, length, groups=channels)
        pointwise = estimate_conv1d_flops(channels, channels, 1, length)
        flops = depthwise + pointwise
    elif 'sep_conv_5x1' in op_name:
        depthwise = estimate_conv1d_flops(channels, channels, 5, length, groups=channels)
        pointwise = estimate_conv1d_flops(channels, channels, 1, length)
        flops = depthwise + pointwise
    elif 'disjoint_cnn_3x1' in op_name:
        # Disjoint CNN: temporal + spatial
        temporal = estimate_conv1d_flops(1, channels, 3, length)
        spatial = estimate_linear_flops(channels, channels, length)
        flops = temporal + spatial
    elif 'disjoint_cnn_5x1' in op_name:
        temporal = estimate_conv1d_flops(1, channels, 5, length)
        spatial = estimate_linear_flops(channels, channels, length)
        flops = temporal + spatial
    elif 'attention' in op_name:
        num_heads = 4  # Use fewer heads for efficiency
        flops = estimate_attention_flops(length, channels, num_heads, batch_size)
    elif 'attention_light' in op_name:
        num_heads = 2  # Lightweight attention with 2 heads
        flops = estimate_attention_flops(length, channels, num_heads, batch_size)
    elif 'lstm' in op_name:
        hidden_size = channels // 2
        flops = estimate_lstm_flops(channels, hidden_size, length, bidirectional=True)
    elif 'max_pool' in op_name or 'avg_pool' in op_name:
        # Pooling is cheap, mostly memory operations
        flops = length * channels
    elif 'fc' in op_name:
        flops = estimate_linear_flops(channels, channels, length)
    elif 'identity' in op_name:
        flops = 0
    elif 'zero' in op_name:
        flops = 0
    else:
        # Default: assume conv-like operation
        flops = estimate_conv1d_flops(channels, channels, 3, length)
    
    # MACs are approximately FLOPs / 2 (multiply-accumulate)
    macs = flops // 2
    
    return flops, macs


def estimate_architecture_efficiency(
    arch: Dict,
    input_channels: int,
    input_length: int,
    init_channels: int = 32,
    num_cells: int = 6,
    num_nodes: int = 3,
    batch_size: int = 1
) -> Dict[str, int]:
    """
    Estimate FLOPs, MACs, and parameters for an architecture
    
    Returns:
        Dictionary with 'flops', 'macs', 'params' keys
    """
    total_flops = 0
    total_macs = 0
    
    # Track channel progression
    current_channels = init_channels
    current_length = input_length // 2  # After stem
    
    # Stem network
    stem_flops = estimate_conv1d_flops(input_channels, init_channels, 7, input_length, stride=2)
    total_flops += stem_flops
    
    # Process each cell
    for cell_idx in range(num_cells):
        # Check if reduction cell
        if cell_idx in [num_cells // 3, 2 * num_cells // 3]:
            current_channels *= 2
            current_length //= 2
        
        # Process each node in cell
        if cell_idx in arch:
            cell_arch = arch[cell_idx]
            for node_idx, edge_arch in enumerate(cell_arch):
                for j, op_name in edge_arch:
                    flops, macs = estimate_operation_flops(
                        op_name, current_channels, current_length, batch_size
                    )
                    total_flops += flops
                    total_macs += macs
        else:
            # Estimate based on default operations
            for i in range(num_nodes):
                for j in range(i + 2):
                    # Assume average operation
                    flops, macs = estimate_operation_flops(
                        'conv_3x1', current_channels, current_length, batch_size
                    )
                    total_flops += flops
                    total_macs += macs
    
    # Global pooling and classifier
    final_channels = current_channels * num_nodes
    pool_flops = current_length * final_channels
    classifier_flops = estimate_linear_flops(final_channels, 20, batch_size)  # Assuming 20 classes
    total_flops += pool_flops + classifier_flops
    
    # Parameter estimation (simplified)
    # This is a rough estimate - actual count would require building the model
    estimated_params = (
        init_channels * input_channels * 7 +  # Stem
        num_cells * num_nodes * current_channels * current_channels * 3 +  # Cells
        final_channels * 20  # Classifier
    )
    
    return {
        'flops': int(total_flops),
        'macs': int(total_macs),
        'params': int(estimated_params)
    }


def check_efficiency_constraints(
    arch: Dict,
    input_channels: int,
    input_length: int,
    max_flops: Optional[int] = None,
    max_macs: Optional[int] = None,
    max_params: Optional[int] = None,
    init_channels: int = 32,
    num_cells: int = 6,
    num_nodes: int = 3
) -> Tuple[bool, Dict[str, int]]:
    """
    Check if architecture meets efficiency constraints
    
    Returns:
        (meets_constraints, efficiency_metrics)
    """
    metrics = estimate_architecture_efficiency(
        arch, input_channels, input_length, init_channels, num_cells, num_nodes
    )
    
    meets_constraints = True
    if max_flops and metrics['flops'] > max_flops:
        meets_constraints = False
    if max_macs and metrics['macs'] > max_macs:
        meets_constraints = False
    if max_params and metrics['params'] > max_params:
        meets_constraints = False
    
    return meets_constraints, metrics

