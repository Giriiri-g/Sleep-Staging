"""
Generate attention heatmaps for MESA Transformer explainability
Visualizes temporal, channel, and inter-epoch attention weights
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from mesa_transformer import MESATransformer
from mesa_dataloader import create_mesa_dataloader


# Sleep stage class names
CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
CLASS_COLORS = {
    'W': '#3498db',    # Blue
    'N1': '#e74c3c',   # Red
    'N2': '#2ecc71',   # Green
    'N3': '#9b59b6',   # Purple
    'N4': '#f39c12',   # Orange
    'REM': '#1abc9c'   # Teal
}


def load_model(
    checkpoint_path: str,
    num_channels: int = 3,
    time_steps: int = 3840,
    seq_len: int = 20,
    num_classes: int = 6,
    device: str = "cpu"
) -> MESATransformer:
    """Load trained MESA Transformer model"""
    device = torch.device(device)
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MESATransformer(
        num_channels=num_channels,
        time_steps=time_steps,
        seq_len=seq_len,
        d_model=256,
        num_classes=num_classes,
        dropout=0.0,  # Disable dropout for inference
        return_attention=True  # Enable attention extraction
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    return model


def average_attention_heads(attention: torch.Tensor) -> np.ndarray:
    """Average attention weights across heads"""
    # attention shape: (batch, nhead, seq_len, seq_len) or (batch, nhead, num_tokens, num_tokens)
    if attention.dim() == 4:
        # Average across heads and take first batch item
        attn_avg = attention[0].mean(dim=0).cpu().numpy()
    elif attention.dim() == 2:
        # Already 2D, just convert
        attn_avg = attention[0].cpu().numpy() if attention.dim() > 2 else attention.cpu().numpy()
    else:
        attn_avg = attention.cpu().numpy()
    
    return attn_avg


def plot_temporal_attention_heatmap(
    attention: torch.Tensor,
    epoch_idx: int,
    channel_idx: int,
    channel_name: str,
    output_path: str,
    num_tokens: Optional[int] = None
) -> None:
    """
    Plot temporal attention heatmap for a specific epoch and channel
    
    Args:
        attention: (batch, nhead, num_tokens, num_tokens) attention weights
        epoch_idx: Index of the epoch
        channel_idx: Index of the channel
        channel_name: Name of the channel (e.g., "EEG1", "EEG2", "EEG3")
        output_path: Path to save the plot
        num_tokens: Number of tokens (for x/y axis labels)
    """
    attn_avg = average_attention_heads(attention)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attn_avg, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Labels
    if num_tokens:
        step = max(1, num_tokens // 10)
        tick_positions = list(range(0, num_tokens, step))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([f'T{i}' for i in tick_positions])
        ax.set_yticklabels([f'T{i}' for i in tick_positions])
    
    ax.set_xlabel('Key Position (Time)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Position (Time)', fontsize=12, fontweight='bold')
    ax.set_title(f'Temporal Attention Heatmap\nEpoch {epoch_idx}, Channel {channel_name}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal attention heatmap to {output_path}")


def plot_channel_attention_heatmap(
    attention: torch.Tensor,
    epoch_idx: int,
    channel_names: List[str],
    output_path: str
):
    """
    Plot channel attention heatmap for a specific epoch
    
    Args:
        attention: (batch, nhead, num_channels, num_channels) attention weights
        epoch_idx: Index of the epoch
        channel_names: List of channel names
        output_path: Path to save the plot
    """
    attn_avg = average_attention_heads(attention)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(attn_avg, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Set channel labels
    num_channels = len(channel_names)
    ax.set_xticks(range(num_channels))
    ax.set_yticks(range(num_channels))
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.set_yticklabels(channel_names)
    
    # Add text annotations
    for i in range(num_channels):
        for j in range(num_channels):
            text = ax.text(j, i, f'{attn_avg[i, j]:.2f}',
                          ha="center", va="center", color="black" if attn_avg[i, j] < 0.5 else "white",
                          fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Key Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Channel', fontsize=12, fontweight='bold')
    ax.set_title(f'Channel Attention Heatmap\nEpoch {epoch_idx}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved channel attention heatmap to {output_path}")


def plot_epoch_attention_heatmap(
    attention: torch.Tensor,
    predictions: np.ndarray,
    output_path: str,
    true_labels: Optional[np.ndarray] = None,
    seq_len: int = 20
):
    """
    Plot inter-epoch attention heatmap
    
    Args:
        attention: (batch, nhead, seq_len, seq_len) attention weights
        predictions: (seq_len,) predicted class indices
        output_path: Path to save the plot
        true_labels: (seq_len,) true class indices (optional)
        seq_len: Sequence length (number of epochs)
    """
    attn_avg = average_attention_heads(attention)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(attn_avg, cmap='plasma', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attention Weight')
    
    # Create epoch labels with predictions
    epoch_labels = []
    for i in range(seq_len):
        pred_class = CLASS_NAMES[predictions[i]]
        if true_labels is not None:
            true_class = CLASS_NAMES[true_labels[i]]
            label = f"E{i}\n{true_class}→{pred_class}"
        else:
            label = f"E{i}\n{pred_class}"
        epoch_labels.append(label)
    
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(epoch_labels, fontsize=9)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Key Epoch (with predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Epoch (with predictions)', fontsize=12, fontweight='bold')
    ax.set_title('Inter-Epoch Attention Heatmap\n(Attention between epochs in sequence)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved epoch attention heatmap to {output_path}")


def plot_multi_epoch_temporal_attention(
    temporal_attentions: List[List[torch.Tensor]],
    channel_names: List[str],
    output_path: str,
    num_epochs: int = 5,
    predictions: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None
):
    """
    Plot temporal attention heatmaps for multiple epochs in one image
    
    Args:
        temporal_attentions: List[List] - [epoch][channel] attention maps
        channel_names: List of channel names
        output_path: Path to save the plot
        num_epochs: Number of epochs to visualize (first N epochs)
        predictions: (seq_len,) predicted class indices (optional)
        true_labels: (seq_len,) true class indices (optional)
    """
    num_epochs = min(num_epochs, len(temporal_attentions))
    num_channels = len(channel_names)
    
    fig, axes = plt.subplots(num_epochs, num_channels, figsize=(4*num_channels, 3*num_epochs))
    
    if num_epochs == 1:
        axes = axes.reshape(1, -1)
    if num_channels == 1:
        axes = axes.reshape(-1, 1)
    
    for epoch_idx in range(num_epochs):
        # Create title for epoch row
        if predictions is not None and epoch_idx < len(predictions):
            pred_class = CLASS_NAMES[predictions[epoch_idx]]
            if true_labels is not None and epoch_idx < len(true_labels):
                true_class = CLASS_NAMES[true_labels[epoch_idx]]
                epoch_title = f"Epoch {epoch_idx} ({true_class}→{pred_class})"
            else:
                epoch_title = f"Epoch {epoch_idx} ({pred_class})"
        else:
            epoch_title = f"Epoch {epoch_idx}"
        
        for ch_idx, ch_name in enumerate(channel_names):
            ax = axes[epoch_idx, ch_idx]
            
            if (epoch_idx < len(temporal_attentions) and 
                ch_idx < len(temporal_attentions[epoch_idx]) and
                temporal_attentions[epoch_idx][ch_idx] is not None):
                
                temp_attn = average_attention_heads(temporal_attentions[epoch_idx][ch_idx])
                im = ax.imshow(temp_attn, cmap='viridis', aspect='auto', interpolation='nearest')
                
                # Add colorbar for first subplot
                if epoch_idx == 0 and ch_idx == 0:
                    plt.colorbar(im, ax=ax, label='Attention Weight')
                
                ax.set_title(f'{ch_name}', fontsize=10, fontweight='bold')
                
                # Add row title on leftmost subplot
                if ch_idx == 0:
                    ax.set_ylabel(f'{epoch_title}\nQuery Time', fontsize=10, fontweight='bold')
                else:
                    ax.set_ylabel('Query Time', fontsize=9)
                
                if epoch_idx == num_epochs - 1:
                    ax.set_xlabel('Key Time', fontsize=9)
                else:
                    ax.set_xticks([])
                    
            else:
                ax.axis('off')
    
    plt.suptitle('Temporal Attention Heatmaps (First {} Epochs)'.format(num_epochs), 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-epoch temporal attention heatmap to {output_path}")


def plot_multi_epoch_channel_attention(
    channel_attentions: List[torch.Tensor],
    channel_names: List[str],
    output_path: str,
    num_epochs: int = 5,
    predictions: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None
):
    """
    Plot channel attention heatmaps for multiple epochs in one image
    
    Args:
        channel_attentions: List - [epoch] channel attention maps
        channel_names: List of channel names
        output_path: Path to save the plot
        num_epochs: Number of epochs to visualize (first N epochs)
        predictions: (seq_len,) predicted class indices (optional)
        true_labels: (seq_len,) true class indices (optional)
    """
    num_epochs = min(num_epochs, len(channel_attentions))
    num_channels = len(channel_names)
    
    # Calculate grid dimensions
    cols = min(5, num_epochs)  # Maximum 5 columns
    rows = (num_epochs + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1) if num_epochs > 1 else axes.reshape(1, 1)
    if cols == 1:
        axes = axes.reshape(-1, 1) if rows > 1 else axes.reshape(1, 1)
    
    for epoch_idx in range(num_epochs):
        row = epoch_idx // cols
        col = epoch_idx % cols
        ax = axes[row, col] if rows > 1 or cols > 1 else axes
        
        if epoch_idx < len(channel_attentions) and channel_attentions[epoch_idx] is not None:
            ch_attn = average_attention_heads(channel_attentions[epoch_idx])
            im = ax.imshow(ch_attn, cmap='YlOrRd', aspect='auto', 
                         interpolation='nearest', vmin=0, vmax=1)
            
            # Add colorbar for first subplot
            if epoch_idx == 0:
                plt.colorbar(im, ax=ax, label='Attention Weight')
            
            # Set channel labels
            ax.set_xticks(range(num_channels))
            ax.set_yticks(range(num_channels))
            ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(channel_names, fontsize=8)
            
            # Add text annotations
            for i in range(num_channels):
                for j in range(num_channels):
                    ax.text(j, i, f'{ch_attn[i, j]:.2f}',
                           ha="center", va="center", 
                           color="black" if ch_attn[i, j] < 0.5 else "white",
                           fontsize=7, fontweight='bold')
            
            # Create title for epoch
            if predictions is not None and epoch_idx < len(predictions):
                pred_class = CLASS_NAMES[predictions[epoch_idx]]
                if true_labels is not None and epoch_idx < len(true_labels):
                    true_class = CLASS_NAMES[true_labels[epoch_idx]]
                    ax.set_title(f'Epoch {epoch_idx}\n{true_class}→{pred_class}', 
                               fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f'Epoch {epoch_idx}\n({pred_class})', 
                               fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'Epoch {epoch_idx}', fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
    
    # Hide empty subplots
    for epoch_idx in range(num_epochs, rows * cols):
        row = epoch_idx // cols
        col = epoch_idx % cols
        ax = axes[row, col] if rows > 1 or cols > 1 else axes
        ax.axis('off')
    
    plt.suptitle('Channel Attention Heatmaps (First {} Epochs)'.format(num_epochs), 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-epoch channel attention heatmap to {output_path}")


def plot_inter_epoch_attention_full(
    epoch_attention: torch.Tensor,
    predictions: np.ndarray,
    output_path: str,
    true_labels: Optional[np.ndarray] = None,
    seq_len: int = 20
):
    """
    Plot full inter-epoch attention heatmap (shows all epochs)
    
    Args:
        epoch_attention: (batch, nhead, seq_len, seq_len) attention weights
        predictions: (seq_len,) predicted class indices
        output_path: Path to save the plot
        true_labels: (seq_len,) true class indices (optional)
        seq_len: Sequence length (number of epochs)
    """
    attn_avg = average_attention_heads(epoch_attention)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(attn_avg, cmap='plasma', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attention Weight', fraction=0.046, pad=0.04)
    
    # Create epoch labels with predictions
    epoch_labels = []
    for i in range(seq_len):
        pred_class = CLASS_NAMES[predictions[i]]
        if true_labels is not None:
            true_class = CLASS_NAMES[true_labels[i]]
            label = f"E{i}\n{true_class}→{pred_class}"
        else:
            label = f"E{i}\n{pred_class}"
        epoch_labels.append(label)
    
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(epoch_labels, fontsize=9)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Key Epoch (with predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Epoch (with predictions)', fontsize=12, fontweight='bold')
    ax.set_title('Inter-Epoch Attention Heatmap\n(Attention between all epochs in sequence)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved inter-epoch attention heatmap to {output_path}")


def plot_comprehensive_attention_dashboard(
    temporal_attentions: List[List[torch.Tensor]],
    channel_attentions: List[torch.Tensor],
    epoch_attention: torch.Tensor,
    predictions: np.ndarray,
    true_labels: Optional[np.ndarray],
    channel_names: List[str],
    output_path: str,
    seq_len: int = 20,
    selected_epochs: Optional[List[int]] = None
):
    """
    Create a comprehensive dashboard with all attention heatmaps
    
    Args:
        temporal_attentions: List[List] - [epoch][channel] attention maps
        channel_attentions: List - [epoch] channel attention maps
        epoch_attention: (batch, nhead, seq_len, seq_len) inter-epoch attention
        predictions: (seq_len,) predicted class indices
        true_labels: (seq_len,) true class indices (optional)
        channel_names: List of channel names
        output_path: Path to save the plot
        seq_len: Sequence length
        selected_epochs: List of epoch indices to visualize in detail (if None, show all)
    """
    if selected_epochs is None:
        selected_epochs = list(range(min(5, seq_len)))  # Show first 5 epochs by default
    
    num_selected = len(selected_epochs)
    num_channels = len(channel_names)
    
    # Create figure with grid
    fig = plt.figure(figsize=(20, 4 + 3 * num_selected))
    gs = GridSpec(2 + num_selected, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Row 1: Inter-epoch attention (full width)
    ax_epoch = fig.add_subplot(gs[0, :])
    attn_epoch = average_attention_heads(epoch_attention)
    im = ax_epoch.imshow(attn_epoch, cmap='plasma', aspect='auto', interpolation='nearest')
    
    # Create epoch labels
    epoch_labels = []
    for i in range(seq_len):
        pred_class = CLASS_NAMES[predictions[i]]
        color = CLASS_COLORS[pred_class]
        if true_labels is not None:
            true_class = CLASS_NAMES[true_labels[i]]
            label = f"E{i}\n{true_class}→{pred_class}"
        else:
            label = f"E{i}\n{pred_class}"
        epoch_labels.append(label)
    
    ax_epoch.set_xticks(range(seq_len))
    ax_epoch.set_yticks(range(seq_len))
    ax_epoch.set_xticklabels(epoch_labels, rotation=45, ha='right', fontsize=8)
    ax_epoch.set_yticklabels(epoch_labels, fontsize=8)
    ax_epoch.set_xlabel('Key Epoch', fontsize=11, fontweight='bold')
    ax_epoch.set_ylabel('Query Epoch', fontsize=11, fontweight='bold')
    ax_epoch.set_title('Inter-Epoch Attention (Context Between Epochs)', 
                      fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax_epoch, label='Attention Weight')
    
    # For each selected epoch, show channel and temporal attention
    for row_idx, epoch_idx in enumerate(selected_epochs):
        row = row_idx + 1
        
        # Channel attention for this epoch
        ax_ch = fig.add_subplot(gs[row, 0])
        if epoch_idx < len(channel_attentions) and channel_attentions[epoch_idx] is not None:
            ch_attn = average_attention_heads(channel_attentions[epoch_idx])
            im_ch = ax_ch.imshow(ch_attn, cmap='YlOrRd', aspect='auto', 
                               interpolation='nearest', vmin=0, vmax=1)
            ax_ch.set_xticks(range(num_channels))
            ax_ch.set_yticks(range(num_channels))
            ax_ch.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
            ax_ch.set_yticklabels(channel_names, fontsize=8)
            ax_ch.set_title(f'Epoch {epoch_idx}: Channel Attention', fontsize=10, fontweight='bold')
            
            # Add text annotations
            for i in range(num_channels):
                for j in range(num_channels):
                    ax_ch.text(j, i, f'{ch_attn[i, j]:.2f}',
                              ha="center", va="center", 
                              color="black" if ch_attn[i, j] < 0.5 else "white",
                              fontsize=7, fontweight='bold')
        
        # Temporal attention for each channel (show as subplots in columns 1 and 2)
        for ch_idx, ch_name in enumerate(channel_names[:2]):  # Show first 2 channels
            ax_temp = fig.add_subplot(gs[row, ch_idx + 1])
            
            if (epoch_idx < len(temporal_attentions) and 
                ch_idx < len(temporal_attentions[epoch_idx]) and
                temporal_attentions[epoch_idx][ch_idx] is not None):
                
                temp_attn = average_attention_heads(temporal_attentions[epoch_idx][ch_idx])
                im_temp = ax_temp.imshow(temp_attn, cmap='viridis', aspect='auto', 
                                        interpolation='nearest')
                ax_temp.set_xlabel('Key Time', fontsize=9)
                ax_temp.set_ylabel('Query Time', fontsize=9)
                ax_temp.set_title(f'Epoch {epoch_idx}: Temporal Attention ({ch_name})', 
                                fontsize=10, fontweight='bold')
            else:
                ax_temp.axis('off')
    
    # Add prediction legend
    legend_elements = [mpatches.Patch(facecolor=CLASS_COLORS[name], label=name) 
                      for name in CLASS_NAMES]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              title='Sleep Stages', fontsize=9, title_fontsize=10)
    
    plt.suptitle('MESA Transformer Attention Visualization Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive attention dashboard to {output_path}")


def visualize_sample_attention(
    model: MESATransformer,
    sample_features: torch.Tensor,
    sample_labels: Optional[torch.Tensor],
    channel_names: List[str],
    output_dir: str,
    sample_idx: int = 0,
    seq_len: int = 20,
    num_epochs_to_plot: int = 5
):
    """
    Visualize attention for a single sample
    
    Args:
        model: Trained MESA Transformer model
        sample_features: (1, seq_len, num_channels, time_steps) input features
        sample_labels: (1, seq_len) true labels (optional)
        channel_names: List of channel names
        output_dir: Directory to save plots
        sample_idx: Index of the sample (for naming files)
        seq_len: Sequence length
        num_epochs_to_plot: Number of epochs to include in grouped plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions and attention
    with torch.no_grad():
        output = model(sample_features, return_attention=True)
    
    predictions = torch.argmax(output['logits'], dim=-1)[0].cpu().numpy()
    true_labels = sample_labels[0].cpu().numpy() if sample_labels is not None else None
    
    temporal_attentions = output.get('temporal_attention', [])
    channel_attentions = output.get('channel_attention', [])
    epoch_attention = output.get('epoch_attention', None)
    
    # 1. Inter-epoch attention (full sequence)
    if epoch_attention is not None:
        epoch_path = output_dir / f"inter_epoch_attention_sample_{sample_idx}.png"
        plot_inter_epoch_attention_full(
            epoch_attention, predictions, str(epoch_path),
            true_labels=true_labels, seq_len=seq_len
        )
    
    # 2. Temporal attention for first N epochs (all in one image)
    if temporal_attentions:
        temporal_path = output_dir / f"temporal_attention_first_{num_epochs_to_plot}_epochs_sample_{sample_idx}.png"
        plot_multi_epoch_temporal_attention(
            temporal_attentions, channel_names, str(temporal_path),
            num_epochs=num_epochs_to_plot,
            predictions=predictions,
            true_labels=true_labels
        )
    
    # 3. Channel attention for first N epochs (all in one image)
    if channel_attentions:
        channel_path = output_dir / f"channel_attention_first_{num_epochs_to_plot}_epochs_sample_{sample_idx}.png"
        plot_multi_epoch_channel_attention(
            channel_attentions, channel_names, str(channel_path),
            num_epochs=num_epochs_to_plot,
            predictions=predictions,
            true_labels=true_labels
        )
    
    # 4. Optional: Comprehensive dashboard (for reference)
    dashboard_path = output_dir / f"attention_dashboard_sample_{sample_idx}.png"
    plot_comprehensive_attention_dashboard(
        temporal_attentions, channel_attentions, epoch_attention,
        predictions, true_labels, channel_names,
        str(dashboard_path), seq_len=seq_len
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps from MESA Transformer")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints_mesa/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--preprocessed_dir", type=str, default=r"C:\mesa",
                       help="Directory containing preprocessed data")
    parser.add_argument("--csv_path", type=str, default="mesa_final.csv",
                       help="Path to CSV file with labels")
    parser.add_argument("--output_dir", type=str, default="attention_visualizations",
                       help="Directory to save attention visualizations")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to visualize")
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Sequence length")
    parser.add_argument("--num_channels", type=int, default=3,
                       help="Number of input channels")
    parser.add_argument("--time_steps", type=int, default=3840,
                       help="Time steps per epoch")
    parser.add_argument("--num_classes", type=int, default=6,
                       help="Number of classes")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use")
    parser.add_argument("--channel_names", type=str, nargs="+", 
                       default=["EEG1", "EEG2", "EEG3"],
                       help="Names of the channels")
    parser.add_argument("--num_epochs_to_plot", type=int, default=5,
                       help="Number of epochs to include in grouped plots")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(
        args.checkpoint_path,
        num_channels=args.num_channels,
        time_steps=args.time_steps,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Create dataloader
    dataloader = create_mesa_dataloader(
        preprocessed_dir=args.preprocessed_dir,
        csv_path=args.csv_path,
        seq_len=args.seq_len,
        batch_size=1,
        shuffle=False,
        filter_unscored=True
    )
    
    # Visualize attention for first N samples
    print(f"\nVisualizing attention for {args.num_samples} samples...")
    for batch_idx, (features, labels) in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break
        
        print(f"\nProcessing sample {batch_idx + 1}/{args.num_samples}...")
        visualize_sample_attention(
            model, features, labels, args.channel_names,
            args.output_dir, sample_idx=batch_idx, seq_len=args.seq_len,
            num_epochs_to_plot=args.num_epochs_to_plot
        )
    
    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
