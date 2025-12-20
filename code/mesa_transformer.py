"""
MESA Transformer - Multi-channel Explainable Sleep Staging Transformer
========================================================================

An explainable sequence-to-sequence multi-channel SleepTransformer-style model
tailored for triplet channels (C4-M1, Fz-Cz, Oz-Cz).

Architecture:
1. Epoch-level CNN embedding (multi-branch per channel)
2. Intra-epoch multi-channel transformer encoder (temporal + channel attention)
3. Inter-epoch transformer for context modeling
4. Classification head with uncertainty estimation
5. Explainability hooks for attention visualization

Expected Input:
    x: (batch, seq_len, num_channels, time_steps)
    - batch: batch size
    - seq_len: number of consecutive epochs (e.g., 20)
    - num_channels: number of EEG channels (e.g., 3 for C4-M1, Fz-Cz, Oz-Cz)
    - time_steps: samples per epoch (e.g., 3840 for 30s @ 128Hz)

Expected Output:
    Dictionary with:
    - 'logits': (batch, seq_len, num_classes) - classification logits
    - 'probs': (batch, seq_len, num_classes) - softmax probabilities
    - 'uncertainty': (batch, seq_len) - prediction entropy
    - 'temporal_attention': List[List] - [epoch][channel] attention maps (if return_attention=True)
    - 'channel_attention': List - [epoch] channel attention maps (if return_attention=True)
    - 'epoch_attention': (batch, nhead, seq_len, seq_len) - inter-epoch attention (if return_attention=True)

Author: Sleep Staging Team
"""

import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mesa_dataloader import create_mesa_dataloader


# ============================================================================
# Model Architecture Components
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ChannelCNN(nn.Module):
    """
    Branch-specific 1D CNN for per-channel feature extraction.
    Shared across epochs for a given channel.
    
    Architecture:
    - Conv1d(k=64, kernel=50, stride=6) → BN → GELU → MaxPool(8)
    - Conv1d(k=128, kernel=8, stride=1) → BN → GELU → MaxPool(4)
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 128,
        kernel1: int = 50,
        stride1: int = 6,
        kernel2: int = 8,
        pool1: int = 8,
        pool2: int = 4
    ):
        super(ChannelCNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel1, stride=stride1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(kernel_size=pool1, stride=pool1)
        
        # Second conv block
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=kernel2, stride=1, padding=kernel2//2)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.activation2 = nn.GELU()
        self.pool2 = nn.MaxPool1d(kernel_size=pool2, stride=pool2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time_steps) or (batch, 1, time_steps)
        Returns:
            features: (batch, num_tokens, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time_steps)
        
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        
        # Transpose to (batch, num_tokens, output_dim)
        x = x.transpose(1, 2)
        
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal self-attention transformer encoder for within-epoch time modeling.
    Processes tokens from CNN output for a single channel.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        return_attention: bool = False
    ):
        super(TemporalTransformerEncoder, self).__init__()
        self.return_attention = return_attention
        self.d_model = d_model
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, num_tokens, d_model)
        Returns:
            encoded: (batch, num_tokens, d_model)
            attention_weights: (batch, nhead, num_tokens, num_tokens) if return_attention
        """
        x = self.pos_encoder(x)
        
        if self.return_attention:
            # Run through all layers except the last
            for layer in self.transformer.layers[:-1]:
                x = layer(x)
            
            # Extract attention from last layer
            last_layer = self.transformer.layers[-1]
            attn_output, attention_weights = last_layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            # Complete the last layer forward pass
            x = x + last_layer.dropout1(attn_output)
            x = last_layer.norm1(x)
            ff_output = last_layer.linear2(last_layer.dropout(last_layer.activation(last_layer.linear1(x))))
            x = x + last_layer.dropout2(ff_output)
            x = last_layer.norm2(x)
            
            return x, attention_weights
        else:
            encoded = self.transformer(x)
            return encoded, None


class ChannelAttentionFusion(nn.Module):
    """
    Channel-wise cross-attention for fusing information from multiple channels.
    Uses multi-head self-attention over channel summaries.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        return_attention: bool = False
    ):
        super(ChannelAttentionFusion, self).__init__()
        self.return_attention = return_attention
        self.d_model = d_model
        
        # Multi-head attention for channel fusion
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        channel_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            channel_embeddings: (batch, num_channels, d_model) - channel summaries
        Returns:
            fused: (batch, num_channels, d_model)
            attention_weights: (batch, nhead, num_channels, num_channels) if return_attention
        """
        # Self-attention over channels
        if self.return_attention:
            fused, attention_weights = self.channel_attention(
                channel_embeddings, channel_embeddings, channel_embeddings,
                need_weights=True, average_attn_weights=False
            )
        else:
            fused, attention_weights = self.channel_attention(
                channel_embeddings, channel_embeddings, channel_embeddings,
                need_weights=False
            )
            # attention_weights will be None when need_weights=False
        
        fused = self.norm(fused + self.dropout(fused))
        
        return fused, attention_weights


class InterEpochTransformer(nn.Module):
    """
    Inter-epoch transformer for modeling context across epochs in a sequence.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 25,
        return_attention: bool = False
    ):
        super(InterEpochTransformer, self).__init__()
        self.return_attention = return_attention
        self.d_model = d_model
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model) - sequence of epoch embeddings
        Returns:
            encoded: (batch, seq_len, d_model)
            attention_weights: (batch, nhead, seq_len, seq_len) if return_attention
        """
        x = self.pos_encoder(x)
        
        if self.return_attention:
            # Run through all layers except the last
            for layer in self.transformer.layers[:-1]:
                x = layer(x)
            
            # Extract attention from last layer
            last_layer = self.transformer.layers[-1]
            attn_output, attention_weights = last_layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            # Complete the last layer forward pass
            x = x + last_layer.dropout1(attn_output)
            x = last_layer.norm1(x)
            ff_output = last_layer.linear2(last_layer.dropout(last_layer.activation(last_layer.linear1(x))))
            x = x + last_layer.dropout2(ff_output)
            x = last_layer.norm2(x)
            
            return x, attention_weights
        else:
            encoded = self.transformer(x)
            return encoded, None


# ============================================================================
# Main MESA Transformer Model
# ============================================================================

class MESATransformer(nn.Module):
    """
    MESA Transformer: Multi-channel Explainable Sleep Staging Transformer
    
    Architecture:
    1. Multi-branch CNN: 1D CNNs per channel (C4-M1, Fz-Cz, Oz-Cz)
    2. Intra-epoch transformer: Temporal self-attention per channel + channel fusion
    3. Inter-epoch transformer: Context modeling across epochs
    4. Classification head: Per-epoch stage prediction with uncertainty
    
    Input: (batch, seq_len, num_channels, time_steps)
    Output: Dictionary with logits, probs, uncertainty, and optional attention maps
    """
    
    def __init__(
        self,
        num_channels: int = 3,
        time_steps: int = 3840,  # 30s * 128 Hz
        seq_len: int = 20,  # Number of epochs per sequence
        d_model: int = 256,
        cnn_hidden_dim: int = 64,
        cnn_output_dim: int = 128,
        temporal_nhead: int = 8,
        temporal_num_layers: int = 2,
        channel_nhead: int = 4,
        inter_epoch_nhead: int = 8,
        inter_epoch_num_layers: int = 2,
        dim_feedforward: int = 512,
        num_classes: int = 6,  # W, N1, N2, N3, N4, REM
        dropout: float = 0.1,
        return_attention: bool = False
    ):
        super(MESATransformer, self).__init__()
        
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_classes = num_classes
        self.return_attention = return_attention
        
        # Per-channel CNN branches (shared across epochs)
        self.channel_cnns = nn.ModuleList([
            ChannelCNN(
                input_dim=1,
                hidden_dim=cnn_hidden_dim,
                output_dim=cnn_output_dim
            ) for _ in range(num_channels)
        ])
        
        # Project CNN output to d_model
        self.channel_projections = nn.ModuleList([
            nn.Linear(cnn_output_dim, d_model) for _ in range(num_channels)
        ])
        
        # Temporal transformer encoders per channel
        self.temporal_encoders = nn.ModuleList([
            TemporalTransformerEncoder(
                d_model=d_model,
                nhead=temporal_nhead,
                num_layers=temporal_num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                return_attention=return_attention
            ) for _ in range(num_channels)
        ])
        
        # Channel attention fusion
        self.channel_fusion = ChannelAttentionFusion(
            d_model=d_model,
            nhead=channel_nhead,
            dropout=dropout,
            return_attention=return_attention
        )
        
        # Project fused channels to epoch embedding dimension
        self.epoch_projection = nn.Linear(num_channels * d_model, d_model)
        
        # Inter-epoch transformer
        self.inter_epoch_transformer = InterEpochTransformer(
            d_model=d_model,
            nhead=inter_epoch_nhead,
            num_layers=inter_epoch_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=seq_len,
            return_attention=return_attention
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MESA Transformer.
        
        Args:
            x: (batch, seq_len, num_channels, time_steps)
            return_attention: Override self.return_attention if provided
        
        Returns:
            Dictionary containing:
            - 'logits': (batch, seq_len, num_classes) - classification logits
            - 'probs': (batch, seq_len, num_classes) - softmax probabilities
            - 'uncertainty': (batch, seq_len) - prediction entropy
            - 'temporal_attention': List[List] - [epoch][channel] attention maps (if return_attention)
            - 'channel_attention': List - [epoch] channel attention maps (if return_attention)
            - 'epoch_attention': (batch, nhead, seq_len, seq_len) - inter-epoch attention (if return_attention)
        """
        return_attention = return_attention if return_attention is not None else self.return_attention
        
        batch_size, seq_len, num_channels, time_steps = x.shape
        
        # Store attention weights for explainability
        temporal_attentions = []  # List of lists: [epoch][channel] -> attention map
        channel_attentions = []  # List: [epoch] -> attention map
        epoch_attention = None
        
        # Process each epoch in the sequence
        epoch_embeddings = []
        
        for t in range(seq_len):
            epoch_data = x[:, t, :, :]  # (batch, num_channels, time_steps)
            
            # Step 1: CNN feature extraction per channel
            channel_features = []
            for c in range(num_channels):
                channel_signal = epoch_data[:, c, :]  # (batch, time_steps)
                cnn_features = self.channel_cnns[c](channel_signal)  # (batch, num_tokens, cnn_output_dim)
                
                # Project to d_model
                projected = self.channel_projections[c](cnn_features)  # (batch, num_tokens, d_model)
                channel_features.append(projected)
            
            # Step 2: Temporal self-attention per channel
            channel_encoded = []
            epoch_temporal_attns = []
            for c in range(num_channels):
                encoded, attn = self.temporal_encoders[c](channel_features[c])
                channel_encoded.append(encoded)
                if return_attention:
                    epoch_temporal_attns.append(attn)
            
            if return_attention:
                temporal_attentions.append(epoch_temporal_attns)
            
            # Step 3: Create channel summaries (mean pooling or CLS token)
            channel_summaries = torch.stack([
                channel_encoded[c].mean(dim=1) for c in range(num_channels)
            ], dim=1)  # (batch, num_channels, d_model)
            
            # Step 4: Channel fusion via cross-attention
            fused_channels, ch_attn = self.channel_fusion(channel_summaries)
            if return_attention:
                channel_attentions.append(ch_attn)
            
            # Step 5: Create epoch embedding
            fused_flat = fused_channels.view(batch_size, -1)  # (batch, num_channels * d_model)
            epoch_emb = self.epoch_projection(fused_flat)  # (batch, d_model)
            epoch_embeddings.append(epoch_emb)
        
        # Stack epoch embeddings
        epoch_embeddings = torch.stack(epoch_embeddings, dim=1)  # (batch, seq_len, d_model)
        
        # Step 6: Inter-epoch transformer for context
        context_embeddings, ep_attn = self.inter_epoch_transformer(epoch_embeddings)
        if return_attention:
            epoch_attention = ep_attn
        
        # Step 7: Classification
        logits = self.classifier(context_embeddings)  # (batch, seq_len, num_classes)
        probs = F.softmax(logits, dim=-1)
        
        # Step 8: Uncertainty estimation (entropy)
        uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (batch, seq_len)
        
        # Prepare output
        output = {
            'logits': logits,
            'probs': probs,
            'uncertainty': uncertainty
        }
        
        if return_attention:
            output['temporal_attention'] = temporal_attentions  # [epoch][channel] -> attention
            output['channel_attention'] = channel_attentions  # [epoch] -> attention
            output['epoch_attention'] = epoch_attention  # (batch, nhead, seq_len, seq_len)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple prediction method that returns class predictions.
        
        Args:
            x: (batch, seq_len, num_channels, time_steps)
        
        Returns:
            predictions: (batch, seq_len) - predicted class indices
        """
        with torch.no_grad():
            output = self.forward(x, return_attention=False)
            predictions = torch.argmax(output['logits'], dim=-1)
        return predictions
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for explainability.
        
        Args:
            x: (batch, seq_len, num_channels, time_steps)
        
        Returns:
            Dictionary with attention maps for visualization
        """
        with torch.no_grad():
            output = self.forward(x, return_attention=True)
        
        return {
            'temporal_attention': output.get('temporal_attention', None),
            'channel_attention': output.get('channel_attention', None),
            'epoch_attention': output.get('epoch_attention', None)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("MESA Transformer Model")
    print("=" * 50)
    
    # Create model
    model = MESATransformer(
        num_channels=3,
        time_steps=3840,  # 30s * 128 Hz
        seq_len=20,
        d_model=256,
        num_classes=6,
        return_attention=True
    )
    
    dataloader = create_mesa_dataloader(
        preprocessed_dir=r"C:\mesa",
        annotation_dir="path/to/annotations",
        seq_len=20,
        batch_size=4,
        shuffle=True
    )
    
    output = model(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    print(f"Number of temporal attention maps: {len(output.get('temporal_attention', []))}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :5]}")
