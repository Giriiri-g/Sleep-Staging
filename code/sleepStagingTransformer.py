"""
Hierarchical Transformer Model for Sequential Spectrogram Processing

This module implements a two-tier transformer architecture that processes
sequential spectrogram-like inputs hierarchically:
1. Local-Level Encoder: Processes individual segments
2. Global-Level Encoder: Processes the sequence of segment embeddings
3. Prediction Head: Produces classification predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding module compatible with Transformer encoders.
    
    Implements the sinusoidal positional encoding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    Args:
        d_model: The dimension of the model (hidden_dim)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LocalTransformerEncoder(nn.Module):
    """
    Local-Level Transformer Encoder (Bottom Tier).
    
    Processes individual segments (S_i) of shape (time_steps, feature_dim)
    through a stack of Transformer Encoder layers to produce local embeddings.
    
    Args:
        input_dim: Number of input features per time step
        hidden_dim: Transformer model dimension (d_model)
        num_heads: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers (N_E)
        dropout: Dropout rate (default: 0.1)
        dim_feedforward: Dimension of feedforward network (default: 2048)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048
    ):
        super(LocalTransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Project input to hidden_dim if dimensions don't match
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Positional encoding for local segments
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for convenience
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a segment through the local transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dim)
            
        Returns:
            Local embedding vector of shape (batch_size, hidden_dim)
            (mean-pooled representation of the segment)
        """
        # Project input to hidden_dim
        x = self.input_projection(x)  # (batch_size, time_steps, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, time_steps, hidden_dim)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, time_steps, hidden_dim)
        
        # Mean pooling to get a single vector representation per segment
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        return x


class GlobalTransformerEncoder(nn.Module):
    """
    Global-Level Transformer Encoder (Middle Tier).
    
    Processes the sequence of local embeddings [x_1, x_2, ..., x_L]
    through a stack of Transformer Encoder layers to produce global
    contextual embeddings.
    
    Args:
        hidden_dim: Transformer model dimension (d_model)
        num_heads: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers (N_S)
        dropout: Dropout rate (default: 0.1)
        dim_feedforward: Dimension of feedforward network (default: 2048)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048
    ):
        super(GlobalTransformerEncoder, self).__init__()
        
        # Positional encoding for global segment sequence
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence of local embeddings through global transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, num_segments, hidden_dim)
            
        Returns:
            Global contextual embeddings of shape (batch_size, num_segments, hidden_dim)
        """
        # Add positional encoding to segment sequence
        x = self.pos_encoder(x)  # (batch_size, num_segments, hidden_dim)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, num_segments, hidden_dim)
        
        return x


class HierarchicalTransformerModel(nn.Module):
    """
    Hierarchical Transformer Model for Sequential Spectrogram Processing.
    
    This model processes sequential spectrogram-like inputs in a hierarchical manner:
    1. Local-Level: Each segment is processed independently through transformer encoders
    2. Global-Level: The sequence of segment embeddings is processed through transformer encoders
    3. Prediction Head: Each global embedding is classified through FC layers
    
    Args:
        input_dim: Number of input features per time step
        hidden_dim: Transformer model dimension (d_model)
        num_heads: Number of attention heads
        num_encoder_layers_local: Number of encoder layers in local transformer (N_E)
        num_encoder_layers_global: Number of encoder layers in global transformer (N_S)
        num_classes: Number of output classes per time step
        dropout: Dropout rate (default: 0.1)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        fc_hidden_dim: Hidden dimension for prediction head FC layers (default: 512)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers_local: int,
        num_encoder_layers_global: int,
        num_classes: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        fc_hidden_dim: int = 512
    ):
        super(HierarchicalTransformerModel, self).__init__()
        
        # Local-level encoder (processes individual segments)
        self.local_encoder = LocalTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers_local,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        
        # Global-level encoder (processes sequence of segment embeddings)
        self.global_encoder = GlobalTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers_global,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        
        # Prediction head (two FC layers + softmax)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hierarchical transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, num_segments, time_steps, input_dim)
               - batch_size: Number of samples in batch
               - num_segments: Number of segments (L)
               - time_steps: Number of time steps per segment
               - input_dim: Number of features per time step
        
        Returns:
            Predictions tensor of shape (batch_size, num_segments, num_classes)
            - Each segment gets a classification prediction
        """
        batch_size, num_segments, time_steps, input_dim = x.shape
        
        # Reshape for local encoder processing
        # Flatten batch and segments: (batch_size * num_segments, time_steps, input_dim)
        x = x.view(batch_size * num_segments, time_steps, input_dim)
        
        # Process each segment through local encoder
        # Output: (batch_size * num_segments, hidden_dim)
        local_embeddings = self.local_encoder(x)
        
        # Reshape back to sequence of segments
        # (batch_size, num_segments, hidden_dim)
        local_embeddings = local_embeddings.view(batch_size, num_segments, -1)
        
        # Process sequence through global encoder
        # Output: (batch_size, num_segments, hidden_dim)
        global_embeddings = self.global_encoder(local_embeddings)
        
        # Apply prediction head to each global embedding
        # Output: (batch_size, num_segments, num_classes)
        predictions = self.prediction_head(global_embeddings)
        
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions (with softmax applied).
        
        Args:
            x: Input tensor of shape (batch_size, num_segments, time_steps, input_dim)
        
        Returns:
            Probability predictions of shape (batch_size, num_segments, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (argmax of logits).
        
        Args:
            x: Input tensor of shape (batch_size, num_segments, time_steps, input_dim)
        
        Returns:
            Class predictions of shape (batch_size, num_segments)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


# Example usage and testing
if __name__ == "__main__":
    # Model parameters
    input_dim = 3000  # e.g., number of frequency bins
    hidden_dim = 256
    num_heads = 8
    num_encoder_layers_local = 4  # N_E
    num_encoder_layers_global = 2  # N_S
    num_classes = 5  # e.g., sleep stages
    
    # Create model
    model = HierarchicalTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers_local=num_encoder_layers_local,
        num_encoder_layers_global=num_encoder_layers_global,
        num_classes=num_classes,
        dropout=0.1
    )
    
    # Example input: (batch_size=2, num_segments=10, time_steps=100, input_dim=128)
    batch_size = 2
    num_segments = 10
    time_steps = 100
    x = torch.randn(batch_size, num_segments, time_steps, input_dim)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_segments}, {num_classes})")
    
    # Test prediction methods
    probas = model.predict_proba(x)
    predictions = model.predict(x)
    print(f"Probability predictions shape: {probas.shape}")
    print(f"Class predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0]}")

