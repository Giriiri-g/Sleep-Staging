"""
Comorbidity Classifier Model
============================
Neural network model for multiclass binary comorbidity classification.

Architecture:
- Embedding layer for sleep stage sequences (0-5)
- RNN (LSTM/GRU) to process variable-length sequences
- Combine RNN output with other PSG features
- Prediction head with 3 binary outputs (sigmoid for multiclass binary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ComorbidityClassifier(nn.Module):
    """
    Comorbidity Classifier for sleep disorders.
    
    Inputs:
    - sleep_stages: (batch, seq_len) - sequence of sleep stage labels (0-5)
    - seq_lengths: (batch,) - actual sequence lengths (for packing)
    - features: (batch, num_features) - other PSG features
    
    Outputs:
    - (batch, 3) - binary predictions for [insomnia, restless leg, apnea]
    """
    
    def __init__(
        self,
        num_sleep_stages: int = 6,  # 0-5: Wake, N1, N2, N3, REM, Movement
        embedding_dim: int = 32,
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: str = 'LSTM',  # 'LSTM' or 'GRU'
        rnn_dropout: float = 0.2,
        num_features: int = 12,  # Number of other PSG features
        combined_hidden_dim: int = 256,
        num_outputs: int = 3,  # insomnia, restless leg, apnea
        dropout: float = 0.3,
        use_bidirectional: bool = True
    ):
        super(ComorbidityClassifier, self).__init__()
        
        self.num_sleep_stages = num_sleep_stages
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.use_bidirectional = use_bidirectional
        
        # Embedding layer for sleep stages
        self.sleep_stage_embedding = nn.Embedding(
            num_embeddings=num_sleep_stages,
            embedding_dim=embedding_dim,
            padding_idx=0  # 0 is Wake, can be used as padding
        )
        
        # RNN for processing sleep stage sequences
        rnn_class = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            bidirectional=use_bidirectional,
            batch_first=True
        )
        
        # RNN output dimension (bidirectional doubles it)
        rnn_output_dim = rnn_hidden_dim * 2 if use_bidirectional else rnn_hidden_dim
        
        # Feature projection for other PSG features
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, rnn_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Combined feature dimension
        combined_input_dim = rnn_output_dim + (rnn_output_dim // 2)
        
        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(combined_input_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_dim, combined_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(combined_hidden_dim // 2, num_outputs)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embedding
        nn.init.normal_(self.sleep_stage_embedding.weight, mean=0, std=0.1)
        
        # Initialize RNN
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    if self.rnn_type == 'LSTM':
                        param.data[(n // 4):(n // 2)].fill_(1)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        sleep_stages: torch.Tensor,
        seq_lengths: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sleep_stages: (batch, max_seq_len) - padded sleep stage sequences
            seq_lengths: (batch,) - actual sequence lengths
            features: (batch, num_features) - other PSG features
        
        Returns:
            logits: (batch, num_outputs) - raw logits for each comorbidity
        """
        batch_size = sleep_stages.size(0)
        
        # Embed sleep stages
        # (batch, max_seq_len) -> (batch, max_seq_len, embedding_dim)
        embedded = self.sleep_stage_embedding(sleep_stages)
        
        # Pack sequences for efficient RNN processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # RNN forward pass
        packed_output, hidden = self.rnn(packed)
        
        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        # Get the last valid output for each sequence
        # output shape: (batch, max_seq_len, rnn_hidden_dim * num_directions)
        rnn_output_dim = output.size(-1)
        
        # Extract last valid timestep for each sequence
        rnn_features = torch.zeros(batch_size, rnn_output_dim, device=sleep_stages.device)
        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            if seq_len > 0:
                rnn_features[i] = output[i, seq_len - 1]
        
        # Process other features
        feature_proj = self.feature_projection(features)  # (batch, rnn_output_dim // 2)
        
        # Combine RNN output and feature projection
        combined = torch.cat([rnn_features, feature_proj], dim=1)  # (batch, combined_input_dim)
        
        # Classification head
        logits = self.classifier(combined)  # (batch, num_outputs)
        
        return logits
    
    def predict_proba(
        self,
        sleep_stages: torch.Tensor,
        seq_lengths: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability predictions (sigmoid for multiclass binary).
        
        Returns:
            probs: (batch, num_outputs) - probabilities for each comorbidity
        """
        logits = self.forward(sleep_stages, seq_lengths, features)
        return torch.sigmoid(logits)
    
    def predict(
        self,
        sleep_stages: torch.Tensor,
        seq_lengths: torch.Tensor,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            threshold: Probability threshold for positive prediction
        
        Returns:
            predictions: (batch, num_outputs) - binary predictions
        """
        probs = self.predict_proba(sleep_stages, seq_lengths, features)
        return (probs >= threshold).long()

