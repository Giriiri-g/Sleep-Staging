"""
Extract Features from Sleep Stage Sequences
===========================================
Extract statistical and pattern features from sleep stage sequences
for use with traditional ML classifiers.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def extract_sequence_features(sequences: List[np.ndarray]) -> np.ndarray:
    """
    Extract features from sleep stage sequences.
    
    Args:
        sequences: List of sleep stage arrays (each is array of 0-5)
    
    Returns:
        features: (n_samples, n_features) array of extracted features
    """
    features_list = []
    
    for seq in sequences:
        if len(seq) == 0:
            # Empty sequence - return zeros
            features_list.append(np.zeros(50))
            continue
        
        seq = seq.astype(float)
        features = []
        
        # Basic statistics
        features.append(len(seq))  # Sequence length
        features.append(np.mean(seq))  # Mean stage
        features.append(np.std(seq))  # Std stage
        features.append(np.median(seq))  # Median stage
        features.append(np.min(seq))  # Min stage
        features.append(np.max(seq))  # Max stage
        
        # Stage distribution (counts and percentages)
        for stage in range(6):
            count = np.sum(seq == stage)
            features.append(count)  # Count
            features.append(count / len(seq))  # Percentage
        
        # Transitions (stage changes)
        if len(seq) > 1:
            transitions = np.diff(seq)
            features.append(np.sum(transitions != 0))  # Number of transitions
            features.append(np.mean(np.abs(transitions)))  # Mean transition magnitude
            features.append(np.std(transitions))  # Std of transitions
            
            # Count transitions between specific stages
            for from_stage in range(6):
                for to_stage in range(6):
                    if from_stage != to_stage:
                        count = np.sum((seq[:-1] == from_stage) & (seq[1:] == to_stage))
                        features.append(count / (len(seq) - 1))  # Normalized transition rate
        else:
            features.extend([0, 0, 0] + [0] * 30)  # No transitions possible
        
        # Sleep efficiency metrics (assuming stages 1-4 are sleep, 0 is wake)
        wake_count = np.sum(seq == 0)
        sleep_count = len(seq) - wake_count
        features.append(sleep_count / len(seq))  # Sleep efficiency
        features.append(wake_count / len(seq))  # Wake percentage
        
        # REM percentage (stage 4)
        rem_count = np.sum(seq == 4)
        features.append(rem_count / len(seq))
        
        # Deep sleep percentage (stage 3)
        deep_count = np.sum(seq == 3)
        features.append(deep_count / len(seq))
        
        # Light sleep percentage (stages 1-2)
        light_count = np.sum((seq == 1) | (seq == 2))
        features.append(light_count / len(seq))
        
        # Longest sleep bout (consecutive non-wake)
        if len(seq) > 0:
            sleep_bouts = []
            current_bout = 0
            for s in seq:
                if s != 0:
                    current_bout += 1
                else:
                    if current_bout > 0:
                        sleep_bouts.append(current_bout)
                    current_bout = 0
            if current_bout > 0:
                sleep_bouts.append(current_bout)
            features.append(max(sleep_bouts) if sleep_bouts else 0)
            features.append(np.mean(sleep_bouts) if sleep_bouts else 0)
        else:
            features.extend([0, 0])
        
        features_list.append(np.array(features))
    
    return np.array(features_list)


def prepare_ml_features(csv_path: str, feature_cols: List[str] = None) -> tuple:
    """
    Prepare features for ML classifiers.
    
    Args:
        csv_path: Path to CSV file
        feature_cols: List of PSG feature column names
    
    Returns:
        X: (n_samples, n_features) - combined features
        y: (n_samples, n_classes) - binary targets
        feature_names: List of feature names
    """
    df = pd.read_csv(csv_path)
    
    if feature_cols is None:
        feature_cols = [
            'ahi_a0h3', 'ai_all5', 'odi35', 'timest1p5', 'timest2p5',
            'times34p5', 'timeremp5', 'slp_eff5', 'waso5', 'plmaslp5',
            'slpprdp5', 'remlaiip5'
        ]
    
    target_cols = ['insomnia', 'restless leg', 'apnea']
    
    # Parse sequences
    sequences = []
    for stages_str in df['sleep_stages'].values:
        if pd.isna(stages_str) or stages_str == '':
            sequences.append(np.array([], dtype=np.int64))
        else:
            seq = np.array([int(c) for c in str(stages_str) if c.isdigit()], dtype=np.int64)
            sequences.append(seq)
    
    # Extract sequence features
    print("Extracting sequence features...")
    seq_features = extract_sequence_features(sequences)
    
    # Get PSG features
    psg_features = df[feature_cols].values.astype(np.float32)
    
    # Handle NaN values
    psg_features = np.nan_to_num(psg_features, nan=0.0)
    
    # Combine features
    X = np.hstack([seq_features, psg_features])
    
    # Get targets
    y = df[target_cols].values.astype(np.float32)
    
    # Feature names
    seq_feature_names = [f'seq_feat_{i}' for i in range(seq_features.shape[1])]
    feature_names = seq_feature_names + feature_cols
    
    print(f"Feature shape: {X.shape}")
    print(f"Sequence features: {seq_features.shape[1]}")
    print(f"PSG features: {psg_features.shape[1]}")
    
    return X, y, feature_names

