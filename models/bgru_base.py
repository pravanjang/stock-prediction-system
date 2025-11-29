#!/usr/bin/env python3
"""
Bidirectional GRU Model for BankNifty Directional Prediction.

This module implements a BGRU-based neural network for predicting
the directional movement of BankNifty index using OHLCV sequences
and additional technical, temporal, and price action features.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Constants
MIN_STD = 1e-8  # Minimum standard deviation to avoid division by zero

# Default feature groups for flexible model configuration
OHLCV_FEATURES = ['open', 'high', 'low', 'close', 'volume']

TECHNICAL_FEATURES = [
    'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_hist',
    'stoch_k', 'stoch_d', 'adx_14', 'momentum_10',
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_20', 'sma_50',
    'supertrend_10_3', 'supertrend_7_2',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'psar', 'volume_sma_20', 'volume_roc_10', 'obv', 'vwap',
    'mfi_14', 'atr_14', 'hist_vol_20', 'bb_width_pct'
]

TEMPORAL_FEATURES = [
    'day_sin', 'day_cos', 'week_of_month', 'month_sin', 'month_cos',
    'days_to_monthly_expiry', 'is_monthly_expiry', 'has_weekly_expiry',
    'days_to_weekly_expiry', 'is_weekly_expiry', 'is_expiry_day',
    'days_to_expiry', 'is_expiry_week'
]

PRICE_ACTION_FEATURES = [
    'is_bullish_engulf', 'is_bearish_engulf', 'is_doji', 'is_hammer',
    'is_shooting_star', 'is_inside_bar',
    'return_1', 'return_5', 'return_15', 'return_30', 'return_60',
    'range_pct', 'close_position', 'gap_at_open', 'body_to_range',
    'volume_price_trend', 'price_volume_correlation',
    'volume_imbalance', 'price_impact', 'volatility_regime',
    'volume_trend', 'tick_direction'
]

# Features that should use min-max normalization (already bounded 0-1 or 0-100)
BOUNDED_FEATURES = [
    'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'mfi_14', 'adx_14',
    'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'close_position', 'body_to_range', 'volatility_regime'
]

# Features that should NOT be normalized (binary or already in proper scale)
NO_NORMALIZE_FEATURES = [
    'is_monthly_expiry', 'has_weekly_expiry', 'is_weekly_expiry',
    'is_expiry_day', 'is_expiry_week', 'week_of_month',
    'is_bullish_engulf', 'is_bearish_engulf', 'is_doji', 'is_hammer',
    'is_shooting_star', 'is_inside_bar', 'tick_direction'
]


def get_all_features() -> List[str]:
    """Return all available feature names."""
    return OHLCV_FEATURES + TECHNICAL_FEATURES + TEMPORAL_FEATURES + PRICE_ACTION_FEATURES


def get_feature_groups() -> Dict[str, List[str]]:
    """Return feature groups dictionary."""
    return {
        'ohlcv': OHLCV_FEATURES,
        'technical': TECHNICAL_FEATURES,
        'temporal': TEMPORAL_FEATURES,
        'price_action': PRICE_ACTION_FEATURES
    }


def print_feature_summary() -> None:
    """Print a summary of available features."""
    groups = get_feature_groups()
    print("\n" + "=" * 60)
    print("AVAILABLE FEATURE GROUPS")
    print("=" * 60)
    
    for group_name, features in groups.items():
        print(f"\n{group_name.upper()} ({len(features)} features):")
        print("-" * 40)
        for i, feat in enumerate(features):
            print(f"  {i+1:2d}. {feat}")
    
    total = sum(len(f) for f in groups.values())
    print(f"\nTotal features available: {total}")
    print("=" * 60 + "\n")


class BGRUModel(nn.Module):
    """
    Bidirectional GRU neural network model.
    
    Architecture:
        Input Layer: [batch, sequence_length, 5] (OHLCV features)
        BGRU Layer 1: 128 units bidirectional (256 total)
        Dropout: 0.3
        BGRU Layer 2: 64 units bidirectional (128 total)
        Dropout: 0.3
        Dense Layer: 32 units, ReLU
        Dropout: 0.2
        Output Layer: 1 unit, Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize the BGRU model.
        
        Args:
            input_dim: Number of input features (5 for OHLCV)
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super(BGRUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # For daily data, can use deeper network with self attention
        # for intraday, simpler model without attention
        self.use_attention = use_attention
        
        # First BGRU layer (no internal dropout - applied via separate Dropout layer)
        self.gru1 = nn.GRU(
            input_dim, hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second BGRU layer (no internal dropout - applied via separate Dropout layer)
        self.gru2 = nn.GRU(
            hidden_dim * 2,  # *2 because bidirectional
            hidden_dim // 2, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Attention mechanism (optional)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,  # hidden_dim from gru2 (*2 for bidirectional = hidden_dim)
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, sequence_length, input_dim]
        
        Returns:
            Output tensor of shape [batch, 1] with sigmoid activation
        """
        # First BGRU layer
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        
        # Second BGRU layer
        out, _ = self.gru2(out)  # [batch, seq_len, hidden_dim]
        out = self.dropout2(out)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over sequence
            attn_out, _ = self.attention(out, out, out)
            # Take last time step after attention
            out = attn_out[:, -1, :]  # [batch, hidden_dim]
        else:
            # Without attention, just take last time step
            out = out[:, -1, :]
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout4(out)
        
        # Output layer
        out = self.fc3(out)
        out = self.sigmoid(out)
        
        return out


class BGRUPredictor:
    """
    Bidirectional GRU predictor for BankNifty directional prediction.
    
    This class handles data preparation, model training, prediction,
    and model persistence. Supports configurable feature sets including
    OHLCV, technical indicators, temporal features, and price action features.
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        feature_groups: Optional[List[str]] = None
    ):
        """
        Initialize the BGRU predictor.
        
        Args:
            input_dim: Number of input features. If None, determined from feature_columns.
            hidden_dim: Hidden dimension for BGRU layers (default: 128)
            num_layers: Number of GRU layers (default: 2)
            dropout: Dropout rate (default: 0.3)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            feature_columns: Explicit list of feature column names to use.
                            If None, uses feature_groups or defaults to OHLCV.
            feature_groups: List of feature group names to use: 
                           'ohlcv', 'technical', 'temporal', 'price_action'.
                           Ignored if feature_columns is provided.
        
        Examples:
            # OHLCV only (default behavior)
            predictor = BGRUPredictor()
            
            # OHLCV + Technical indicators
            predictor = BGRUPredictor(feature_groups=['ohlcv', 'technical'])
            
            # All features
            predictor = BGRUPredictor(feature_groups=['ohlcv', 'technical', 'temporal', 'price_action'])
            
            # Custom feature selection
            predictor = BGRUPredictor(feature_columns=['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'macd'])
        """
        # Determine feature columns
        if feature_columns is not None:
            self.feature_columns = feature_columns
        elif feature_groups is not None:
            self.feature_columns = self._get_columns_from_groups(feature_groups)
        else:
            self.feature_columns = OHLCV_FEATURES.copy()
        
        # Set input dimension
        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = len(self.feature_columns)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = None
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Normalization parameters
        self.norm_params: Optional[Dict] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized BGRUPredictor with {len(self.feature_columns)} features")
        self.logger.info(f"Features: {self.feature_columns}")
    
    def _get_columns_from_groups(self, groups: List[str]) -> List[str]:
        """
        Get feature column names from group names.
        
        Args:
            groups: List of group names ('ohlcv', 'technical', 'temporal', 'price_action')
        
        Returns:
            List of feature column names
        """
        all_groups = get_feature_groups()
        columns = []
        for group in groups:
            group_lower = group.lower()
            if group_lower in all_groups:
                columns.extend(all_groups[group_lower])
            else:
                self.logger.warning(f"Unknown feature group: {group}. Available: {list(all_groups.keys())}")
        return columns
    
    def build_model(self) -> BGRUModel:
        """
        Constructs the BGRU model using PyTorch.
        
        Returns:
            BGRUModel instance moved to the appropriate device
        """
        self.model = BGRUModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.logger.info(f"Model built and moved to {self.device}")
        self.logger.info(f"Model architecture:\n{self.model}")
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = 'rolling_zscore',
        window: int = 100,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalizes feature data using appropriate methods for each feature type.
        
        Normalization strategies:
        - OHLCV features: Rolling z-score normalization
        - Bounded features (RSI, etc.): Min-max scaling to [0, 1]
        - Binary features: No normalization (already 0/1)
        - Other features: Rolling z-score normalization
        
        Args:
            df: DataFrame with feature columns
            method: Base normalization method (only 'rolling_zscore' supported)
            window: Rolling window size for z-score calculation
            feature_columns: Columns to normalize. If None, uses self.feature_columns.
        
        Returns:
            Normalized DataFrame
        """
        if method != 'rolling_zscore':
            raise ValueError(f"Unsupported normalization method: {method}")
        
        df = df.copy()
        
        # Use provided columns or instance columns
        cols_to_normalize = feature_columns if feature_columns is not None else self.feature_columns
        
        # Filter to columns that exist in dataframe
        available_cols = [col for col in cols_to_normalize if col in df.columns]
        if not available_cols:
            raise ValueError(f"No feature columns found in DataFrame. Expected: {cols_to_normalize[:5]}...")
        
        missing_cols = set(cols_to_normalize) - set(available_cols)
        if missing_cols:
            self.logger.warning(f"Missing columns in DataFrame: {missing_cols}")
        
        empty_cols = []
        
        for col in available_cols:
            # Skip binary/categorical features that don't need normalization
            if col in NO_NORMALIZE_FEATURES:
                continue
            
            # Use min-max scaling for bounded features
            if col in BOUNDED_FEATURES:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max - col_min > MIN_STD:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.5  # Default to midpoint if no variation
                continue
            
            # Use rolling z-score for unbounded continuous features
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, MIN_STD)
            normalized = (df[col] - rolling_mean) / rolling_std
            normalized = normalized.replace([np.inf, -np.inf], np.nan)
            
            if normalized.isna().all():
                empty_cols.append(col)
                df[col] = 0.0
                continue
            
            # Clip extreme values to prevent outliers from affecting training
            normalized = normalized.clip(-5, 5)
            normalized = normalized.ffill().bfill().fillna(0.0)
            df[col] = normalized
        
        if empty_cols:
            self.logger.warning(
                "Found columns with no valid values after normalization: %s. Filling with zeros.",
                empty_cols
            )
        
        # Handle any remaining NaN/inf values
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        
        return df
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares sequential data for BGRU.
        
        Creates sliding windows of specified length for the configured features.
        
        Args:
            df: DataFrame with feature columns and 'target' column
            sequence_length: Number of time steps in each sequence (default: 60)
            feature_columns: Optional override for feature columns. 
                            If None, uses self.feature_columns.
        
        Returns:
            X: Sequences array of shape [num_samples, sequence_length, num_features]
            y: Target array of shape [num_samples]
        """
        # Use provided columns or instance columns
        cols_to_use = feature_columns if feature_columns is not None else self.feature_columns
        
        # Filter to columns that exist in dataframe
        available_cols = [col for col in cols_to_use if col in df.columns]
        if not available_cols:
            raise ValueError(f"No feature columns found in DataFrame. Expected: {cols_to_use[:5]}...")
        
        if len(available_cols) < len(cols_to_use):
            missing = set(cols_to_use) - set(available_cols)
            raise ValueError(
                f"Missing {len(missing)} required feature columns in DataFrame: {list(missing)[:10]}...\n"
                f"Expected {len(cols_to_use)} columns but found only {len(available_cols)}.\n"
                f"Make sure all data files (train.csv, val.csv, test.csv) have the required features.\n"
                f"You may need to regenerate the data with all features using the feature engineering pipeline."
            )
        
        # Normalize data
        df_norm = self.normalize_data(df, method='rolling_zscore', window=100, feature_columns=available_cols)
        
        # Extract features and ensure they are finite
        features = df_norm[available_cols].values.astype(np.float32, copy=False)
        invalid_features = ~np.isfinite(features)
        if invalid_features.any():
            invalid_count = int(invalid_features.sum())
            self.logger.warning(
                "Detected %d non-finite feature values. Replacing them with zeros before training.",
                invalid_count
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract targets (if available)
        if 'target' in df.columns:
            targets = df['target'].values.astype(np.float32, copy=False)
            if np.isnan(targets).any():
                raise ValueError("Target column contains NaN values. Please regenerate the dataset.")
            if np.logical_or(targets < 0, targets > 1).any():
                unique_vals = np.unique(targets)
                raise ValueError(
                    f"Target column must be binary in [0, 1]. Found values: {unique_vals}"
                )
        else:
            targets = None
        
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            if targets is not None:
                # Target for this sequence is the target at the end of the sequence
                y.append(targets[i + sequence_length - 1])
        
        X = np.array(X, dtype=np.float32)
        
        if targets is not None:
            y = np.array(y, dtype=np.float32)
        else:
            y = np.array([])
        
        self.logger.info(f"Prepared {len(X)} sequences of length {sequence_length} with {len(available_cols)} features")
        
        return X, y
        
        return X, y
    
    def _log_class_distribution(self, targets: np.ndarray, split: str = 'Train') -> Dict[int, int]:
        """Log and return class distribution for diagnostics."""
        if targets.size == 0:
            self.logger.warning(f"No targets available to log distribution for {split} split.")
            return {}
        labels = targets.astype(int)
        unique_classes, counts = np.unique(labels, return_counts=True)
        distribution = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)}
        self.logger.info(f"{split} class distribution: {distribution}")
        return distribution
    
    def _compute_class_weights(self, targets: np.ndarray) -> torch.Tensor:
        """Compute balanced class weights given binary targets."""
        if targets.size == 0:
            raise ValueError("Cannot compute class weights with empty target array.")
        labels = targets.astype(int)
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            self.logger.warning(
                "Only one class present in training targets. Using equal class weights."
            )
            return torch.ones(2, dtype=torch.float32, device=self.device)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=labels
        )
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.logger.info(
            "Computed balanced class weights (neg,pos): %s",
            weights.tolist()
        )
        return weights_tensor
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        sequence_length: int = 60,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = 'models/checkpoints/',
        log_dir: str = 'models/logs/'
    ) -> Dict[str, List[float]]:
        """
        Train the BGRU model.
        
        Training configuration:
        - Optimizer: Adam with lr=0.001, weight_decay=1e-5
        - Loss: Binary Cross-Entropy with optional class weights
        - Batch size: 64
        - Epochs: 50 with early stopping (patience=10)
        - Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
        - Gradient clipping: max_norm=1.0
        
        Args:
            train_df: Training DataFrame with OHLCV and target columns
            val_df: Validation DataFrame with OHLCV and target columns
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 64)
            lr: Learning rate (default: 0.001)
            sequence_length: Sequence length for BGRU input (default: 60)
            class_weights: Optional class weights for imbalanced data (neg,pos). If None,
                balanced weights are computed from the training targets.
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        
        Returns:
            Dictionary containing training history
        """
        # Create directories
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(log_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        self.logger.info("=" * 60)
        self.logger.info("Starting BGRU Training")
        self.logger.info("=" * 60)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        if self.model is None:
            raise RuntimeError("Model initialization failed during training.")
        model = self.model
        
        # Prepare data
        self.logger.info("Preparing training sequences...")
        X_train, y_train = self.prepare_sequences(train_df, sequence_length)
        self.logger.info("Preparing validation sequences...")
        X_val, y_val = self.prepare_sequences(val_df, sequence_length)
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient data for training. Need more samples than sequence_length.")
        
        # Log class distribution and determine weights
        self._log_class_distribution(y_train, split='Train')
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                class_weights_tensor = class_weights.to(self.device, dtype=torch.float32)
            else:
                class_weights_tensor = torch.tensor(
                    class_weights,
                    dtype=torch.float32,
                    device=self.device
                )
            self.logger.info(
                "Using provided class weights (neg,pos): %s",
                class_weights_tensor.tolist()
            )
        else:
            class_weights_tensor = self._compute_class_weights(y_train)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        # Note: pin_memory is disabled since tensors are already on the target device
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Loss function - using BCELoss since model has sigmoid output
        # For class weights, we use pos_weight with BCELoss using reduction='none' and manual weighting
        if class_weights_tensor is not None:
            neg_weight = torch.clamp(class_weights_tensor[0], min=MIN_STD)
            pos_weight = class_weights_tensor[1] / neg_weight
            
            def weighted_bce_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                """Weighted BCE loss for handling class imbalance."""
                # Apply weights: higher weight for positive class
                weights = torch.ones_like(targets)
                weights[targets == 1] = pos_weight
                bce = nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
                return (bce * weights).mean()
            
            criterion = weighted_bce_loss
        else:
            criterion = nn.BCELoss()
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 20
        patience_counter = 0
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        self.logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        self.logger.info("-" * 60)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
            
            for batch_X, batch_y in train_pbar:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                predictions = (outputs >= 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct / train_total:.4f}'
                })
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
                
                for batch_X, batch_y in val_pbar:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{val_correct / val_total:.4f}'
                    })
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Log epoch results
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Early stopping and checkpointing based on validation accuracy (ties broken by loss)
            improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(checkpoint_dir, 'bgru_baseline.pt')
                self.save_model(checkpoint_path)
                self.logger.info(f"Saved best model with Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")
        
        # Plot training curves
        self._plot_training_curves(checkpoint_dir)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
        self.logger.info("=" * 60)
        
        # Remove and close file handler
        self.logger.removeHandler(file_handler)
        file_handler.close()
        
        return self.training_history
    
    def _plot_training_curves(self, output_dir: str) -> None:
        """
        Plot and save training/validation curves.
        
        Args:
            output_dir: Directory to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, self.training_history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.training_history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Training curves saved to {plot_path}")
    
    def predict(
        self,
        test_df: pd.DataFrame,
        sequence_length: int = 60,
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data.
        
        Args:
            test_df: Test DataFrame with OHLCV columns
            sequence_length: Sequence length for BGRU input (default: 60)
            batch_size: Batch size for inference (default: 64)
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probability scores
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
        
        # Prepare sequences
        X_test, y_test = self.prepare_sequences(test_df, sequence_length)
        
        if len(X_test) == 0:
            self.logger.warning("No sequences generated from test data")
            return np.array([]), np.array([])
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Create data loader
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Inference
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for (batch_X,) in tqdm(test_loader, desc="Predicting"):
                outputs = self.model(batch_X)
                all_probs.extend(outputs.cpu().numpy())
        
        probabilities = np.array(all_probs).flatten()
        predictions = (probabilities >= 0.5).astype(int)
        
        self.logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions, probabilities
    
    def save_model(self, path: str = 'models/checkpoints/bgru_baseline.pt') -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save. Call build_model() first.")
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'feature_columns': self.feature_columns,  # Save feature configuration
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
        self.logger.info(f"Saved with {len(self.feature_columns)} features: {self.feature_columns[:5]}...")
    
    def load_model(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the model checkpoint
        
        Note:
            Uses weights_only=False to load configuration and training history.
            Only load checkpoints from trusted sources.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint - weights_only=False needed for config/history dicts
        # Only load checkpoints from trusted sources
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'model_state_dict'")
        
        # Update model configuration with validation
        self.input_dim = int(checkpoint.get('input_dim', self.input_dim))
        self.hidden_dim = int(checkpoint.get('hidden_dim', self.hidden_dim))
        self.num_layers = int(checkpoint.get('num_layers', self.num_layers))
        self.dropout = float(checkpoint.get('dropout', self.dropout))
        
        # Load feature columns if available
        if 'feature_columns' in checkpoint:
            self.feature_columns = checkpoint['feature_columns']
            self.logger.info(f"Loaded feature configuration with {len(self.feature_columns)} features")
        
        # Build model with loaded configuration
        self.build_model()
        if self.model is None:
            raise RuntimeError("Model initialization failed while loading checkpoint.")
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Model loaded from {path}")
        self.logger.info(f"Model uses {len(self.feature_columns)} features: {self.feature_columns[:5]}...")


def setup_logging(log_dir: str = 'models/logs/') -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def check_class_distribution(df: pd.DataFrame) -> np.ndarray:
    """Check target distribution and return balanced class weights."""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Check target distribution
    target_counts = df['target'].value_counts()
    print(f"\nTarget Distribution:")
    print(f"  Class 0 (DOWN): {target_counts[0]} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"  Class 1 (UP):   {target_counts[1]} ({target_counts[1]/len(df)*100:.2f}%)")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(df['target']), 
                                         y=df['target'])
    print(f"\nRecommended Class Weights:")
    print(f"  Class 0: {class_weights[0]:.4f}")
    print(f"  Class 1: {class_weights[1]:.4f}")
    print("="*60 + "\n")
    
    return class_weights


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Train or evaluate BGRU model for BankNifty prediction'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run predictions on test data'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/',
        help='Directory containing train.csv, val.csv, test.csv'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Sequence length for BGRU input (default: 60)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='models/checkpoints/',
        help='Directory for model checkpoints'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='models/logs/',
        help='Directory for training logs'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model checkpoint for loading'
    )
    parser.add_argument(
        '--feature_groups',
        type=str,
        nargs='+',
        default=['ohlcv'],
        choices=['ohlcv', 'technical', 'temporal', 'price_action'],
        help='Feature groups to use (default: ohlcv). Options: ohlcv, technical, temporal, price_action'
    )
    parser.add_argument(
        '--all_features',
        action='store_true',
        help='Use all available features (ohlcv + technical + temporal + price_action)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    # Determine feature groups
    if args.all_features:
        feature_groups = ['ohlcv', 'technical', 'temporal', 'price_action']
    else:
        feature_groups = args.feature_groups
    
    logger.info(f"Using feature groups: {feature_groups}")
    
    # Initialize predictor with feature groups
    predictor = BGRUPredictor(feature_groups=feature_groups)
    
    if args.train:
        # Load training and validation data
        train_path = os.path.join(args.data_dir, 'train.csv')
        val_path = os.path.join(args.data_dir, 'val.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            logger.error(f"Training data not found in {args.data_dir}")
            logger.error("Please run data/data_loader.py first to prepare the data.")
            return 1
        
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)

        class_weights = check_class_distribution(train_df)
        logger.info(f"Using class weights: {class_weights}")
        
        logger.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        logger.info(f"Model input dimension: {predictor.input_dim} features")
        
        # Train model
        predictor.train(
            train_df=train_df,
            val_df=val_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sequence_length=args.sequence_length,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        
        logger.info("Training complete!")
    
    if args.predict:
        # Load model
        if args.model_path is None:
            model_path = os.path.join(args.checkpoint_dir, 'bgru_baseline.pt')
        else:
            model_path = args.model_path
        
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found at {model_path}")
            return 1
        
        predictor.load_model(model_path)
        
        # Load test data
        test_path = os.path.join(args.data_dir, 'test.csv')
        
        if not os.path.exists(test_path):
            logger.error(f"Test data not found at {test_path}")
            return 1
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Generate predictions
        predictions, probabilities = predictor.predict(
            test_df=test_df,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size
        )
        
        # Save predictions
        output_path = os.path.join(args.checkpoint_dir, 'predictions.csv')
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Calculate accuracy if targets are available
        if 'target' in test_df.columns:
            _, y_test = predictor.prepare_sequences(test_df, args.sequence_length)
            if len(y_test) == len(predictions):
                accuracy = (predictions == y_test).mean()
                logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    if not args.train and not args.predict:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
