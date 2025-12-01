#!/usr/bin/env python3
"""
Hybrid BGRU Model for BankNifty Price Prediction.

This module implements a hybrid architecture combining:
- Path 1: BGRU processing sequential OHLCV data
- Path 2: Dense layers processing static features (technical indicators, temporal)
- Fusion: Concatenate BGRU output + static features
- Output: Dense layers for regression (next day close price prediction)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Constants
MIN_STD = 1e-8  # Minimum standard deviation to avoid division by zero

# Default OHLCV features for sequential path
OHLCV_FEATURES = ['open', 'high', 'low', 'close', 'volume']

# Technical indicator features for static path
TECHNICAL_FEATURES = [
    'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_hist',
    'stoch_k', 'stoch_d', 'adx_14', 'momentum_10',
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_20', 'sma_50',
    'supertrend_10_3', 'supertrend_7_2',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'psar', 'volume_sma_20', 'volume_roc_10', 'obv', 'vwap',
    'mfi_14', 'atr_14', 'hist_vol_20', 'bb_width_pct',
    # Momentum signal features for improved recall
    'rsi_bullish_divergence', 'rsi_bearish_divergence',
    'macd_bullish_cross', 'macd_bearish_cross',
    'volume_spike', 'high_volume_up', 'momentum_score',
    'ema_bullish_alignment', 'ema_bearish_alignment',
    'breakout_up', 'breakout_down',
    'rsi_oversold', 'rsi_overbought', 'rsi_momentum_entry',
    'bullish_signal_count'
]

# Temporal features for static path
TEMPORAL_FEATURES = [
    'day_sin', 'day_cos', 'week_of_month', 'month_sin', 'month_cos',
    'days_to_monthly_expiry', 'is_monthly_expiry', 'has_weekly_expiry',
    'days_to_weekly_expiry', 'is_weekly_expiry', 'is_expiry_day',
    'days_to_expiry', 'is_expiry_week'
]

# Price action features for static path
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


def get_static_features() -> List[str]:
    """Return all static feature names (technical + temporal + price action)."""
    return TECHNICAL_FEATURES + TEMPORAL_FEATURES + PRICE_ACTION_FEATURES


class HybridBGRUNetwork(nn.Module):
    """
    Hybrid BGRU neural network with dual-path architecture.
    
    Architecture:
        [Sequential Path - OHLCV sequences]
        Input: [batch, sequence_length, n_price_features]
            ↓
        BGRU Layer 1: hidden_dim units, bidirectional (output: hidden_dim*2)
            ↓
        Dropout: dropout
            ↓
        BGRU Layer 2: hidden_dim//2 units, bidirectional (output: hidden_dim)
            ↓
        Dropout: dropout
            ↓
        (Optional additional layers based on num_layers)
            ↓
        Output: [batch, hidden_dim]
        
        [Static Path - Technical/Temporal/Price Action features]
        Input: [batch, n_static_features]
            ↓
        Dense: 64 units, ReLU
            ↓
        Dropout: dropout
            ↓
        Dense: 32 units, ReLU
            ↓
        Output: [batch, 32]
        
        [Fusion Layer]
        Concat: [batch, hidden_dim + 32]
            ↓
        Dense: 64 units, ReLU
            ↓
        Dropout: 0.3
            ↓
        Dense: 32 units, ReLU
            ↓
        Dropout: 0.2
            ↓
        Output: 1 unit, Sigmoid
    """
    
    def __init__(
        self,
        n_price_features: int = 5,
        n_static_features: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the Hybrid BGRU network.
        
        Args:
            n_price_features: Number of OHLCV features (default: 5)
            n_static_features: Number of static features (default: 50)
            hidden_dim: Hidden dimension for BGRU layers (default: 128)
            num_layers: Number of BGRU layers (default: 2)
            dropout: Dropout probability (default: 0.3)
        """
        super(HybridBGRUNetwork, self).__init__()
        
        self.n_price_features = n_price_features
        self.n_static_features = n_static_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # =========== Sequential Path (BGRU) ===========
        # Build BGRU layers dynamically based on num_layers
        self.gru_layers = nn.ModuleList()
        self.gru_dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> hidden_dim
                input_size = n_price_features
                output_size = hidden_dim
            else:
                # Subsequent layers: hidden_dim*2 -> hidden_dim // (2^i)
                input_size = hidden_dim * 2 if i == 1 else self.gru_layers[i-1].hidden_size * 2
                output_size = max(hidden_dim // (2 ** i), 32)  # Minimum 32 units
            
            gru = nn.GRU(
                input_size,
                output_size,
                batch_first=True,
                bidirectional=True
            )
            self.gru_layers.append(gru)
            self.gru_dropouts.append(nn.Dropout(dropout))
        
        # Calculate final GRU output dimension
        self.gru_output_dim = self.gru_layers[-1].hidden_size * 2  # *2 for bidirectional
        
        # =========== Static Path (Dense) ===========
        # Dense 64 -> Dense 32
        self.static_fc1 = nn.Linear(n_static_features, 64)
        self.static_relu1 = nn.ReLU()
        self.static_dropout1 = nn.Dropout(dropout)
        
        self.static_fc2 = nn.Linear(64, 32)
        self.static_relu2 = nn.ReLU()
        
        # =========== Fusion Layer ===========
        # BGRU output (gru_output_dim) + Static output (32)
        fusion_input_dim = self.gru_output_dim + 32
        
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 64)
        self.fusion_relu1 = nn.ReLU()
        self.fusion_dropout1 = nn.Dropout(dropout)
        
        self.fusion_fc2 = nn.Linear(64, 32)
        self.fusion_relu2 = nn.ReLU()
        self.fusion_dropout2 = nn.Dropout(0.2)
        
        # Output layer
        self.output_fc = nn.Linear(32, 1)
        # Removed sigmoid - no activation for regression
    
    def forward(
        self,
        seq_input: torch.Tensor,
        static_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        
        Args:
            seq_input: Sequential OHLCV data [batch, sequence_length, n_price_features]
            static_input: Static features [batch, n_static_features]
        
        Returns:
            Output tensor of shape [batch, 1] for regression
        """
        # Sequential Path - pass through all GRU layers
        out_seq = seq_input
        for gru, dropout in zip(self.gru_layers, self.gru_dropouts):
            out_seq, _ = gru(out_seq)
            out_seq = dropout(out_seq)
        
        # Take last time step output [batch, gru_output_dim]
        out_seq = out_seq[:, -1, :]
        
        # Static Path
        out_static = self.static_fc1(static_input)
        out_static = self.static_relu1(out_static)
        out_static = self.static_dropout1(out_static)
        
        out_static = self.static_fc2(out_static)
        out_static = self.static_relu2(out_static)
        
        # Fusion
        fusion = torch.cat([out_seq, out_static], dim=1)
        
        out = self.fusion_fc1(fusion)
        out = self.fusion_relu1(out)
        out = self.fusion_dropout1(out)
        
        out = self.fusion_fc2(out)
        out = self.fusion_relu2(out)
        out = self.fusion_dropout2(out)
        
        # Output
        out = self.output_fc(out)
        return out  # No sigmoid activation for regression


class HybridBGRUModel:
    """
    Hybrid BGRU Model for BankNifty directional prediction.
    
    Combines sequential OHLCV data with static technical, temporal,
    and price action features through a dual-path architecture.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_price_features: int = 5,
        n_static_features: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the Hybrid BGRU Model.
        
        Args:
            sequence_length: Number of time steps for sequential input (default: 60)
            n_price_features: Number of OHLCV features (default: 5)
            n_static_features: Number of static features (default: 50)
            hidden_dim: Hidden dimension for BGRU layers (default: 128)
            num_layers: Number of BGRU layers (default: 2)
            dropout: Dropout probability (default: 0.3)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.sequence_length = sequence_length
        self.n_price_features = n_price_features
        self.n_static_features = n_static_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Feature columns
        self.ohlcv_columns = OHLCV_FEATURES.copy()
        self.static_columns = get_static_features()
        
        # Initialize model
        self.model: Optional[HybridBGRUNetwork] = None
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized HybridBGRUModel with sequence_length={sequence_length}, "
            f"n_price_features={n_price_features}, n_static_features={n_static_features}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}"
        )
    
    def build_model(self) -> HybridBGRUNetwork:
        """
        Constructs the Hybrid BGRU model.
        
        Returns:
            HybridBGRUNetwork instance moved to the appropriate device
        """
        self.model = HybridBGRUNetwork(
            n_price_features=self.n_price_features,
            n_static_features=self.n_static_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.logger.info(
            f"Model built with hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, dropout={self.dropout}, device={self.device}"
        )
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        window: int = 100
    ) -> pd.DataFrame:
        """
        Normalize feature data using appropriate methods for each feature type.
        
        Args:
            df: DataFrame with feature columns
            columns: Columns to normalize
            window: Rolling window size for z-score calculation
        
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Skip binary/categorical features
            if col in NO_NORMALIZE_FEATURES:
                continue
            
            # Use min-max scaling for bounded features
            if col in BOUNDED_FEATURES:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max - col_min > MIN_STD:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.5
                continue
            
            # Use rolling z-score for unbounded features
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            rolling_std = rolling_std.replace(0, MIN_STD)
            normalized = (df[col] - rolling_mean) / rolling_std
            normalized = normalized.replace([np.inf, -np.inf], np.nan)
            normalized = normalized.clip(-5, 5)
            normalized = normalized.ffill().bfill().fillna(0.0)
            df[col] = normalized
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares two types of inputs for the hybrid model.
        
        Args:
            df: DataFrame with OHLCV, technical, temporal, price action features
            fit_scaler: Whether to fit target normalization scaler (True for training)
        
        Returns:
            Tuple of (seq_data, static_data, targets)
            - seq_data: shape [batch, sequence_length, n_price_features]
            - static_data: shape [batch, n_static_features]
            - targets: shape [batch]
        """
        # Get available columns
        available_ohlcv = [c for c in self.ohlcv_columns if c in df.columns]
        available_static = [c for c in self.static_columns if c in df.columns]
        
        if len(available_ohlcv) < len(self.ohlcv_columns):
            missing = set(self.ohlcv_columns) - set(available_ohlcv)
            raise ValueError(f"Missing OHLCV columns: {missing}")
        
        # Update n_static_features based on available columns
        self.n_static_features = len(available_static)
        self.logger.info(f"Using {len(available_static)} static features")
        
        # Normalize data
        df_norm = self.normalize_data(df, available_ohlcv + available_static)
        
        # Extract features
        ohlcv_data = df_norm[available_ohlcv].values.astype(np.float32)
        static_data = df_norm[available_static].values.astype(np.float32)
        
        # Handle NaN/Inf
        ohlcv_data = np.nan_to_num(ohlcv_data, nan=0.0, posinf=0.0, neginf=0.0)
        static_data = np.nan_to_num(static_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract targets
        if 'target' in df.columns:
            targets = df['target'].values.astype(np.float32)
            
            # Normalize targets for better training stability
            if fit_scaler:
                self.target_mean = float(targets.mean())
                self.target_std = float(targets.std())
                self.logger.info(f"Target normalization - Mean: {self.target_mean:.2f}, Std: {self.target_std:.2f}")
            
            # Apply normalization if scaler exists
            if hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
                targets = (targets - self.target_mean) / (self.target_std + MIN_STD)
        else:
            targets = None
        
        # Create sequences
        X_seq, X_static, y = [], [], []
        
        for i in range(len(df) - self.sequence_length):
            X_seq.append(ohlcv_data[i:i + self.sequence_length])
            X_static.append(static_data[i + self.sequence_length - 1])
            if targets is not None:
                y.append(targets[i + self.sequence_length - 1])
        
        X_seq = np.array(X_seq, dtype=np.float32)
        X_static = np.array(X_static, dtype=np.float32)
        
        if targets is not None:
            y = np.array(y, dtype=np.float32)
        else:
            y = np.array([])
        
        self.logger.info(
            f"Prepared {len(X_seq)} samples: "
            f"seq_shape={X_seq.shape}, static_shape={X_static.shape}"
        )
        
        return X_seq, X_static, y
    
    def _log_class_distribution(
        self,
        targets: np.ndarray,
        split: str = 'Train'
    ) -> Dict[int, int]:
        """Log and return class distribution for diagnostics."""
        if targets.size == 0:
            return {}
        labels = targets.astype(int)
        unique_classes, counts = np.unique(labels, return_counts=True)
        distribution = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)}
        self.logger.info(f"{split} class distribution: {distribution}")
        return distribution
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 64,
        checkpoint_dir: str = 'models/checkpoints/',
        log_dir: str = 'models/logs/'
    ) -> Dict[str, List[float]]:
        """
        Train the Hybrid BGRU model with two-phase approach for regression.
        
        Phase 1 (20 epochs): Train entire model with lr=0.001
        Phase 2 (30 epochs): Fine-tune with lr=0.0001
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            epochs: Total number of training epochs (default: 50)
            batch_size: Batch size (default: 64)
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        
        Returns:
            Dictionary containing training history
        """
        # Create directories
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(log_dir, 'hybrid_training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Hybrid BGRU Training")
        self.logger.info("=" * 60)
        
        # Prepare data
        self.logger.info("Preparing training data...")
        X_seq_train, X_static_train, y_train = self.prepare_data(train_df, fit_scaler=True)
        self.logger.info("Preparing validation data...")
        X_seq_val, X_static_val, y_val = self.prepare_data(val_df, fit_scaler=False)
        
        if len(X_seq_train) == 0 or len(X_seq_val) == 0:
            raise ValueError("Insufficient data for training.")
        
        # Validate and update static features dimension before building model
        actual_static_dim = X_static_train.shape[1]
        if self.n_static_features != actual_static_dim:
            self.logger.info(
                f"Updating n_static_features from {self.n_static_features} "
                f"to {actual_static_dim} based on data"
            )
            self.n_static_features = actual_static_dim
        
        # Build model with correct dimensions
        if self.model is None:
            self.build_model()
        if self.model is None:
            raise RuntimeError("Model initialization failed.")
        
        # Convert to tensors
        X_seq_train_t = torch.FloatTensor(X_seq_train).to(self.device)
        X_static_train_t = torch.FloatTensor(X_static_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        X_seq_val_t = torch.FloatTensor(X_seq_val).to(self.device)
        X_static_val_t = torch.FloatTensor(X_static_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_seq_train_t, X_static_train_t, y_train_t)
        val_dataset = TensorDataset(X_seq_val_t, X_static_val_t, y_val_t)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Use Mean Squared Error Loss for regression
        self.logger.info("Using MSE Loss for regression")
        criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # Two-phase training
        phase1_epochs = min(20, epochs)
        phase2_epochs = epochs - phase1_epochs
        
        self.logger.info(f"Training samples: {len(X_seq_train)}")
        self.logger.info(f"Validation samples: {len(X_seq_val)}")
        self.logger.info(f"Phase 1: {phase1_epochs} epochs with lr=0.001")
        self.logger.info(f"Phase 2: {phase2_epochs} epochs with lr=0.0001")
        self.logger.info("-" * 60)
        
        # Phase 1: lr = 0.001
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        for epoch in range(epochs):
            # Switch to Phase 2 optimizer
            if epoch == phase1_epochs:
                self.logger.info("=" * 60)
                self.logger.info("Starting Phase 2: Fine-tuning with lr=0.0001")
                self.logger.info("=" * 60)
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=0.0001,
                    weight_decay=1e-5
                )
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            train_total = 0
            
            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs} [Train]"
            )
            
            for batch_seq, batch_static, batch_y in train_pbar:
                optimizer.zero_grad()
                
                outputs = self.model(batch_seq, batch_static)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_seq.size(0)
                mae = torch.abs(outputs - batch_y).mean()
                train_mae += mae.item() * batch_seq.size(0)
                train_total += batch_y.size(0)
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{mae.item():.4f}'
                })
            
            train_loss /= train_total
            train_mae /= train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Val]"
                )
                
                for batch_seq, batch_static, batch_y in val_pbar:
                    outputs = self.model(batch_seq, batch_static)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_seq.size(0)
                    mae = torch.abs(outputs - batch_y).mean()
                    val_mae += mae.item() * batch_seq.size(0)
                    val_total += batch_y.size(0)
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{mae.item():.4f}'
                    })
            
            val_loss /= val_total
            val_mae /= val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_mae'].append(val_mae)
            
            # Log epoch results
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Early stopping
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = os.path.join(checkpoint_dir, 'bgru_hybrid.pt')
                self.save_model(checkpoint_path)
                self.logger.info(
                    f"Saved best model with Val Loss: {val_loss:.4f}, "
                    f"Val MAE: {val_mae:.4f}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, 'hybrid_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")
        
        # Plot training curves
        self._plot_training_curves(checkpoint_dir)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
        self.logger.info("=" * 60)
        
        # Cleanup
        self.logger.removeHandler(file_handler)
        file_handler.close()
        
        return self.training_history
    
    def _plot_training_curves(self, output_dir: str) -> None:
        """Plot and save training/validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(
            epochs_range, self.training_history['train_loss'],
            'b-', label='Train Loss'
        )
        axes[0].plot(
            epochs_range, self.training_history['val_loss'],
            'r-', label='Val Loss'
        )
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(
            epochs_range, self.training_history['train_mae'],
            'b-', label='Train MAE'
        )
        axes[1].plot(
            epochs_range, self.training_history['val_mae'],
            'r-', label='Val MAE'
        )
        axes[1].set_title('Training and Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'hybrid_training_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Training curves saved to {plot_path}")
    
    def predict(
        self,
        test_df: pd.DataFrame,
        batch_size: int = 64,
        denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price predictions on test data.
        
        Args:
            test_df: Test DataFrame
            batch_size: Batch size for inference
            denormalize: Whether to denormalize predictions back to original scale
        
        Returns:
            predictions: Predicted next day close prices
            current_prices: Current day close prices for reference
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Prepare data (don't fit new scaler)
        X_seq, X_static, _ = self.prepare_data(test_df, fit_scaler=False)
        
        if len(X_seq) == 0:
            self.logger.warning("No sequences generated from test data")
            return np.array([]), np.array([])
        
        # Get current prices for reference
        # After sequence creation, each prediction corresponds to the last element of each sequence
        start_idx = self.sequence_length - 1
        end_idx = start_idx + len(X_seq)
        current_prices = test_df['close'].iloc[start_idx:end_idx].values
        
        # Convert to tensors
        X_seq_t = torch.FloatTensor(X_seq).to(self.device)
        X_static_t = torch.FloatTensor(X_static).to(self.device)
        
        # Create data loader
        test_dataset = TensorDataset(X_seq_t, X_static_t)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Inference
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch_seq, batch_static in tqdm(test_loader, desc="Predicting"):
                outputs = self.model(batch_seq, batch_static)
                all_preds.extend(outputs.cpu().numpy())
        
        predictions = np.array(all_preds).flatten()
        
        # Denormalize predictions if requested
        if denormalize and hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
            predictions = predictions * self.target_std + self.target_mean
            self.logger.info("Predictions denormalized to original price scale")
        
        self.logger.info(f"Generated {len(predictions)} predictions")
        self.logger.info(f"Predicted price range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        self.logger.info(f"Current price range: [{current_prices.min():.2f}, {current_prices.max():.2f}]")
        
        return predictions, current_prices
    
    def save_model(self, path: str = 'models/checkpoints/bgru_hybrid.pt') -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'n_price_features': self.n_price_features,
            'n_static_features': self.n_static_features,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'ohlcv_columns': self.ohlcv_columns,
            'static_columns': self.static_columns,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save target normalization parameters for regression
        if hasattr(self, 'target_mean'):
            checkpoint['target_mean'] = self.target_mean
        if hasattr(self, 'target_std'):
            checkpoint['target_std'] = self.target_std
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the model checkpoint
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint - weights_only=False is required to load config dicts
        # Only load checkpoints from trusted sources (locally saved models)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'model_state_dict'")
        
        # Update configuration
        self.sequence_length = int(
            checkpoint.get('sequence_length', self.sequence_length)
        )
        self.n_price_features = int(
            checkpoint.get('n_price_features', self.n_price_features)
        )
        
        # Load hyperparameters from checkpoint (with backward compatibility)
        self.hidden_dim = int(checkpoint.get('hidden_dim', 128))
        self.num_layers = int(checkpoint.get('num_layers', 2))
        self.dropout = float(checkpoint.get('dropout', 0.3))
        
        # Load target normalization parameters for regression
        if 'target_mean' in checkpoint:
            self.target_mean = checkpoint['target_mean']
        if 'target_std' in checkpoint:
            self.target_std = checkpoint['target_std']
        
        # Load column configurations first (needed to determine n_static_features)
        # Load column configurations first
        if 'ohlcv_columns' in checkpoint:
            self.ohlcv_columns = checkpoint['ohlcv_columns']
        if 'static_columns' in checkpoint:
            self.static_columns = checkpoint['static_columns']
        
        # Determine n_static_features from the actual model weights
        # This is the most reliable way to get the correct value
        if 'static_fc1.weight' in checkpoint['model_state_dict']:
            weight_shape = checkpoint['model_state_dict']['static_fc1.weight'].shape
            actual_n_static = weight_shape[1]  # Input dimension is the second dim
            
            if actual_n_static == 0:
                self.logger.warning(
                    "Model was trained with n_static_features=0 (no static features). "
                    "The model will not use static features for predictions."
                )
                self.n_static_features = 0
                self.static_columns = []  # Clear static columns since model doesn't use them
            else:
                self.n_static_features = actual_n_static
                # Verify static_columns matches
                if self.static_columns and len(self.static_columns) != actual_n_static:
                    self.logger.warning(
                        f"static_columns length ({len(self.static_columns)}) doesn't match "
                        f"model weights ({actual_n_static}). Using weight dimension."
                    )
        else:
            # Fallback to checkpoint value or static_columns length
            if self.static_columns:
                self.n_static_features = len(self.static_columns)
            else:
                self.n_static_features = int(
                    checkpoint.get('n_static_features', self.n_static_features)
                )
        
        self.logger.info(
            f"Loading model with n_price_features={self.n_price_features}, "
            f"n_static_features={self.n_static_features}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, dropout={self.dropout}"
        )
        
        # Build and load model
        self.build_model()
        if self.model is None:
            raise RuntimeError("Model initialization failed.")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Model loaded from {path}")


def setup_logging(log_dir: str = 'models/logs/') -> None:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Train or evaluate Hybrid BGRU model for BankNifty prediction'
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
        '--data',
        type=str,
        default='data/processed/train_final.csv',
        help='Path to training data CSV'
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    # Initialize model
    model = HybridBGRUModel(sequence_length=args.sequence_length)
    
    if args.train:
        # Load training and validation data
        train_path = os.path.join(args.data_dir, 'train.csv')
        val_path = os.path.join(args.data_dir, 'val.csv')
        
        # Also try train_final.csv if train.csv doesn't exist
        if not os.path.exists(train_path):
            train_path = os.path.join(args.data_dir, 'train_final.csv')
        if not os.path.exists(val_path):
            val_path = os.path.join(args.data_dir, 'val_final.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            logger.error(f"Training data not found in {args.data_dir}")
            return 1
        
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        
        logger.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Train model
        model.train(
            train_df=train_df,
            val_df=val_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        
        logger.info("Training complete!")
    
    if args.predict:
        # Load model
        if args.model_path is None:
            model_path = os.path.join(args.checkpoint_dir, 'bgru_hybrid.pt')
        else:
            model_path = args.model_path
        
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found at {model_path}")
            return 1
        
        model.load_model(model_path)
        
        # Load test data
        test_path = os.path.join(args.data_dir, 'test.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(args.data_dir, 'test_final.csv')
        
        if not os.path.exists(test_path):
            logger.error("Test data not found")
            return 1
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Generate predictions
        predictions, current_prices = model.predict(
            test_df=test_df,
            batch_size=args.batch_size,
            denormalize=True
        )
        
        # Get actual values if available
        actual_values = None
        if 'target' in test_df.columns:
            actual_values = test_df['target'].iloc[model.sequence_length-1:model.sequence_length-1+len(predictions)].values
        
        # Calculate metrics if we have actual values
        if actual_values is not None and len(actual_values) == len(predictions):
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mae = mean_absolute_error(actual_values, predictions)
            rmse = np.sqrt(mean_squared_error(actual_values, predictions))
            r2 = r2_score(actual_values, predictions)
            # Use epsilon to prevent division by zero in MAPE calculation
            mape = np.mean(np.abs((actual_values - predictions) / (actual_values + MIN_STD))) * 100
            
            # Directional accuracy
            actual_direction = (actual_values > current_prices).astype(int)
            pred_direction = (predictions > current_prices).astype(int)
            directional_acc = (actual_direction == pred_direction).mean()
            
            logger.info(f"\n{'='*50}")
            logger.info("Test Metrics:")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"R²: {r2:.4f}")
            logger.info(f"MAPE: {mape:.2f}%")
            logger.info(f"Directional Accuracy: {directional_acc:.4f}")
            logger.info(f"{'='*50}\n")
        
        # Save predictions with details
        output_path = os.path.join(args.checkpoint_dir, 'hybrid_predictions.csv')
        results_df = pd.DataFrame({
            'current_price': current_prices,
            'predicted_price': predictions
        })
        
        if actual_values is not None:
            results_df['actual_price'] = actual_values
            results_df['prediction_error'] = predictions - actual_values
            results_df['percent_error'] = ((predictions - actual_values) / actual_values) * 100
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    if not args.train and not args.predict:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
