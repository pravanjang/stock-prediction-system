#!/usr/bin/env python3
"""
Hybrid BGRU Model for BankNifty Directional Prediction.

This module implements a hybrid architecture combining:
- Path 1: BGRU processing sequential OHLCV data
- Path 2: Dense layers processing static features (technical indicators, temporal)
- Fusion: Concatenate BGRU output + static features
- Output: Dense layers for classification
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
from sklearn.utils.class_weight import compute_class_weight
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
    'mfi_14', 'atr_14', 'hist_vol_20', 'bb_width_pct'
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
        BGRU Layer 1: 128 units, bidirectional (output: 256)
            ↓
        Dropout: 0.3
            ↓
        BGRU Layer 2: 64 units, bidirectional (output: 128)
            ↓
        Dropout: 0.3
            ↓
        Output: [batch, 128]
        
        [Static Path - Technical/Temporal/Price Action features]
        Input: [batch, n_static_features]
            ↓
        Dense: 64 units, ReLU
            ↓
        Dropout: 0.3
            ↓
        Dense: 32 units, ReLU
            ↓
        Output: [batch, 32]
        
        [Fusion Layer]
        Concat: [batch, 160] (128 from BGRU + 32 from static)
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
        dropout: float = 0.3
    ):
        """
        Initialize the Hybrid BGRU network.
        
        Args:
            n_price_features: Number of OHLCV features (default: 5)
            n_static_features: Number of static features (default: 50)
            hidden_dim: Hidden dimension for BGRU layers (default: 128)
            dropout: Dropout probability (default: 0.3)
        """
        super(HybridBGRUNetwork, self).__init__()
        
        self.n_price_features = n_price_features
        self.n_static_features = n_static_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # =========== Sequential Path (BGRU) ===========
        # BGRU Layer 1: 128 units bidirectional
        self.gru1 = nn.GRU(
            n_price_features,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # BGRU Layer 2: 64 units bidirectional
        self.gru2 = nn.GRU(
            hidden_dim * 2,  # *2 because bidirectional
            hidden_dim // 2,  # 64 units
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # =========== Static Path (Dense) ===========
        # Dense 64 -> Dense 32
        self.static_fc1 = nn.Linear(n_static_features, 64)
        self.static_relu1 = nn.ReLU()
        self.static_dropout1 = nn.Dropout(dropout)
        
        self.static_fc2 = nn.Linear(64, 32)
        self.static_relu2 = nn.ReLU()
        
        # =========== Fusion Layer ===========
        # BGRU output (128) + Static output (32) = 160
        fusion_input_dim = hidden_dim + 32  # 128 + 32 = 160
        
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 64)
        self.fusion_relu1 = nn.ReLU()
        self.fusion_dropout1 = nn.Dropout(dropout)
        
        self.fusion_fc2 = nn.Linear(64, 32)
        self.fusion_relu2 = nn.ReLU()
        self.fusion_dropout2 = nn.Dropout(0.2)
        
        # Output layer
        self.output_fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
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
            Output tensor of shape [batch, 1] with sigmoid activation
        """
        # Sequential Path
        out_seq, _ = self.gru1(seq_input)
        out_seq = self.dropout1(out_seq)
        
        out_seq, _ = self.gru2(out_seq)
        out_seq = self.dropout2(out_seq)
        
        # Take last time step output [batch, hidden_dim]
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
        out = self.sigmoid(out)
        
        return out


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
        device: Optional[str] = None
    ):
        """
        Initialize the Hybrid BGRU Model.
        
        Args:
            sequence_length: Number of time steps for sequential input (default: 60)
            n_price_features: Number of OHLCV features (default: 5)
            n_static_features: Number of static features (default: 50)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.sequence_length = sequence_length
        self.n_price_features = n_price_features
        self.n_static_features = n_static_features
        
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
            f"n_price_features={n_price_features}, n_static_features={n_static_features}"
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
            hidden_dim=128,
            dropout=0.3
        ).to(self.device)
        
        self.logger.info(f"Model built and moved to {self.device}")
        
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
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares two types of inputs for the hybrid model.
        
        Args:
            df: DataFrame with OHLCV, technical, temporal, price action features
        
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
    
    def _compute_class_weights(self, targets: np.ndarray) -> torch.Tensor:
        """Compute balanced class weights given binary targets."""
        if targets.size == 0:
            raise ValueError("Cannot compute class weights with empty target array.")
        labels = targets.astype(int)
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            self.logger.warning(
                "Only one class present. Using equal class weights."
            )
            return torch.ones(2, dtype=torch.float32, device=self.device)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=labels
        )
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
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
        Train the Hybrid BGRU model with two-phase approach.
        
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
        X_seq_train, X_static_train, y_train = self.prepare_data(train_df)
        self.logger.info("Preparing validation data...")
        X_seq_val, X_static_val, y_val = self.prepare_data(val_df)
        
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
        
        # Log class distribution and compute weights
        self._log_class_distribution(y_train, split='Train')
        class_weights = self._compute_class_weights(y_train)
        
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
        
        # Loss function with class weights
        neg_weight = torch.clamp(class_weights[0], min=MIN_STD)
        pos_weight = class_weights[1] / neg_weight
        
        def weighted_bce_loss(
            outputs: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
            weights = torch.ones_like(targets)
            weights[targets == 1] = pos_weight
            bce = nn.functional.binary_cross_entropy(
                outputs, targets, reduction='none'
            )
            return (bce * weights).mean()
        
        criterion = weighted_bce_loss
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 10
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
            train_correct = 0
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
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
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
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{val_correct / val_total:.4f}'
                    })
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Update scheduler
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
            
            # Early stopping
            improved = (
                val_acc > best_val_acc or
                (val_acc == best_val_acc and val_loss < best_val_loss)
            )
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = os.path.join(checkpoint_dir, 'bgru_hybrid.pt')
                self.save_model(checkpoint_path)
                self.logger.info(
                    f"Saved best model with Val Acc: {val_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
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
        self.logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
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
        
        # Accuracy plot
        axes[1].plot(
            epochs_range, self.training_history['train_acc'],
            'b-', label='Train Acc'
        )
        axes[1].plot(
            epochs_range, self.training_history['val_acc'],
            'r-', label='Val Acc'
        )
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
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
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data.
        
        Args:
            test_df: Test DataFrame
            batch_size: Batch size for inference
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probability scores
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Prepare data
        X_seq, X_static, _ = self.prepare_data(test_df)
        
        if len(X_seq) == 0:
            self.logger.warning("No sequences generated from test data")
            return np.array([]), np.array([])
        
        # Convert to tensors
        X_seq_t = torch.FloatTensor(X_seq).to(self.device)
        X_static_t = torch.FloatTensor(X_static).to(self.device)
        
        # Create data loader
        test_dataset = TensorDataset(X_seq_t, X_static_t)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Inference
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch_seq, batch_static in tqdm(test_loader, desc="Predicting"):
                outputs = self.model(batch_seq, batch_static)
                all_probs.extend(outputs.cpu().numpy())
        
        probabilities = np.array(all_probs).flatten()
        predictions = (probabilities >= 0.5).astype(int)
        
        self.logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions, probabilities
    
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
            'ohlcv_columns': self.ohlcv_columns,
            'static_columns': self.static_columns,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        
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
        self.n_static_features = int(
            checkpoint.get('n_static_features', self.n_static_features)
        )
        
        if 'ohlcv_columns' in checkpoint:
            self.ohlcv_columns = checkpoint['ohlcv_columns']
        if 'static_columns' in checkpoint:
            self.static_columns = checkpoint['static_columns']
        
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
            logger.error(f"Test data not found")
            return 1
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Generate predictions
        predictions, probabilities = model.predict(
            test_df=test_df,
            batch_size=args.batch_size
        )
        
        # Save predictions
        output_path = os.path.join(args.checkpoint_dir, 'hybrid_predictions.csv')
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Calculate accuracy if targets are available
        if 'target' in test_df.columns:
            _, _, y_test = model.prepare_data(test_df)
            if len(y_test) == len(predictions):
                accuracy = (predictions == y_test).mean()
                logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    if not args.train and not args.predict:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
