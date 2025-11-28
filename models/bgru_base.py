#!/usr/bin/env python3
"""
Bidirectional GRU Model for BankNifty Directional Prediction.

This module implements a BGRU-based neural network for predicting
the directional movement of BankNifty index using OHLCV sequences.
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
        dropout: float = 0.3
    ):
        """
        Initialize the BGRU model.
        
        Args:
            input_dim: Number of input features (default: 5 for OHLCV)
            hidden_dim: Hidden dimension for first BGRU layer (default: 128)
            num_layers: Number of GRU layers (default: 2)
            dropout: Dropout rate for GRU layers (default: 0.3)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First BGRU layer: 128 units bidirectional
        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second BGRU layer: 64 units bidirectional
        # Input is hidden_dim * 2 (bidirectional output from first layer)
        self.gru2 = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim // 2,  # 64 units
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Dense layer: 32 units with ReLU
        # Input is (hidden_dim // 2) * 2 = 128 (bidirectional output from second layer)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer: 1 unit with Sigmoid
        self.fc2 = nn.Linear(32, 1)
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
        out, _ = self.gru2(out)
        out = self.dropout2(out)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        # Output layer
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class BGRUPredictor:
    """
    Bidirectional GRU predictor for BankNifty directional prediction.
    
    This class handles data preparation, model training, prediction,
    and model persistence.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the BGRU predictor.
        
        Args:
            input_dim: Number of input features (default: 5 for OHLCV)
            hidden_dim: Hidden dimension for BGRU layers (default: 128)
            num_layers: Number of GRU layers (default: 2)
            dropout: Dropout rate (default: 0.3)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.input_dim = input_dim
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
        window: int = 100
    ) -> pd.DataFrame:
        """
        Normalizes OHLCV data using rolling z-score.
        
        Args:
            df: DataFrame with OHLCV columns
            method: Normalization method (only 'rolling_zscore' supported)
            window: Rolling window size for z-score calculation
        
        Returns:
            Normalized DataFrame
        """
        if method != 'rolling_zscore':
            raise ValueError(f"Unsupported normalization method: {method}")
        
        df = df.copy()
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure columns exist
        available_cols = [col for col in ohlcv_cols if col in df.columns]
        if not available_cols:
            raise ValueError("No OHLCV columns found in DataFrame")
        
        empty_cols = []
        for col in available_cols:
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
            normalized = normalized.ffill().bfill().fillna(0.0)
            df[col] = normalized
        
        if empty_cols:
            self.logger.warning(
                "Found columns with no valid values after normalization: %s. Filling with zeros.",
                empty_cols
            )
        
        # Handle any remaining NaN values outside OHLCV columns
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        
        return df
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares sequential data for BGRU.
        
        Creates sliding windows of specified length for OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns and 'target' column
            sequence_length: Number of time steps in each sequence (default: 60)
        
        Returns:
            X: Sequences array of shape [num_samples, sequence_length, 5]
            y: Target array of shape [num_samples]
        """
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Normalize data
        df_norm = self.normalize_data(df, method='rolling_zscore', window=100)
        
        # Extract OHLCV features and ensure they are finite
        features = df_norm[ohlcv_cols].values.astype(np.float32, copy=False)
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
        
        self.logger.info(f"Prepared {len(X)} sequences of length {sequence_length}")
        
        return X, y
    
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
            class_weights: Optional class weights for imbalanced data
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
        
        # Prepare data
        self.logger.info("Preparing training sequences...")
        X_train, y_train = self.prepare_sequences(train_df, sequence_length)
        self.logger.info("Preparing validation sequences...")
        X_val, y_val = self.prepare_sequences(val_df, sequence_length)
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient data for training. Need more samples than sequence_length.")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function - using BCELoss since model has sigmoid output
        # For class weights, we use pos_weight with BCELoss using reduction='none' and manual weighting
        if class_weights is not None:
            # Calculate positive weight for imbalanced data
            pos_weight = class_weights[1] / class_weights[0]
            
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
            self.model.parameters(),
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
        patience = 10
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
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
            
            for batch_X, batch_y in train_pbar:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
                
                for batch_X, batch_y in val_pbar:
                    outputs = self.model(batch_X)
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
            
            # Early stopping and checkpointing based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(checkpoint_dir, 'bgru_baseline.pt')
                self.save_model(checkpoint_path)
                self.logger.info(f"Saved best model with Val Acc: {val_acc:.4f}")
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
        
        # Remove file handler
        self.logger.removeHandler(file_handler)
        
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
        
        # Build model with loaded configuration
        self.build_model()
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Model loaded from {path}")


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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    # Initialize predictor
    predictor = BGRUPredictor()
    
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
        
        logger.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
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
