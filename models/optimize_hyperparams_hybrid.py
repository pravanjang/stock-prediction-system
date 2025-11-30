#!/usr/bin/env python3
"""
Hyperparameter Optimization for Hybrid BGRU Model.

This module implements Bayesian optimization using Optuna
to find optimal hyperparameters for the HybridBGRUModel.

Optimization targets:
- BGRU hidden dimensions: [64, 128, 256]
- Number of BGRU layers: [1, 2, 3]
- Dropout rates: [0.2, 0.3, 0.4, 0.5]
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [32, 64, 128]

Outputs:
- models/checkpoints/best_hyperparams.json
- models/checkpoints/optimization_history.csv
- models/checkpoints/optimized_bgru_hybrid.pt
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.trial import Trial
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.bgru_hybrid import (
    HybridBGRUModel,
    OHLCV_FEATURES,
    TECHNICAL_FEATURES,
    TEMPORAL_FEATURES,
    PRICE_ACTION_FEATURES,
    get_static_features,
)

# Constants
MIN_STD = 1e-8
IMBALANCE_THRESHOLD = 0.55  # 55:45 ratio threshold


def setup_logging(log_dir: str = 'models/logs/') -> logging.Logger:
    """
    Setup logging configuration for optimization.
    
    Args:
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('hyperopt_hybrid')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, 'hyperopt_hybrid.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger


def apply_class_balancing(train_df: pd.DataFrame) -> Optional[torch.Tensor]:
    """
    Check class distribution and calculate class weights if imbalanced.
    
    Class weights are calculated as: weight = total / (n_classes * count_per_class)
    
    Args:
        train_df: Training DataFrame with 'target' column
    
    Returns:
        Class weights tensor if imbalanced (>55:45), None otherwise
    """
    logger = logging.getLogger('hyperopt_hybrid')
    
    if 'target' not in train_df.columns:
        raise ValueError("DataFrame must contain 'target' column")
    
    # Calculate class distribution
    target_counts = train_df['target'].value_counts().sort_index()
    total = len(train_df)
    n_classes = len(target_counts)
    
    if n_classes != 2:
        logger.warning(f"Expected 2 classes, found {n_classes}")
        return None
    
    # Calculate ratios
    class_0_ratio = target_counts.iloc[0] / total
    class_1_ratio = target_counts.iloc[1] / total
    
    logger.info("=" * 60)
    logger.info("CLASS DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Class 0 (DOWN): {target_counts.iloc[0]} ({class_0_ratio*100:.2f}%)")
    logger.info(f"Class 1 (UP):   {target_counts.iloc[1]} ({class_1_ratio*100:.2f}%)")
    
    # Check if imbalanced (>55:45)
    max_ratio = max(class_0_ratio, class_1_ratio)
    is_imbalanced = max_ratio > IMBALANCE_THRESHOLD
    
    if is_imbalanced:
        logger.info(f"Dataset is imbalanced ({max_ratio*100:.1f}% > {IMBALANCE_THRESHOLD*100:.0f}%)")
        logger.info("Applying weighted loss for class balancing...")
        
        # Calculate class weights: weight = total / (n_classes * count_per_class)
        class_weights = []
        for i in range(n_classes):
            count = target_counts.iloc[i]
            weight = total / (n_classes * count)
            class_weights.append(weight)
        
        logger.info(f"Class weights: Class 0 = {class_weights[0]:.4f}, Class 1 = {class_weights[1]:.4f}")
        logger.info("=" * 60)
        
        return torch.tensor(class_weights, dtype=torch.float32)
    else:
        logger.info(f"Dataset is balanced ({max_ratio*100:.1f}% <= {IMBALANCE_THRESHOLD*100:.0f}%)")
        logger.info("No class balancing required.")
        logger.info("=" * 60)
        return None


def setup_optuna_study(
    study_name: str = "hybrid_bgru_optimization",
    direction: str = "maximize",
    storage: Optional[str] = None,
    load_if_exists: bool = True
) -> optuna.Study:
    """
    Setup Optuna study for Bayesian optimization.
    
    Args:
        study_name: Name of the study
        direction: Optimization direction ('maximize' for accuracy)
        storage: Optional storage URL for persistence
        load_if_exists: Whether to load existing study if found
    
    Returns:
        Configured Optuna study
    """
    logger = logging.getLogger('hyperopt_hybrid')
    
    # Create pruner for early stopping of unpromising trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5
    )
    
    # Create sampler for Bayesian optimization (TPE)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        pruner=pruner,
        sampler=sampler
    )
    
    logger.info(f"Created Optuna study: {study_name}")
    logger.info(f"Direction: {direction}")
    logger.info(f"Sampler: TPE (Bayesian)")
    logger.info(f"Pruner: MedianPruner")
    
    return study


class HybridOptunaObjective:
    """
    Optuna objective function wrapper for HybridBGRUModel optimization.
    
    This class encapsulates the training data and configuration
    needed to evaluate each hyperparameter trial.
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        class_weights: Optional[torch.Tensor] = None,
        sequence_length: int = 60,
        epochs: int = 20,
        device: Optional[str] = None
    ):
        """
        Initialize the objective function.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            class_weights: Optional class weights for imbalanced data
            sequence_length: Sequence length for BGRU input
            epochs: Number of training epochs per trial
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.train_df = train_df
        self.val_df = val_df
        self.class_weights = class_weights
        self.sequence_length = sequence_length
        self.epochs = epochs
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger('hyperopt_hybrid')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.best_hyperparams = None
    
    def __call__(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization of HybridBGRUModel.
        
        Optimizes:
        - BGRU hidden dimensions: [64, 128, 256]
        - Number of BGRU layers: [1, 2, 3]
        - Dropout rates: [0.2, 0.3, 0.4, 0.5]
        - Learning rate: [1e-4, 5e-4, 1e-3]
        - Batch size: [32, 64, 128]
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Validation accuracy (to be maximized)
        """
        # Sample hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5])
        lr = trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        self.logger.info(f"\nTrial {trial.number}: hidden_dim={hidden_dim}, "
                        f"num_layers={num_layers}, dropout={dropout}, "
                        f"lr={lr}, batch_size={batch_size}")
        
        try:
            # Create HybridBGRUModel with sampled hyperparameters
            model = HybridBGRUModel(
                sequence_length=self.sequence_length,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=str(self.device)
            )
            
            # Prepare data using the hybrid model's methods
            # This both prepares columns and builds the model
            X_price_train, X_static_train, y_train = model.prepare_data(self.train_df)
            X_price_val, X_static_val, y_val = model.prepare_data(self.val_df)
            
            # Build model after we know the data dimensions
            if model.model is None:
                model.build_model()
            
            if model.model is None:
                self.logger.warning("Failed to build model")
                return 0.0
            
            if len(X_price_train) == 0 or len(X_price_val) == 0:
                self.logger.warning("Insufficient data for training")
                return 0.0
            
            # Convert to tensors
            X_price_train_t = torch.FloatTensor(X_price_train).to(self.device)
            X_static_train_t = torch.FloatTensor(X_static_train).to(self.device)
            y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            
            X_price_val_t = torch.FloatTensor(X_price_val).to(self.device)
            X_static_val_t = torch.FloatTensor(X_static_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_price_train_t, X_static_train_t, y_train_t)
            val_dataset = TensorDataset(X_price_val_t, X_static_val_t, y_val_t)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # Setup loss function
            if self.class_weights is not None:
                class_weights_device = self.class_weights.to(self.device)
                pos_weight = class_weights_device[1] / class_weights_device[0]
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight], device=self.device)
                )
                use_logits = True
            else:
                criterion = nn.BCELoss()
                use_logits = False
            
            # Setup optimizer
            optimizer = torch.optim.Adam(
                model.model.parameters(),
                lr=lr,
                weight_decay=1e-5
            )
            
            # Learning rate scheduler
            scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )
            
            # Training loop
            model.model.train()
            best_val_acc = 0.0
            
            for epoch in range(self.epochs):
                # Training
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    X_price, X_static, y = batch
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if use_logits:
                        # Get logits before sigmoid
                        output = model.model(X_price, X_static)
                        loss = criterion(output, y)
                        preds = (torch.sigmoid(output) > 0.5).float()
                    else:
                        output = model.model(X_price, X_static)
                        loss = criterion(output, y)
                        preds = (output > 0.5).float()
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_correct += (preds == y).sum().item()
                    train_total += y.size(0)
                
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                
                # Validation
                model.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        X_price, X_static, y = batch
                        output = model.model(X_price, X_static)
                        preds = (output > 0.5).float()
                        val_correct += (preds == y).sum().item()
                        val_total += y.size(0)
                
                val_acc = val_correct / val_total if val_total > 0 else 0.0
                model.model.train()
                
                # Update scheduler
                scheduler.step(val_acc)
                
                # Track best
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # Report intermediate value for pruning
                trial.report(val_acc, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            self.logger.info(f"Trial {trial.number} completed: val_accuracy={best_val_acc:.4f}")
            
            # Track globally best model
            if best_val_acc > self.best_val_accuracy:
                self.best_val_accuracy = best_val_acc
                self.best_model_state = model.model.state_dict().copy()
                self.best_hyperparams = {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'val_accuracy': best_val_acc
                }
            
            return best_val_acc
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0


def run_optimization(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 50,
    epochs_per_trial: int = 20,
    sequence_length: int = 60,
    checkpoint_dir: str = 'models/checkpoints/',
    log_dir: str = 'models/logs/',
    device: Optional[str] = None
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run hyperparameter optimization for HybridBGRUModel.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        n_trials: Number of optimization trials
        epochs_per_trial: Training epochs per trial
        sequence_length: Sequence length for BGRU input
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        Tuple of (best_hyperparams, study)
    """
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("HYBRID BGRU HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Epochs per trial: {epochs_per_trial}")
    
    # Check class balance and get weights
    class_weights = apply_class_balancing(train_df)
    
    # Create Optuna study
    study = setup_optuna_study(study_name="hybrid_bgru_optimization")
    
    # Create objective function
    objective = HybridOptunaObjective(
        train_df=train_df,
        val_df=val_df,
        class_weights=class_weights,
        sequence_length=sequence_length,
        epochs=epochs_per_trial,
        device=device
    )
    
    # Run optimization
    logger.info("\nStarting optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Get best trial
    best_trial = study.best_trial
    best_hyperparams = {
        'hidden_dim': best_trial.params['hidden_dim'],
        'num_layers': best_trial.params['num_layers'],
        'dropout': best_trial.params['dropout'],
        'learning_rate': best_trial.params['learning_rate'],
        'batch_size': best_trial.params['batch_size'],
        'val_accuracy': best_trial.value,
        'trial_number': best_trial.number,
        'optimized_at': datetime.now().isoformat()
    }
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best validation accuracy: {best_trial.value:.4f}")
    logger.info(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Save best hyperparameters
    hyperparams_path = os.path.join(checkpoint_dir, 'best_hyperparams.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=2)
    logger.info(f"\nSaved best hyperparameters to {hyperparams_path}")
    
    # Save optimization history
    history_df = study.trials_dataframe()
    history_path = os.path.join(checkpoint_dir, 'optimization_history.csv')
    history_df.to_csv(history_path, index=False)
    logger.info(f"Saved optimization history to {history_path}")
    
    # Save best model if available
    if objective.best_model_state is not None:
        model_path = os.path.join(checkpoint_dir, 'optimized_bgru_hybrid.pt')
        
        # Create model with best hyperparameters to save properly
        best_model = HybridBGRUModel(
            sequence_length=sequence_length,
            hidden_dim=best_hyperparams['hidden_dim'],
            num_layers=best_hyperparams['num_layers'],
            dropout=best_hyperparams['dropout'],
            device=device
        )
        best_model.prepare_data(train_df)
        best_model.build_model()
        
        if best_model.model is not None:
            best_model.model.load_state_dict(objective.best_model_state)
            best_model.save_model(model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            logger.warning("Could not save best model - model initialization failed")
    
    return best_hyperparams, study


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for Hybrid BGRU model'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/',
        help='Directory containing train.csv, val.csv files'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Training epochs per trial (default: 20)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Sequence length for BGRU input (default: 60)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='models/checkpoints/',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='models/logs/',
        help='Directory for logs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Load data
    train_path = os.path.join(args.data_dir, 'train.csv')
    val_path = os.path.join(args.data_dir, 'val.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Error: train.csv or val.csv not found in {args.data_dir}")
        return 1
    
    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    
    print(f"Loading validation data from {val_path}")
    val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
    
    # Run optimization
    best_hyperparams, study = run_optimization(
        train_df=train_df,
        val_df=val_df,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs,
        sequence_length=args.sequence_length,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_hyperparams['val_accuracy']:.4f}")
    print(f"Best hyperparameters saved to: {args.checkpoint_dir}/best_hyperparams.json")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
