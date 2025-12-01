#!/usr/bin/env python3
"""
Hyperparameter Optimization for BGRU Model.

This module implements Bayesian optimization using Optuna
to find optimal hyperparameters for the BGRU model.

Optimization targets:
- BGRU hidden dimensions: [64, 128, 256]
- Number of BGRU layers: [1, 2, 3]
- Dropout rates: [0.2, 0.3, 0.4, 0.5]
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [32, 64, 128]

Outputs:
- models/best_hyperparams.json
- models/optimization_history.csv
- models/checkpoints/optimized_bgru.pt
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

from models.bgru_base import (
    BGRUModel,
    BGRUPredictor,
    OHLCV_FEATURES,
    TECHNICAL_FEATURES,
    TEMPORAL_FEATURES,
    PRICE_ACTION_FEATURES,
    get_feature_groups,
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
    
    logger = logging.getLogger('hyperopt')
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
    log_file = os.path.join(log_dir, 'hyperopt.log')
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
    logger = logging.getLogger('hyperopt')
    
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
    study_name: str = "bgru_optimization",
    direction: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    regression: bool = False
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
    logger = logging.getLogger('hyperopt')
    
    # Create pruner for early stopping of unpromising trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5
    )
    
    # Create sampler for Bayesian optimization (TPE)
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # Default direction: minimize for regression (RMSE), maximize for classification (accuracy)
    if direction is None:
        direction = 'minimize' if regression else 'maximize'

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


class OptunaObjective:
    """
    Optuna objective function wrapper class.
    
    This class encapsulates the training data and configuration
    needed to evaluate each hyperparameter trial.
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_groups: list,
        class_weights: Optional[torch.Tensor] = None,
        sequence_length: int = 60,
        epochs: int = 20,
        device: Optional[str] = None
        ,
        regression: bool = False
    ):
        """
        Initialize the objective function.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            feature_groups: List of feature groups to use
            class_weights: Optional class weights for imbalanced data
            sequence_length: Sequence length for BGRU input
            epochs: Number of training epochs per trial
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.train_df = train_df
        self.val_df = val_df
        self.feature_groups = feature_groups
        self.class_weights = class_weights
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.regression = regression
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger = logging.getLogger('hyperopt')
    
    def __call__(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        
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
            # Create predictor with sampled hyperparameters
            predictor = BGRUPredictor(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=str(self.device),
                feature_groups=self.feature_groups
            )
            
            # Build model
            predictor.build_model()
            
            # Prepare data
            X_train, y_train = predictor.prepare_sequences(
                self.train_df, self.sequence_length
            )
            X_val, y_val = predictor.prepare_sequences(
                self.val_df, self.sequence_length
            )
            
            if len(X_train) == 0 or len(X_val) == 0:
                self.logger.warning("Insufficient data for training")
                return 0.0
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # Setup loss function (classification or regression)
            if not self.regression:
                if self.class_weights is not None:
                    class_weights_device = self.class_weights.to(self.device)
                    class_0_weight = torch.clamp(class_weights_device[0], min=MIN_STD)
                    # Compute relative weight for class 1 samples
                    class_1_relative_weight = class_weights_device[1] / class_0_weight

                    def weighted_bce_loss(
                        outputs: torch.Tensor,
                        targets: torch.Tensor,
                        pos_weight: torch.Tensor = class_1_relative_weight
                    ) -> torch.Tensor:
                        # Use where for efficient weight computation
                        sample_weights = torch.where(
                            targets == 1,
                            pos_weight,
                            torch.ones(1, device=targets.device, dtype=targets.dtype)
                        )
                        bce = nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
                        return (bce * sample_weights).mean()

                    criterion = weighted_bce_loss
                else:
                    criterion = nn.BCELoss()
            else:
                criterion = nn.MSELoss()
            
            # Setup optimizer
            optimizer = torch.optim.Adam(
                predictor.model.parameters(),
                lr=lr,
                weight_decay=1e-5
            )
            
            # Training loop
            best_val_acc = 0.0
            best_val_rmse = float('inf')
            model = predictor.model
            
            for epoch in range(self.epochs):
                # Training phase
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss_accum = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        if self.regression:
                            loss_val = nn.functional.mse_loss(outputs, batch_y)
                            val_loss_accum += loss_val.item() * batch_y.size(0)
                            val_total += batch_y.size(0)
                        else:
                            predictions = (outputs >= 0.5).float()
                            val_correct += (predictions == batch_y).sum().item()
                            val_total += batch_y.size(0)
                
                if self.regression:
                    # compute RMSE over validation set
                    val_mse = val_loss_accum / val_total if val_total > 0 else float('inf')
                    val_rmse = float(np.sqrt(val_mse))
                    best_val_rmse = min(best_val_rmse, val_rmse)
                    # Report rmse (smaller is better)
                    trial.report(val_rmse, epoch)
                    val_metric_for_pruning = val_rmse
                else:
                    val_acc = val_correct / val_total
                    best_val_acc = max(best_val_acc, val_acc)
                    trial.report(val_acc, epoch)
                    val_metric_for_pruning = val_acc
                
                # Report intermediate value for pruning
                trial.report(val_metric_for_pruning, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
            
            if self.regression:
                self.logger.info(f"Trial {trial.number} completed with val_rmse={best_val_rmse:.4f}")
                return best_val_rmse
            else:
                self.logger.info(f"Trial {trial.number} completed with val_acc={best_val_acc:.4f}")
                return best_val_acc
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0


def objective_function(trial: Trial) -> float:
    """
    Standalone objective function for Optuna optimization.
    
    This function provides the optimization logic directly usable by Optuna.
    For full configuration, use the OptunaObjective class.
    
    Optimize:
    - BGRU hidden dimensions: [64, 128, 256]
    - Number of BGRU layers: [1, 2, 3]
    - Dropout rates: [0.2, 0.3, 0.4, 0.5]
    - Learning rate: [1e-4, 5e-4, 1e-3]
    - Batch size: [32, 64, 128]
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Validation accuracy (to be maximized)
    
    Note:
        This function requires the following user attributes on the trial:
        - 'train_df': Training DataFrame
        - 'val_df': Validation DataFrame
        - 'feature_groups': List of feature groups
        - 'class_weights': Optional class weights tensor
        - 'sequence_length': Sequence length (default: 60)
        - 'epochs': Number of epochs (default: 20)
        - 'device': Device to use (default: auto-detect)
    
    Example:
        >>> study = optuna.create_study(direction='maximize')
        >>> study.set_user_attr('train_df', train_df)
        >>> study.set_user_attr('val_df', val_df)
        >>> study.set_user_attr('feature_groups', ['ohlcv'])
        >>> study.optimize(objective_function, n_trials=30)
    """
    # Get data from study user attributes
    study = trial.study
    
    train_df = study.user_attrs.get('train_df')
    val_df = study.user_attrs.get('val_df')
    feature_groups = study.user_attrs.get('feature_groups', ['ohlcv'])
    class_weights = study.user_attrs.get('class_weights')
    regression = study.user_attrs.get('regression', False)
    sequence_length = study.user_attrs.get('sequence_length', 60)
    epochs = study.user_attrs.get('epochs', 20)
    device = study.user_attrs.get('device')
    
    if train_df is None or val_df is None:
        raise ValueError(
            "train_df and val_df must be set as study user attributes. "
            "Use study.set_user_attr('train_df', df) before optimization."
        )
    
    # Create objective instance and call it
    objective = OptunaObjective(
        train_df=train_df,
        val_df=val_df,
        feature_groups=feature_groups,
        class_weights=class_weights,
        sequence_length=sequence_length,
        epochs=epochs,
        device=device,
        regression=regression
    )
    
    return objective(trial)


def train_with_best_params(
    best_params: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_groups: list,
    class_weights: Optional[torch.Tensor] = None,
    sequence_length: int = 60,
    epochs: int = 50,
    checkpoint_dir: str = 'models/checkpoints/',
    log_dir: str = 'models/logs/',
    device: Optional[str] = None,
    regression: bool = False
) -> Tuple[BGRUPredictor, Dict[str, list]]:
    """
    Train BGRU model with the best hyperparameters.
    
    Args:
        best_params: Dictionary of best hyperparameters
        train_df: Training DataFrame
        val_df: Validation DataFrame
        feature_groups: List of feature groups to use
        class_weights: Optional class weights for imbalanced data
        sequence_length: Sequence length for BGRU input
        epochs: Number of training epochs
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
        device: Device to use
    
    Returns:
        Tuple of (trained predictor, training history)
    """
    logger = logging.getLogger('hyperopt')
    
    logger.info("=" * 60)
    logger.info("TRAINING WITH BEST HYPERPARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Parameters: {best_params}")
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract hyperparameters
    hidden_dim = best_params.get('hidden_dim', 128)
    num_layers = best_params.get('num_layers', 2)
    dropout = best_params.get('dropout', 0.3)
    lr = best_params.get('learning_rate', 0.001)
    batch_size = best_params.get('batch_size', 64)
    
    # Create predictor
    predictor = BGRUPredictor(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        feature_groups=feature_groups
        ,regression=regression
    )
    
    # Train with best parameters
    history = predictor.train(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        sequence_length=sequence_length,
        class_weights=class_weights,
        regression=regression,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Save optimized model
    optimized_path = os.path.join(checkpoint_dir, 'optimized_bgru.pt')
    predictor.save_model(optimized_path)
    logger.info(f"Optimized model saved to {optimized_path}")
    
    return predictor, history


def save_optimization_results(
    study: optuna.Study,
    output_dir: str = 'models/'
) -> None:
    """
    Save optimization results to files.
    
    Outputs:
    - best_hyperparams.json: Best hyperparameters
    - optimization_history.csv: Trial history
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save results
    """
    logger = logging.getLogger('hyperopt')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save best hyperparameters
    best_params = study.best_params
    best_params['best_value'] = study.best_value
    best_params['optimization_time'] = datetime.now().isoformat()
    
    hyperparams_path = os.path.join(output_dir, 'best_hyperparams.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best hyperparameters saved to {hyperparams_path}")
    
    # Save optimization history
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value if trial.value is not None else float('nan'),
            'state': trial.state.name,
            **trial.params
        }
        trials_data.append(trial_data)
    
    history_df = pd.DataFrame(trials_data)
    history_path = os.path.join(output_dir, 'optimization_history.csv')
    history_df.to_csv(history_path, index=False)
    logger.info(f"Optimization history saved to {history_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    # Best value depends on regression vs classification study direction and metric
    best_metric_label = 'val_rmse' if study.direction == optuna.study.StudyDirection.MINIMIZE else 'val_acc'
    try:
        logger.info(f"Best value ({best_metric_label}): {study.best_value:.4f}")
    except Exception:
        logger.info(f"Best value: {study.best_value}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def run_optimization(
    data_path: str,
    val_path: Optional[str] = None,
    n_trials: int = 30,
    epochs_per_trial: int = 20,
    final_epochs: int = 50,
    sequence_length: int = 60,
    feature_groups: Optional[list] = None,
    output_dir: str = 'models/',
    checkpoint_dir: str = 'models/checkpoints/',
    log_dir: str = 'models/logs/',
    regression: bool = False
) -> optuna.Study:
    """
    Run the complete hyperparameter optimization pipeline.
    
    Args:
        data_path: Path to training data CSV
        val_path: Path to validation data CSV (optional, will split if not provided)
        n_trials: Maximum number of optimization trials
        epochs_per_trial: Epochs for each trial (shorter for speed)
        final_epochs: Epochs for final training with best params
        sequence_length: Sequence length for BGRU
        feature_groups: Feature groups to use
        output_dir: Directory for output files
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for logs
    
    Returns:
        Completed Optuna study
    """
    # Setup logging
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("BGRU HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Validation path: {val_path}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Epochs per trial: {epochs_per_trial}")
    logger.info(f"Final epochs: {final_epochs}")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    train_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded training data: {len(train_df)} samples")
    
    # Load or split validation data
    if val_path and os.path.exists(val_path):
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded validation data: {len(val_df)} samples")
    else:
        # Split training data 80/20
        split_idx = int(len(train_df) * 0.8)
        val_df = train_df.iloc[split_idx:]
        train_df = train_df.iloc[:split_idx]
        logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}")
    
    # Set default feature groups
    if feature_groups is None:
        feature_groups = ['ohlcv']
    logger.info(f"Feature groups: {feature_groups}")
    
    # Check class distribution and apply balancing if needed (only for classification)
    class_weights = None
    if not regression:
        class_weights = apply_class_balancing(train_df)
    
    # Setup Optuna study
    study = setup_optuna_study(regression=regression)
    
    # Create objective function
    objective = OptunaObjective(
        train_df=train_df,
        val_df=val_df,
        feature_groups=feature_groups,
        class_weights=class_weights,
        sequence_length=sequence_length,
        regression=regression,
        epochs=epochs_per_trial
    )
    
    # Run optimization
    logger.info(f"\nStarting optimization with {n_trials} trials...")
    # Save regression flag as study attribute
    study.set_user_attr('regression', bool(regression))

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Save optimization results
    save_optimization_results(study, output_dir)
    
    # Train with best parameters
    predictor, history = train_with_best_params(
        best_params=study.best_params,
        train_df=train_df,
        val_df=val_df,
        feature_groups=feature_groups,
        class_weights=class_weights,
        sequence_length=sequence_length,
        epochs=final_epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
        ,
        regression=regression
    )
    
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best hyperparameters saved to: {output_dir}/best_hyperparams.json")
    logger.info(f"Optimization history saved to: {output_dir}/optimization_history.csv")
    logger.info(f"Optimized model saved to: {checkpoint_dir}/optimized_bgru.pt")
    
    return study


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for BGRU model'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data CSV (e.g., data/processed/train_final.csv)'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Path to validation data CSV (optional, will split if not provided)'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=30,
        help='Maximum number of optimization trials (default: 30)'
    )
    parser.add_argument(
        '--epochs_per_trial',
        type=int,
        default=20,
        help='Training epochs per trial (default: 20)'
    )
    parser.add_argument(
        '--final_epochs',
        type=int,
        default=50,
        help='Training epochs for final model (default: 50)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Sequence length for BGRU input (default: 60)'
    )
    parser.add_argument(
        '--feature_groups',
        type=str,
        nargs='+',
        default=['ohlcv'],
        choices=['ohlcv', 'technical', 'temporal', 'price_action'],
        help='Feature groups to use (default: ohlcv)'
    )
    parser.add_argument(
        '--all_features',
        action='store_true',
        help='Use all available features'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/',
        help='Directory for output files (default: models/)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='models/checkpoints/',
        help='Directory for model checkpoints (default: models/checkpoints/)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='models/logs/',
        help='Directory for logs (default: models/logs/)'
    )
    parser.add_argument(
        '--regression',
        action='store_true',
        help='Optimize regression task (minimize RMSE) instead of classification tasks (maximize accuracy)'
    )
    
    args = parser.parse_args()
    
    # Determine feature groups
    if args.all_features:
        feature_groups = ['ohlcv', 'technical', 'temporal', 'price_action']
    else:
        feature_groups = args.feature_groups
    
    # Run optimization
    study = run_optimization(
        data_path=args.data,
        val_path=args.val_data,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        final_epochs=args.final_epochs,
        sequence_length=args.sequence_length,
        feature_groups=feature_groups,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
        ,regression=args.regression
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
