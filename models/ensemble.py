#!/usr/bin/env python3
"""
Ensemble Model combining BGRU and XGBoost for BankNifty Directional Prediction.

This module implements an ensemble architecture that combines:
- Model 1: Trained BGRU model (from Phase 2)
- Model 2: XGBoost classifier on same features
- Fusion: Weighted average, stacking, or voting

The ensemble leverages the sequential modeling capability of BGRU with
the gradient boosting power of XGBoost for improved prediction accuracy.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports when running as script
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BGRU hybrid model
from models.bgru_hybrid import (
    OHLCV_FEATURES,
    PRICE_ACTION_FEATURES,
    TECHNICAL_FEATURES,
    TEMPORAL_FEATURES,
    HybridBGRUModel,
    get_static_features,
)


def load_hyperparameters(
    hyperparams_path: str = 'models/checkpoints/best_hyperparams.json'
) -> Optional[Dict[str, Any]]:
    """
    Load best hyperparameters from a JSON file.
    
    This function reads hyperparameters saved by the hyperparameter optimization
    process and returns them as a dictionary.
    
    Args:
        hyperparams_path: Path to the hyperparameters JSON file
        
    Returns:
        Dictionary of hyperparameters, or None if file doesn't exist
    """
    if os.path.exists(hyperparams_path):
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
            logging.getLogger(__name__).info(
                f"Loaded hyperparameters from {hyperparams_path}"
            )
            return hyperparams
        except (json.JSONDecodeError, IOError) as e:
            logging.getLogger(__name__).warning(
                f"Could not load hyperparameters from {hyperparams_path}: {e}"
            )
            return None
    else:
        logging.getLogger(__name__).info(
            f"No hyperparameters file found at {hyperparams_path}, using defaults"
        )
        return None


def load_selected_features(
    selected_features_path: str = 'models/checkpoints/selected_features.txt'
) -> Optional[List[str]]:
    """
    Load selected features from a text file.
    
    This function reads the list of selected features saved by the feature
    selection process (feature_selection.py). Each line in the file should
    contain one feature name.
    
    Args:
        selected_features_path: Path to the selected features text file
        
    Returns:
        List of feature names, or None if file doesn't exist
    """
    logger = logging.getLogger(__name__)
    
    if os.path.exists(selected_features_path):
        try:
            with open(selected_features_path, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            
            if features:
                logger.info(
                    f"Loaded {len(features)} selected features from {selected_features_path}"
                )
                return features
            else:
                logger.warning(
                    f"Selected features file is empty: {selected_features_path}"
                )
                return None
        except IOError as e:
            logger.warning(
                f"Could not load selected features from {selected_features_path}: {e}"
            )
            return None
    else:
        logger.info(
            f"No selected features file found at {selected_features_path}, using all features"
        )
        return None


def get_all_features() -> List[str]:
    """Return all feature names for XGBoost training."""
    return OHLCV_FEATURES + TECHNICAL_FEATURES + TEMPORAL_FEATURES + PRICE_ACTION_FEATURES


class BGRUXGBoostEnsemble:
    """
    Ensemble model combining Hybrid BGRU and XGBoost for directional prediction.
    
    Ensemble Strategy:
        Model 1: Trained BGRU model (HybridBGRUModel from Phase 2)
        Model 2: XGBoost classifier on same features
        Fusion: Weighted average, stacking, or voting
    
    Attributes:
        bgru_model: HybridBGRUModel instance
        xgb_model: XGBoost classifier
        stacking_model: LogisticRegression for stacking ensemble
        weights: Ensemble weights for weighted average
        sequence_length: Sequence length for BGRU input
        feature_columns: List of feature column names for XGBoost
    """
    
    def __init__(
        self,
        bgru_model_path: str,
        n_features: Optional[int] = None,
        sequence_length: int = 60,
        device: Optional[str] = None,
        hyperparams_path: Optional[str] = None,
        selected_features_path: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the Ensemble model.
        
        Ensemble Strategy:
            Model 1: Trained BGRU model (from Phase 2)
            Model 2: XGBoost classifier on same features
            Fusion: Weighted average or stacking
        
        Args:
            bgru_model_path: Path to the trained BGRU model checkpoint
            n_features: Number of features for XGBoost. If None, determined from data.
            sequence_length: Sequence length for BGRU input (default: 60)
            device: Device for BGRU model ('cuda', 'cpu', or None for auto-detect)
            hyperparams_path: Path to hyperparameters JSON file. If provided,
                hyperparameters will be loaded from this file.
            selected_features_path: Path to selected features text file. If provided,
                only these features will be used for XGBoost training.
            hidden_dim: Hidden dimension for BGRU (default: 128, overridden by file)
            num_layers: Number of GRU layers (default: 2, overridden by file)
            dropout: Dropout rate (default: 0.3, overridden by file)
        """
        self.bgru_model_path = bgru_model_path
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load hyperparameters from file if path provided
        loaded_hyperparams = None
        if hyperparams_path:
            loaded_hyperparams = load_hyperparameters(hyperparams_path)
        
        # Use loaded hyperparameters if available, otherwise use provided defaults
        if loaded_hyperparams:
            hidden_dim = loaded_hyperparams.get('hidden_dim', hidden_dim)
            num_layers = loaded_hyperparams.get('num_layers', num_layers)
            dropout = loaded_hyperparams.get('dropout', dropout)
            # Store any additional hyperparams for reference
            self.hyperparams = loaded_hyperparams
            self.logger.info(
                f"Using hyperparameters: hidden_dim={hidden_dim}, "
                f"num_layers={num_layers}, dropout={dropout}"
            )
        else:
            self.hyperparams = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout
            }
        
        # Initialize BGRU model with hyperparameters
        # n_static_features will be updated when model is loaded or data is prepared
        self.bgru_model = HybridBGRUModel(
            sequence_length=sequence_length,
            device=device,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Load trained BGRU model (this will also set correct n_static_features)
        if os.path.exists(bgru_model_path):
            self.bgru_model.load_model(bgru_model_path)
            self.logger.info(f"Loaded BGRU model from {bgru_model_path}")
        else:
            self.logger.warning(
                f"BGRU model not found at {bgru_model_path}. "
                "Model must be trained or provided before making predictions."
            )
        
        # Initialize XGBoost model (will be trained later)
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        
        # Initialize stacking model
        self.stacking_model: Optional[LogisticRegression] = None
        
        # Default ensemble weights: BGRU (60%), XGBoost (40%)
        self.weights: List[float] = [0.6, 0.4]
        
        # Load selected features if path provided, otherwise use all features
        self.selected_features_path = selected_features_path
        loaded_features = None
        if selected_features_path:
            loaded_features = load_selected_features(selected_features_path)
        
        if loaded_features:
            self.feature_columns: List[str] = loaded_features
            self.logger.info(
                f"Using {len(loaded_features)} selected features for XGBoost"
            )
        else:
            self.feature_columns = get_all_features()
            self.logger.info(
                f"Using all {len(self.feature_columns)} features for XGBoost"
            )
        
        self.logger.info(
            f"Initialized BGRUXGBoostEnsemble with sequence_length={sequence_length}"
        )
    
    def _prepare_xgb_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for XGBoost training/prediction.
        
        Uses OHLCV + technical + temporal + price action features.
        
        Args:
            df: DataFrame with feature columns
        
        Returns:
            X: Feature array of shape [num_samples, num_features]
            y: Target array of shape [num_samples] if 'target' column exists
        """
        # Get available feature columns
        available_cols = [c for c in self.feature_columns if c in df.columns]
        
        if len(available_cols) == 0:
            raise ValueError(
                f"No feature columns found in DataFrame. "
                f"Expected columns like: {self.feature_columns[:5]}..."
            )
        
        missing_cols = set(self.feature_columns) - set(available_cols)
        if missing_cols:
            self.logger.warning(f"Missing {len(missing_cols)} columns: {list(missing_cols)[:5]}...")
        
        # Update feature columns to available ones
        self.feature_columns = available_cols
        
        # Extract features
        X = df[available_cols].values.astype(np.float32)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract target if available
        if 'target' in df.columns:
            y = df['target'].values.astype(np.float32)
        else:
            y = None
        
        return X, y
    
    def train_xgboost(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Train XGBoost on technical + temporal + price action features.
        
        Hyperparameters:
            - max_depth: 6
            - learning_rate: 0.1
            - n_estimators: 100
            - subsample: 0.8
            - colsample_bytree: 0.8
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            max_depth: Maximum depth of trees (default: 6)
            learning_rate: Learning rate (default: 0.1)
            n_estimators: Number of boosting rounds (default: 100)
            subsample: Subsample ratio of training instances (default: 0.8)
            colsample_bytree: Subsample ratio of columns (default: 0.8)
            random_state: Random seed (default: 42)
            verbose: Whether to print training progress (default: False)
        
        Returns:
            Dictionary containing training and validation metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("Training XGBoost Model")
        self.logger.info("=" * 60)
        
        # Prepare features
        X_train, y_train = self._prepare_xgb_features(train_df)
        X_val, y_val = self._prepare_xgb_features(val_df)
        
        if y_train is None or y_val is None:
            raise ValueError("Target column 'target' not found in DataFrame")
        
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Validation samples: {len(X_val)}")
        self.logger.info(f"Features: {len(self.feature_columns)}")
        
        # Calculate class weights
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.logger.info(f"Class distribution: {n_neg} negative, {n_pos} positive")
        self.logger.info(f"Scale pos weight: {scale_pos_weight:.4f}")
        
        # Initialize XGBoost classifier
        self.xgb_model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        # Train with early stopping
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=verbose
        )
        
        # Evaluate
        train_preds = self.xgb_model.predict(X_train)
        val_preds = self.xgb_model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        
        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
        
        self.logger.info(f"XGBoost Train Accuracy: {train_acc:.4f}")
        self.logger.info(f"XGBoost Validation Accuracy: {val_acc:.4f}")
        self.logger.info("=" * 60)
        
        return metrics
    
    def _get_bgru_predictions(
        self,
        df: pd.DataFrame,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Get probability predictions from BGRU model.
        
        Args:
            df: DataFrame with features
            batch_size: Batch size for inference
        
        Returns:
            Probability array of shape [num_samples]
        """
        if self.bgru_model.model is None:
            raise ValueError("BGRU model not loaded or trained")
        
        _, probabilities = self.bgru_model.predict(df, batch_size=batch_size)
        return probabilities
    
    def _get_xgb_predictions(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Get probability predictions from XGBoost model.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Probability array of shape [num_samples]
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
        
        X, _ = self._prepare_xgb_features(df)
        
        # Align with BGRU output: 
        # BGRU outputs len(df) - sequence_length predictions
        # Starting from feature index sequence_length - 1
        # Ending at feature index len(df) - 2 (since BGRU's last target is at len(df)-1)
        n_bgru_samples = len(df) - self.sequence_length
        X_aligned = X[self.sequence_length - 1:self.sequence_length - 1 + n_bgru_samples]
        
        probabilities = self.xgb_model.predict_proba(X_aligned)[:, 1]
        return probabilities
    
    def predict_ensemble(
        self,
        test_df: pd.DataFrame,
        method: str = 'weighted',
        weights: Optional[List[float]] = None,
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions using specified method.
        
        Ensemble methods:
            1. Weighted average: BGRU (60%) + XGBoost (40%)
            2. Stacking: Use logistic regression on predictions
            3. Voting: Majority vote
        
        Args:
            test_df: Test DataFrame
            method: Ensemble method ('weighted', 'stacking', or 'voting')
            weights: Ensemble weights for weighted method [bgru_weight, xgb_weight].
                    If None, uses self.weights.
            batch_size: Batch size for BGRU inference
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Combined probability scores
        """
        if weights is None:
            weights = self.weights
        
        # Get predictions from both models
        bgru_probs = self._get_bgru_predictions(test_df, batch_size)
        xgb_probs = self._get_xgb_predictions(test_df)
        
        # Align predictions (both should have same length now)
        min_len = min(len(bgru_probs), len(xgb_probs))
        bgru_probs = bgru_probs[:min_len]
        xgb_probs = xgb_probs[:min_len]
        
        if method == 'weighted':
            # Weighted average
            if len(weights) != 2:
                raise ValueError("Weights must have exactly 2 elements [bgru_weight, xgb_weight]")
            
            probabilities = weights[0] * bgru_probs + weights[1] * xgb_probs
            predictions = (probabilities >= 0.5).astype(int)
            
        elif method == 'stacking':
            # Stacking with logistic regression
            if self.stacking_model is None:
                raise ValueError(
                    "Stacking model not trained. Call optimize_weights first "
                    "or train_stacking_model."
                )
            
            # Stack predictions as features
            stacked_features = np.column_stack([bgru_probs, xgb_probs])
            probabilities = self.stacking_model.predict_proba(stacked_features)[:, 1]
            predictions = self.stacking_model.predict(stacked_features)
            
        elif method == 'voting':
            # Majority voting
            bgru_votes = (bgru_probs >= 0.5).astype(int)
            xgb_votes = (xgb_probs >= 0.5).astype(int)
            
            # Sum votes (with optional weighting)
            vote_sum = weights[0] * bgru_votes + weights[1] * xgb_votes
            predictions = (vote_sum >= 0.5).astype(int)
            
            # Probabilities as weighted average of probabilities
            probabilities = weights[0] * bgru_probs + weights[1] * xgb_probs
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}. "
                           "Choose from 'weighted', 'stacking', or 'voting'.")
        
        self.logger.info(f"Generated {len(predictions)} ensemble predictions using {method} method")
        
        return predictions, probabilities
    
    def optimize_weights(
        self,
        val_df: pd.DataFrame,
        batch_size: int = 64,
        method: str = 'grid_search'
    ) -> List[float]:
        """
        Find optimal ensemble weights using validation set.
        
        Also trains a stacking model (logistic regression) for the
        stacking ensemble method.
        
        Args:
            val_df: Validation DataFrame
            batch_size: Batch size for BGRU inference
            method: Optimization method ('grid_search' or 'scipy')
        
        Returns:
            Optimal weights [bgru_weight, xgb_weight]
        """
        self.logger.info("=" * 60)
        self.logger.info("Optimizing Ensemble Weights")
        self.logger.info("=" * 60)
        
        # Get predictions from both models
        bgru_probs = self._get_bgru_predictions(val_df, batch_size)
        xgb_probs = self._get_xgb_predictions(val_df)
        
        # Both should now have the same length due to proper alignment
        # But take min_len just in case
        min_len = min(len(bgru_probs), len(xgb_probs))
        bgru_probs = bgru_probs[:min_len]
        xgb_probs = xgb_probs[:min_len]
        
        # Get aligned targets - BGRU outputs predictions for targets starting at 
        # index sequence_length - 1, with len(df) - sequence_length total predictions
        n_samples = len(val_df) - self.sequence_length
        targets = val_df['target'].values[self.sequence_length - 1:self.sequence_length - 1 + n_samples]
        targets = targets[:min_len]
        
        # Log individual model performance
        bgru_acc = accuracy_score(targets, (bgru_probs >= 0.5).astype(int))
        xgb_acc = accuracy_score(targets, (xgb_probs >= 0.5).astype(int))
        self.logger.info(f"BGRU Validation Accuracy: {bgru_acc:.4f}")
        self.logger.info(f"XGBoost Validation Accuracy: {xgb_acc:.4f}")
        
        # Train stacking model
        self.logger.info("Training stacking model...")
        stacked_features = np.column_stack([bgru_probs, xgb_probs])
        self.stacking_model = LogisticRegression(random_state=42, max_iter=1000)
        self.stacking_model.fit(stacked_features, targets)
        stacking_acc = accuracy_score(targets, self.stacking_model.predict(stacked_features))
        self.logger.info(f"Stacking model accuracy: {stacking_acc:.4f}")
        
        # Optimize weighted average
        best_weights = [0.6, 0.4]
        best_acc = 0.0
        
        if method == 'grid_search':
            # Grid search over weight combinations
            for w in np.arange(0.0, 1.01, 0.05):
                weights = [w, 1 - w]
                combined = weights[0] * bgru_probs + weights[1] * xgb_probs
                preds = (combined >= 0.5).astype(int)
                acc = accuracy_score(targets, preds)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights.copy()
                    
        elif method == 'scipy':
            # Scipy optimization
            def neg_accuracy(w):
                w_bgru = w[0]
                w_xgb = 1 - w_bgru
                combined = w_bgru * bgru_probs + w_xgb * xgb_probs
                preds = (combined >= 0.5).astype(int)
                return -accuracy_score(targets, preds)
            
            result = minimize(
                neg_accuracy,
                x0=[0.5],
                bounds=[(0, 1)],
                method='L-BFGS-B'
            )
            
            best_weights = [result.x[0], 1 - result.x[0]]
            best_acc = -result.fun
        
        self.weights = best_weights
        
        self.logger.info("-" * 60)
        self.logger.info(f"Optimal weights: BGRU={best_weights[0]:.4f}, XGBoost={best_weights[1]:.4f}")
        self.logger.info(f"Weighted average accuracy: {best_acc:.4f}")
        self.logger.info("=" * 60)
        
        return best_weights
    
    def save_ensemble(
        self,
        path: str = 'models/checkpoints/ensemble_model.pkl'
    ) -> None:
        """
        Save the ensemble model and components.
        
        Saves:
            - XGBoost model
            - Stacking model
            - Ensemble weights
            - Feature columns
            - Model configuration
        
        Args:
            path: Path to save the ensemble model
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained. Call train_xgboost first.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        ensemble_data = {
            'xgb_model': self.xgb_model,
            'stacking_model': self.stacking_model,
            'weights': self.weights,
            'feature_columns': self.feature_columns,
            'bgru_model_path': self.bgru_model_path,
            'sequence_length': self.sequence_length,
            'n_features': len(self.feature_columns),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        self.logger.info(f"Ensemble model saved to {path}")
    
    def load_ensemble(self, path: str) -> None:
        """
        Load a saved ensemble model.
        
        Args:
            path: Path to the saved ensemble model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ensemble model not found at {path}")
        
        with open(path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.xgb_model = ensemble_data['xgb_model']
        self.stacking_model = ensemble_data.get('stacking_model')
        self.weights = ensemble_data.get('weights', [0.6, 0.4])
        self.feature_columns = ensemble_data.get('feature_columns', get_all_features())
        
        # Update BGRU model path if different
        if 'bgru_model_path' in ensemble_data:
            new_bgru_path = ensemble_data['bgru_model_path']
            if new_bgru_path != self.bgru_model_path and os.path.exists(new_bgru_path):
                self.bgru_model_path = new_bgru_path
                self.bgru_model.load_model(new_bgru_path)
        
        self.logger.info(f"Ensemble model loaded from {path}")
        self.logger.info(f"Weights: BGRU={self.weights[0]:.4f}, XGBoost={self.weights[1]:.4f}")
    
    def save_xgboost(
        self,
        path: str = 'models/checkpoints/xgboost_model.pkl'
    ) -> None:
        """
        Save XGBoost model separately.
        
        Args:
            path: Path to save the XGBoost model
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        self.logger.info(f"XGBoost model saved to {path}")
    
    def save_weights(
        self,
        path: str = 'models/ensemble_weights.json'
    ) -> None:
        """
        Save ensemble weights to JSON file.
        
        Args:
            path: Path to save the weights
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        weights_data = {
            'bgru_weight': self.weights[0],
            'xgb_weight': self.weights[1],
            'stacking_trained': self.stacking_model is not None,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        self.logger.info(f"Ensemble weights saved to {path}")
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        method: str = 'weighted',
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate ensemble model on test data.
        
        Args:
            test_df: Test DataFrame
            method: Ensemble method ('weighted', 'stacking', or 'voting')
            batch_size: Batch size for inference
        
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions, probabilities = self.predict_ensemble(
            test_df, method=method, batch_size=batch_size
        )
        
        # Get aligned targets - matches BGRU output alignment
        n_samples = len(test_df) - self.sequence_length
        targets = test_df['target'].values[self.sequence_length - 1:self.sequence_length - 1 + n_samples]
        targets = targets[:len(predictions)]
        
        accuracy = accuracy_score(targets, predictions)
        
        # Calculate individual model accuracies
        bgru_probs = self._get_bgru_predictions(test_df, batch_size)[:len(predictions)]
        xgb_probs = self._get_xgb_predictions(test_df)[:len(predictions)]
        
        bgru_acc = accuracy_score(targets, (bgru_probs >= 0.5).astype(int))
        xgb_acc = accuracy_score(targets, (xgb_probs >= 0.5).astype(int))
        
        metrics = {
            'ensemble_accuracy': accuracy,
            'bgru_accuracy': bgru_acc,
            'xgb_accuracy': xgb_acc,
            'ensemble_method': method,
            'weights': self.weights
        }
        
        self.logger.info("-" * 60)
        self.logger.info(f"Evaluation Results ({method} method)")
        self.logger.info("-" * 60)
        self.logger.info(f"BGRU Accuracy: {bgru_acc:.4f}")
        self.logger.info(f"XGBoost Accuracy: {xgb_acc:.4f}")
        self.logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        self.logger.info("-" * 60)
        
        return metrics


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
        description='Train and evaluate BGRU+XGBoost ensemble model'
    )
    parser.add_argument(
        '--bgru_model',
        type=str,
        default='models/checkpoints/bgru_hybrid.pt',
        help='Path to trained BGRU model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/train_final.csv',
        help='Path to training data CSV (or directory with train/val/test.csv)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory containing train.csv, val.csv, test.csv'
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
        help='Directory for model checkpoints'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='models/logs/',
        help='Directory for training logs'
    )
    parser.add_argument(
        '--ensemble_method',
        type=str,
        default='weighted',
        choices=['weighted', 'stacking', 'voting'],
        help='Ensemble method (default: weighted)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the XGBoost component and optimize ensemble weights'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate ensemble on test data'
    )
    parser.add_argument(
        '--hyperparams_path',
        type=str,
        default=None,
        help='Path to best hyperparameters JSON file (from optimization)'
    )
    parser.add_argument(
        '--selected_features_path',
        type=str,
        default=None,
        help='Path to selected features text file (from feature selection)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif os.path.isdir(args.data):
        data_dir = args.data
    else:
        data_dir = os.path.dirname(args.data)
    
    # Initialize ensemble
    logger.info("=" * 60)
    logger.info("BGRU + XGBoost Ensemble Model")
    logger.info("=" * 60)
    
    # Default hyperparams path if not specified
    hyperparams_path = args.hyperparams_path
    if hyperparams_path is None:
        default_path = os.path.join(args.checkpoint_dir, 'best_hyperparams.json')
        if os.path.exists(default_path):
            hyperparams_path = default_path
            logger.info(f"Using default hyperparameters file: {hyperparams_path}")
    
    # Default selected features path if not specified
    selected_features_path = args.selected_features_path
    if selected_features_path is None:
        default_path = os.path.join(args.checkpoint_dir, 'selected_features.txt')
        if os.path.exists(default_path):
            selected_features_path = default_path
            logger.info(f"Using default selected features file: {selected_features_path}")
    
    ensemble = BGRUXGBoostEnsemble(
        bgru_model_path=args.bgru_model,
        sequence_length=args.sequence_length,
        hyperparams_path=hyperparams_path,
        selected_features_path=selected_features_path
    )
    
    if args.train:
        # Load training and validation data
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'val.csv')
        
        # Try alternative names
        if not os.path.exists(train_path):
            train_path = os.path.join(data_dir, 'train_final.csv')
        if not os.path.exists(val_path):
            val_path = os.path.join(data_dir, 'val_final.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            logger.error(f"Training/validation data not found in {data_dir}")
            return 1
        
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        
        logger.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Train XGBoost
        ensemble.train_xgboost(train_df, val_df)
        
        # Optimize weights
        optimal_weights = ensemble.optimize_weights(val_df)
        
        # Save models
        xgb_path = os.path.join(args.checkpoint_dir, 'xgboost_model.pkl')
        ensemble_path = os.path.join(args.checkpoint_dir, 'ensemble_model.pkl')
        weights_path = 'models/ensemble_weights.json'
        
        ensemble.save_xgboost(xgb_path)
        ensemble.save_ensemble(ensemble_path)
        ensemble.save_weights(weights_path)
        
        logger.info("Training complete!")
        logger.info(f"XGBoost model saved to: {xgb_path}")
        logger.info(f"Ensemble model saved to: {ensemble_path}")
        logger.info(f"Weights saved to: {weights_path}")
    
    if args.evaluate:
        # Load test data
        test_path = os.path.join(data_dir, 'test.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join(data_dir, 'test_final.csv')
        
        if not os.path.exists(test_path):
            logger.error(f"Test data not found in {data_dir}")
            return 1
        
        # Load ensemble if not trained
        if ensemble.xgb_model is None:
            ensemble_path = os.path.join(args.checkpoint_dir, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                ensemble.load_ensemble(ensemble_path)
            else:
                logger.error("Ensemble model not found. Train first using --train flag.")
                return 1
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        
        # Evaluate
        metrics = ensemble.evaluate(test_df, method=args.ensemble_method)
        
        # Save results
        results_path = os.path.join(args.checkpoint_dir, 'ensemble_evaluation.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = {k: float(v) if isinstance(v, np.floating) else v 
                          for k, v in metrics.items()}
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Evaluation results saved to {results_path}")
    
    if not args.train and not args.evaluate:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
