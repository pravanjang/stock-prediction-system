#!/usr/bin/env python3
"""
Baseline Evaluation Script for BankNifty BGRU Model.

This script evaluates the trained BGRU model on test data and generates
comprehensive reports including metrics, plots, and trading simulation results.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bgru_base import BGRUPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_data(
    model_path: str,
    test_data_path: str,
    sequence_length: int = 60,
    force_regression: Optional[bool] = None
) -> Tuple[BGRUPredictor, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the trained model and test data.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        test_data_path: Path to the test CSV file
        sequence_length: Sequence length used during training
    
    Returns:
        Tuple of (predictor, test_df, y_true, y_pred, y_pred_proba)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Initialize predictor and load model
    predictor = BGRUPredictor()
    predictor.load_model(model_path)
    # If CLI forces regression vs classification, override predictor flag
    if force_regression is not None:
        predictor.regression = bool(force_regression)
        if hasattr(predictor.model, 'regression'):
            predictor.model.regression = predictor.regression
    
    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Generate predictions
    y_pred, y_pred_proba = predictor.predict(
        test_df=test_df,
        sequence_length=sequence_length
    )
    
    # Get ground truth labels
    _, y_true = predictor.prepare_sequences(test_df, sequence_length)
    
    # Ensure alignment
    min_len = min(len(y_true), len(y_pred))
    # For regression mode, keep y_true as float; for classification cast to int
    if getattr(predictor, 'regression', False):
        y_true = y_true[:min_len].astype(float)
    else:
        y_true = y_true[:min_len].astype(int)
    y_pred = y_pred[:min_len]
    y_pred_proba = y_pred_proba[:min_len]
    
    logger.info(f"Evaluation samples: {len(y_true)}")
    
    return predictor, test_df, y_true, y_pred, y_pred_proba


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_pred_proba: Prediction probabilities
    
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    # Basic classification metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # ROC-AUC score
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        # Handle case where only one class is present
        metrics['roc_auc'] = 0.5
        logger.warning("Could not compute ROC-AUC (only one class present)")
    
    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Class distribution in predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip(unique.astype(int).tolist(), counts.tolist()))
    metrics['prediction_distribution'] = pred_dist
    
    # Class distribution in ground truth
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    true_dist = dict(zip(unique_true.astype(int).tolist(), counts_true.tolist()))
    metrics['ground_truth_distribution'] = true_dist
    
    # Sample counts
    metrics['total_samples'] = int(len(y_true))
    metrics['correct_predictions'] = int((y_pred == y_true).sum())
    metrics['incorrect_predictions'] = int((y_pred != y_true).sum())
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics (MSE, RMSE, MAE, R2).
    """
    metrics = {}
    # Flatten to 1D arrays
    y_true = np.asarray(y_true).astype(float).flatten()
    y_pred = np.asarray(y_pred).astype(float).flatten()
    # Basic metrics
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics['mse'] = mse
    metrics['rmse'] = rmse
    metrics['mae'] = mae
    metrics['r2'] = r2
    metrics['total_samples'] = int(len(y_true))

    logger.info(f"Regression MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
) -> None:
    """
    Plot and save confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['DOWN (0)', 'UP (1)'],
        yticklabels=['DOWN (0)', 'UP (1)'],
        annot_kws={'size': 14}
    )
    plt.title('Confusion Matrix - BGRU Baseline', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_prediction_confidence(
    y_pred_proba: np.ndarray,
    save_path: str,
    y_true: Optional[np.ndarray] = None
) -> None:
    """
    Plot prediction confidence distribution.
    
    Args:
        y_pred_proba: Prediction probabilities
        save_path: Path to save the plot
        y_true: Optional ground truth for coloring by correctness
    """
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of prediction probabilities
    axes[0].hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[0].set_title('Prediction Probability Distribution', fontsize=12)
    axes[0].set_xlabel('Probability (P(UP))', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence by correctness (if ground truth provided)
    if y_true is not None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
        correct_mask = y_pred == y_true
        
        axes[1].hist(
            y_pred_proba[correct_mask], bins=30, alpha=0.6,
            label=f'Correct ({correct_mask.sum()})', color='green', edgecolor='black'
        )
        axes[1].hist(
            y_pred_proba[~correct_mask], bins=30, alpha=0.6,
            label=f'Incorrect ({(~correct_mask).sum()})', color='red', edgecolor='black'
        )
        axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
        axes[1].set_title('Confidence by Prediction Correctness', fontsize=12)
        axes[1].set_xlabel('Probability (P(UP))', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Box plot of probabilities
        axes[1].boxplot(y_pred_proba)
        axes[1].set_title('Probability Distribution Box Plot', fontsize=12)
        axes[1].set_ylabel('Probability', fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Prediction confidence plot saved to {save_path}")


def plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    save_path: str,
    sequence_length: int = 60
) -> None:
    """
    Plot regression results including predicted vs actual, residuals, and timeseries overlay.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter: Predicted vs Actual with identity line
    axes[0, 0].scatter(y_true, y_pred, s=8, alpha=0.6)
    minv = min(y_true.min(), y_pred.min())
    maxv = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([minv, maxv], [minv, maxv], 'r--')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].grid(True)

    # Residual histogram
    residuals = y_pred - y_true
    axes[0, 1].hist(residuals, bins=50, color='gray', edgecolor='black')
    axes[0, 1].set_title('Residuals (Predicted - Actual)')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].grid(True)

    # Time series overlay (subset to avoid large plots)
    ts_len = min(len(y_true), 200)
    t = np.arange(ts_len)
    axes[1, 0].plot(t, y_true[:ts_len], label='Actual')
    axes[1, 0].plot(t, y_pred[:ts_len], label='Predicted')
    axes[1, 0].set_title('Time Series: Actual vs Predicted (sample)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Scatter over time: predicted - actual
    axes[1, 1].plot(t, residuals[:ts_len], label='Residual')
    axes[1, 1].axhline(0, color='black', linestyle='--')
    axes[1, 1].set_title('Residuals over time (sample)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Regression plots saved to {save_path}")


def simulate_trading(
    df: pd.DataFrame,
    predictions: np.ndarray,
    regression: bool = False,
    sequence_length: int = 60,
    transaction_cost: float = 0.0003,
    lot_size: int = 15,  # BankNifty lot size
    tick_size: float = 0.05
) -> Dict[str, float]:
    """
    Simulate trading based on model predictions.
    
    Trading Strategy:
    - Prediction 1 (UP): Go LONG at close of prediction candle
    - Prediction 0 (DOWN): Go SHORT at close of prediction candle
    - Exit at close of next candle
    - Transaction cost: 0.03% per trade (entry + exit)
    
    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions (0 or 1)
        sequence_length: Sequence length used (to align predictions with data)
        transaction_cost: Transaction cost per trade (default: 0.03%)
        lot_size: Number of units per lot (default: 15 for BankNifty)
        tick_size: Minimum price movement
    
    Returns:
        Dictionary with trading simulation results
    """
    logger.info("Starting trading simulation...")
    
    # Align predictions with data (predictions start after sequence_length)
    # The prediction at index i corresponds to the close price at index (sequence_length + i - 1)
    start_idx = sequence_length - 1
    
    # Ensure we have enough data for exit
    max_trades = min(len(predictions), len(df) - start_idx - 1)
    
    trades = []
    cumulative_pnl = 0.0
    cumulative_returns = [0.0]
    peak_value = 0.0
    max_drawdown = 0.0
    
    for i in range(max_trades):
        data_idx = start_idx + i
        
        if data_idx + 1 >= len(df):
            break
        
        entry_price = df.iloc[data_idx]['close']
        exit_price = df.iloc[data_idx + 1]['close']
        prediction = predictions[i]

        # Determine trading signal. For regression, predictions are continuous
        # predicted next_close > entry_price => LONG, else SHORT
        if regression:
            try:
                # cast prediction to float
                pred_val = float(prediction)
                signal_val = 1 if pred_val > float(entry_price) else 0
            except Exception:
                signal_val = int(prediction)
        else:
            # classification predictions are expected as 0/1 labels
            signal_val = int(prediction)
        
        # Calculate raw PnL (per unit)
        if signal_val == 1:  # LONG
            raw_pnl = exit_price - entry_price
            direction = 'LONG'
        else:  # SHORT
            raw_pnl = entry_price - exit_price
            direction = 'SHORT'
        
        # Calculate return percentage
        raw_return = raw_pnl / entry_price
        
        # Apply transaction cost (entry + exit = 2 * transaction_cost)
        net_return = raw_return - (2 * transaction_cost)
        net_pnl = raw_pnl * lot_size - (2 * transaction_cost * entry_price * lot_size)
        
        # Track trade
        trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'raw_pnl': raw_pnl * lot_size,
            'net_pnl': net_pnl,
            'raw_return': raw_return,
            'net_return': net_return,
            'won': net_pnl > 0
        })
        
        cumulative_pnl += net_pnl
        cumulative_returns.append(cumulative_pnl)
        
        # Track max drawdown
        if cumulative_pnl > peak_value:
            peak_value = cumulative_pnl
        drawdown = peak_value - cumulative_pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate summary statistics
    if not trades:
        logger.warning("No trades executed in simulation")
        return {
            'total_trades': 0,
            'total_return': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'avg_trade_return': 0.0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
        }
    
    trades_df = pd.DataFrame(trades)
    
    results = {
        'total_trades': len(trades),
        'total_pnl': float(cumulative_pnl),
        'total_return': float(sum(t['net_return'] for t in trades) * 100),  # percentage
        'win_rate': float(trades_df['won'].mean() * 100),
        'max_drawdown': float(max_drawdown),
        'max_drawdown_pct': float(max_drawdown / peak_value * 100) if peak_value > 0 else 0.0,
        'avg_trade_pnl': float(trades_df['net_pnl'].mean()),
        'avg_trade_return': float(trades_df['net_return'].mean() * 100),
        'std_trade_return': float(trades_df['net_return'].std() * 100),
        'profitable_trades': int(trades_df['won'].sum()),
        'losing_trades': int((~trades_df['won']).sum()),
        'long_trades': int((trades_df['direction'] == 'LONG').sum()),
        'short_trades': int((trades_df['direction'] == 'SHORT').sum()),
        'best_trade': float(trades_df['net_pnl'].max()),
        'worst_trade': float(trades_df['net_pnl'].min()),
        'sharpe_ratio': float(
            trades_df['net_return'].mean() / trades_df['net_return'].std()
            if trades_df['net_return'].std() > 0 else 0.0
        ) * np.sqrt(252),  # Annualized
    }
    
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Total PnL: ₹{results['total_pnl']:,.2f}")
    logger.info(f"Win rate: {results['win_rate']:.2f}%")
    logger.info(f"Max drawdown: ₹{results['max_drawdown']:,.2f}")
    
    return results


def generate_report(
    metrics: Dict,
    trading_results: Dict,
    training_history_path: Optional[str] = None,
    save_path: str = 'evaluation/reports/baseline_report.txt',
    is_regression: bool = False
) -> str:
    """
    Generate a comprehensive text report of the evaluation.
    
    Args:
        metrics: Classification metrics dictionary
        trading_results: Trading simulation results dictionary
        training_history_path: Optional path to training history JSON
        save_path: Path to save the report
    
    Returns:
        Report text content
    """
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load training history if available
    train_acc = None
    val_acc = None
    if training_history_path and os.path.exists(training_history_path):
        with open(training_history_path, 'r') as f:
            history = json.load(f)
        if 'train_acc' in history and history['train_acc']:
            train_acc = history['train_acc'][-1]
        if 'val_acc' in history and history['val_acc']:
            val_acc = history['val_acc'][-1]
    
    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("BGRU BASELINE MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Metrics Section (classification or regression)
    lines.append("-" * 70)
    if not is_regression:
        lines.append("CLASSIFICATION METRICS")
        lines.append("-" * 70)
        lines.append(f"Total Samples:      {metrics.get('total_samples', 'N/A')}")
        lines.append(f"Correct Predictions: {metrics.get('correct_predictions', 'N/A')}")
        lines.append(f"Incorrect Predictions: {metrics.get('incorrect_predictions', 'N/A')}")
        lines.append("")
        lines.append(f"Accuracy:           {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
        lines.append(f"Precision:          {metrics.get('precision', 0):.4f}")
        lines.append(f"Recall:             {metrics.get('recall', 0):.4f}")
        lines.append(f"F1-Score:           {metrics.get('f1_score', 0):.4f}")
        lines.append(f"ROC-AUC:            {metrics.get('roc_auc', 0):.4f}")
        lines.append(f"Specificity:        {metrics.get('specificity', 0):.4f}")
    else:
        lines.append("REGRESSION METRICS")
        lines.append("-" * 70)
        lines.append(f"Total Samples:      {metrics.get('total_samples', 'N/A')}")
        lines.append("")
        lines.append(f"MSE:                {metrics.get('mse', 0):.4f}")
        lines.append(f"RMSE:               {metrics.get('rmse', 0):.4f}")
        lines.append(f"MAE:                {metrics.get('mae', 0):.4f}")
        lines.append(f"R2 Score:           {metrics.get('r2', 0):.4f}")
    lines.append("")
    
    if not is_regression:
        # Confusion Matrix
        lines.append("Confusion Matrix:")
        lines.append(f"  True Negatives:   {metrics.get('true_negatives', 'N/A')}")
        lines.append(f"  False Positives:  {metrics.get('false_positives', 'N/A')}")
        lines.append(f"  False Negatives:  {metrics.get('false_negatives', 'N/A')}")
        lines.append(f"  True Positives:   {metrics.get('true_positives', 'N/A')}")
        lines.append("")
        # Class Distribution
        lines.append("Class Distribution:")
        lines.append(f"  Ground Truth: {metrics.get('ground_truth_distribution', {})}")
        lines.append(f"  Predictions:  {metrics.get('prediction_distribution', {})}")
        lines.append("")
    
    # Trading Simulation Section
    lines.append("-" * 70)
    lines.append("TRADING SIMULATION RESULTS")
    lines.append("-" * 70)
    lines.append(f"Total Trades:       {trading_results.get('total_trades', 0)}")
    lines.append(f"Total PnL:          ₹{trading_results.get('total_pnl', 0):,.2f}")
    lines.append(f"Total Return:       {trading_results.get('total_return', 0):.2f}%")
    lines.append(f"Win Rate:           {trading_results.get('win_rate', 0):.2f}%")
    lines.append(f"Max Drawdown:       ₹{trading_results.get('max_drawdown', 0):,.2f}")
    lines.append(f"Max Drawdown %:     {trading_results.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"Avg Trade PnL:      ₹{trading_results.get('avg_trade_pnl', 0):,.2f}")
    lines.append(f"Avg Trade Return:   {trading_results.get('avg_trade_return', 0):.4f}%")
    lines.append(f"Sharpe Ratio:       {trading_results.get('sharpe_ratio', 0):.4f}")
    lines.append(f"Profitable Trades:  {trading_results.get('profitable_trades', 0)}")
    lines.append(f"Losing Trades:      {trading_results.get('losing_trades', 0)}")
    lines.append(f"Long Trades:        {trading_results.get('long_trades', 0)}")
    lines.append(f"Short Trades:       {trading_results.get('short_trades', 0)}")
    lines.append(f"Best Trade:         ₹{trading_results.get('best_trade', 0):,.2f}")
    lines.append(f"Worst Trade:        ₹{trading_results.get('worst_trade', 0):,.2f}")
    lines.append("")
    
    # Success Criteria Section
    lines.append("-" * 70)
    lines.append("SUCCESS CRITERIA CHECK")
    lines.append("-" * 70)
    
    if not is_regression:
        # Accuracy check
        accuracy = metrics.get('accuracy', 0)
        acc_pass = accuracy > 0.53
        lines.append(f"Test Accuracy > 53%:        {'PASS ✓' if acc_pass else 'FAIL ✗'} ({accuracy*100:.2f}%)")

        # Precision check
        precision = metrics.get('precision', 0)
        prec_pass = precision > 0.50
        lines.append(f"Precision > 50%:            {'PASS ✓' if prec_pass else 'FAIL ✗'} ({precision*100:.2f}%)")

        # Recall check
        recall = metrics.get('recall', 0)
        rec_pass = recall > 0.50
        lines.append(f"Recall > 50%:               {'PASS ✓' if rec_pass else 'FAIL ✗'} ({recall*100:.2f}%)")
    else:
        # Regression metrics: show RMSE/MAE and R2; no hard pass thresholds by default
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        lines.append(f"MSE:                {metrics.get('mse', 0):.4f}")
        lines.append(f"RMSE:               {rmse:.4f}")
        lines.append(f"MAE:                {mae:.4f}")
        lines.append(f"R2 Score:           {r2:.4f}")
    
    # Overfitting check
    if not is_regression:
        if train_acc is not None and val_acc is not None:
            gap = abs(train_acc - val_acc) * 100
            overfit_pass = gap < 5.0
            lines.append(f"Train-Val Gap < 5%:         {'PASS ✓' if overfit_pass else 'FAIL ✗'} ({gap:.2f}%)")
            lines.append(f"  (Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%)")
        else:
            lines.append("Train-Val Gap < 5%:         N/A (training history not found)")
    else:
        # Regression: compare RMSE from training history if available
        train_rmse = None
        val_rmse = None
        if training_history_path and os.path.exists(training_history_path):
            try:
                with open(training_history_path, 'r') as f:
                    history = json.load(f)
                if 'train_rmse' in history and history['train_rmse']:
                    train_rmse = history['train_rmse'][-1]
                if 'val_rmse' in history and history['val_rmse']:
                    val_rmse = history['val_rmse'][-1]
            except Exception:
                pass

        if train_rmse is not None and val_rmse is not None:
            # gap as relative percent difference
            gap = abs(train_rmse - val_rmse) / max(train_rmse, 1e-8) * 100
            overfit_pass = gap < 10.0
            lines.append(f"Train-Val RMSE Gap < 10%:    {'PASS ✓' if overfit_pass else 'FAIL ✗'} ({gap:.2f}%)")
            lines.append(f"  (Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f})")
        else:
            lines.append("Train-Val RMSE Gap < 10%:    N/A (training history not found)")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 70)
    
    if not is_regression:
        all_pass = acc_pass and prec_pass and rec_pass
        if all_pass:
            lines.append("Status: BASELINE ACHIEVED ✓")
            lines.append("The model meets minimum baseline requirements for Phase 1.")
        else:
            lines.append("Status: BASELINE NOT MET ✗")
            lines.append("The model does not meet all baseline requirements.")
            lines.append("Consider:")
            if not acc_pass:
                lines.append("  - Increasing training epochs or adjusting learning rate")
            if not prec_pass or not rec_pass:
                lines.append("  - Adjusting class weights or decision threshold")
    else:
        lines.append("Status: REGRESSION EVALUATION COMPLETED")
        lines.append("Note: Regression models do not have fixed 'baseline' thresholds in this report. Review MSE/RMSE/MAE/R2 metrics for model quality.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    report_text = '\n'.join(lines)
    
    # Save report with UTF-8 encoding to support special characters (₹)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to {save_path}")
    
    return report_text


def main():
    """Main entry point for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate BGRU baseline model on test data'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/bgru_baseline.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/test.csv',
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Sequence length (default: 60)'
    )
    parser.add_argument(
        '--training_history',
        type=str,
        default='models/checkpoints/training_history.json',
        help='Path to training history JSON'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation',
        help='Output directory for reports and plots'
    )
    parser.add_argument(
        '--transaction_cost',
        type=float,
        default=0.0003,
        help='Transaction cost per trade (default: 0.03%%)'
    )
    parser.add_argument(
        '--regression',
        action='store_true',
        help='Evaluate in regression mode (next day close predictions)'
    )
    
    args = parser.parse_args()
    
    # Setup output directories
    reports_dir = os.path.join(args.output_dir, 'reports')
    plots_dir = os.path.join(args.output_dir, 'plots')
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return 1
    
    if not os.path.exists(args.data):
        logger.error(f"Test data not found: {args.data}")
        return 1
    
    logger.info("=" * 60)
    logger.info("BGRU Baseline Evaluation")
    logger.info("=" * 60)
    
    # Load model and data
    predictor, test_df, y_true, y_pred, y_pred_proba = load_model_and_data(
        model_path=args.model,
        test_data_path=args.data,
        sequence_length=args.sequence_length,
        force_regression=args.regression if args.regression else None
    )
    
    logger.info("-" * 60)
    logger.info("Generating metrics and plots...")
    if predictor.regression:
        metrics = calculate_regression_metrics(y_true, y_pred)
        # Regression-specific plots
        plot_regression_predictions(
            y_true, y_pred, test_df,
            save_path=os.path.join(plots_dir, 'regression_plots.png'),
            sequence_length=args.sequence_length
        )
    else:
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(plots_dir, 'confusion_matrix.png')
        )
        # Plot prediction confidence
        plot_prediction_confidence(
            y_pred_proba,
            save_path=os.path.join(plots_dir, 'prediction_confidence.png'),
            y_true=y_true
        )
    
    # Run trading simulation
    logger.info("-" * 60)
    logger.info("Running trading simulation...")
    trading_results = simulate_trading(
        df=test_df,
        predictions=y_pred,
        regression=predictor.regression,
        sequence_length=args.sequence_length,
        transaction_cost=args.transaction_cost
    )
    
    # Generate text report
    logger.info("-" * 60)
    report_text = generate_report(
        metrics=metrics,
        trading_results=trading_results,
        training_history_path=args.training_history,
        save_path=os.path.join(reports_dir, 'baseline_report.txt'),
        is_regression=predictor.regression
    )
    
    # Save metrics as JSON
    all_results = {
        'regression_metrics' if predictor.regression else 'classification_metrics': metrics,
        'trading_results': trading_results,
        'evaluation_config': {
            'model_path': args.model,
            'data_path': args.data,
            'sequence_length': args.sequence_length,
            'transaction_cost': args.transaction_cost,
            'evaluated_at': datetime.now().isoformat()
        }
    }
    
    json_path = os.path.join(reports_dir, 'baseline_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Metrics JSON saved to {json_path}")
    
    # Print report summary
    print("\n" + report_text)
    
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
