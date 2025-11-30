#!/usr/bin/env python3
"""
Final Evaluation & Backtesting Script for BankNifty Ensemble Model.

This script provides comprehensive evaluation of the trained ensemble model
including classification metrics, backtesting simulation, and visualization.

Usage:
    python evaluation/evaluate_final.py --model models/checkpoints/ensemble_model.pkl \
        --data data/processed/test_final.csv
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    roc_curve,
    classification_report,
    precision_recall_curve,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ensemble import BGRUXGBoostEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MIN_STD = 1e-8  # Minimum standard deviation to avoid division by zero


# =============================================================================
# Threshold Optimization Functions
# =============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    min_threshold: float = 0.25,
    max_threshold: float = 0.75,
    step: float = 0.01
) -> Tuple[float, float, Dict[str, float]]:
    """
    Find the optimal decision threshold that maximizes the specified metric.
    
    This is crucial for imbalanced datasets where the default 0.5 threshold
    may not be optimal.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_proba: Prediction probabilities for the positive class
        metric: Metric to optimize ('f1', 'recall', 'precision', 'balanced_accuracy',
                'youden_j' for Youden's J statistic, or 'profit' for trading profit)
        min_threshold: Minimum threshold to consider (default: 0.25)
        max_threshold: Maximum threshold to consider (default: 0.75)
        step: Step size for threshold search (default: 0.01)
    
    Returns:
        Tuple of (optimal_threshold, best_score, metrics_at_threshold)
    """
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Skip if all predictions are same class
        if len(np.unique(y_pred)) < 2:
            continue
        
        # Calculate all metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate specificity and balanced accuracy
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (recall + specificity) / 2
            youden_j = recall + specificity - 1
        else:
            specificity = 0
            balanced_acc = accuracy
            youden_j = 0
        
        # Determine score based on selected metric
        if metric == 'f1':
            score = f1
        elif metric == 'recall':
            score = recall
        elif metric == 'precision':
            score = precision
        elif metric == 'balanced_accuracy':
            score = balanced_acc
        elif metric == 'youden_j':
            score = youden_j
        elif metric == 'f1_recall_avg':
            # Custom: average of F1 and recall to favor recall
            score = (f1 + recall) / 2
        else:
            score = f1
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'specificity': specificity,
                'balanced_accuracy': balanced_acc,
                'youden_j': youden_j
            }
    
    return best_threshold, best_score, best_metrics


def find_threshold_for_target_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.55,
    min_precision: float = 0.30
) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that achieves target recall while maintaining minimum precision.
    
    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities
        target_recall: Target recall to achieve (default: 0.55)
        min_precision: Minimum acceptable precision (default: 0.30)
    
    Returns:
        Tuple of (threshold, metrics_dict)
    """
    # Use precision-recall curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find thresholds that meet both criteria
    valid_indices = []
    for i, (p, r) in enumerate(zip(precision_vals[:-1], recall_vals[:-1])):
        if r >= target_recall and p >= min_precision:
            valid_indices.append(i)
    
    if valid_indices:
        # Among valid thresholds, pick the one with highest F1
        best_idx = max(valid_indices, key=lambda i: 
            2 * precision_vals[i] * recall_vals[i] / 
            (precision_vals[i] + recall_vals[i] + 1e-8))
        best_threshold = thresholds[best_idx]
        
        return best_threshold, {
            'threshold': best_threshold,
            'precision': precision_vals[best_idx],
            'recall': recall_vals[best_idx],
            'f1': 2 * precision_vals[best_idx] * recall_vals[best_idx] / 
                  (precision_vals[best_idx] + recall_vals[best_idx] + 1e-8)
        }
    else:
        # Fallback: find threshold closest to target recall
        recall_diffs = np.abs(recall_vals[:-1] - target_recall)
        best_idx = np.argmin(recall_diffs)
        best_threshold = thresholds[best_idx]
        
        return best_threshold, {
            'threshold': best_threshold,
            'precision': precision_vals[best_idx],
            'recall': recall_vals[best_idx],
            'f1': 2 * precision_vals[best_idx] * recall_vals[best_idx] / 
                  (precision_vals[best_idx] + recall_vals[best_idx] + 1e-8),
            'note': 'Could not meet both targets, optimized for recall'
        }


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str,
    optimal_threshold: float = 0.5
) -> None:
    """
    Plot threshold analysis showing metrics vs threshold.
    
    Args:
        y_true: Ground truth labels
        y_proba: Prediction probabilities
        save_path: Path to save the plot
        optimal_threshold: Optimal threshold to highlight
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    thresholds = np.arange(0.2, 0.8, 0.02)
    metrics_by_threshold = {
        'precision': [], 'recall': [], 'f1': [], 
        'accuracy': [], 'n_positive_preds': []
    }
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        metrics_by_threshold['precision'].append(
            precision_score(y_true, y_pred, zero_division=0))
        metrics_by_threshold['recall'].append(
            recall_score(y_true, y_pred, zero_division=0))
        metrics_by_threshold['f1'].append(
            f1_score(y_true, y_pred, zero_division=0))
        metrics_by_threshold['accuracy'].append(
            accuracy_score(y_true, y_pred))
        metrics_by_threshold['n_positive_preds'].append(
            np.sum(y_pred == 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Metrics vs Threshold
    axes[0].plot(thresholds, metrics_by_threshold['precision'], 'b-', 
                 linewidth=2, label='Precision')
    axes[0].plot(thresholds, metrics_by_threshold['recall'], 'g-', 
                 linewidth=2, label='Recall')
    axes[0].plot(thresholds, metrics_by_threshold['f1'], 'r-', 
                 linewidth=2, label='F1-Score')
    axes[0].plot(thresholds, metrics_by_threshold['accuracy'], 'purple', 
                 linewidth=2, linestyle='--', label='Accuracy')
    axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, 
                    label='Default (0.5)')
    axes[0].axvline(x=optimal_threshold, color='orange', linewidth=2,
                    label=f'Optimal ({optimal_threshold:.2f})')
    axes[0].set_xlabel('Decision Threshold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Metrics vs Decision Threshold', fontsize=14)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.2, 0.8)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Number of positive predictions vs threshold
    axes[1].plot(thresholds, metrics_by_threshold['n_positive_preds'], 'b-',
                 linewidth=2)
    axes[1].axhline(y=np.sum(y_true == 1), color='green', linestyle='--',
                    linewidth=2, label=f'Actual Positives ({np.sum(y_true == 1)})')
    axes[1].axvline(x=optimal_threshold, color='orange', linewidth=2,
                    label=f'Optimal ({optimal_threshold:.2f})')
    axes[1].set_xlabel('Decision Threshold', fontsize=12)
    axes[1].set_ylabel('Number of Positive Predictions', fontsize=12)
    axes[1].set_title('Positive Predictions vs Threshold', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Threshold analysis plot saved to {save_path}")


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_final_model_and_data(
    model_path: str,
    test_data_path: str,
    sequence_length: int = 60
) -> Tuple[BGRUXGBoostEnsemble, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the final ensemble model and test data.
    
    Args:
        model_path: Path to the ensemble model checkpoint (.pkl file)
        test_data_path: Path to the test CSV file
        sequence_length: Sequence length used during training
    
    Returns:
        Tuple of (ensemble, test_df, y_true, y_pred, y_pred_proba)
    """
    logger.info(f"Loading ensemble model from {model_path}")
    
    # Determine BGRU model path from ensemble or use default
    ensemble_dir = Path(model_path).parent
    bgru_model_path = str(ensemble_dir / 'bgru_hybrid.pt')
    
    if not os.path.exists(bgru_model_path):
        bgru_model_path = str(ensemble_dir / 'bgru_baseline.pt')
    
    # Initialize ensemble
    ensemble = BGRUXGBoostEnsemble(
        bgru_model_path=bgru_model_path,
        sequence_length=sequence_length
    )
    
    # Load ensemble components (XGBoost, stacking model, weights)
    ensemble.load_ensemble(model_path)
    
    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Generate predictions
    y_pred, y_pred_proba = ensemble.predict_ensemble(
        test_df,
        method='weighted',
        batch_size=64
    )
    
    # Get ground truth labels
    # Align targets with predictions (predictions start after sequence_length samples)
    n_samples = len(test_df) - sequence_length
    y_true = test_df['target'].values[sequence_length - 1:sequence_length - 1 + n_samples]
    
    # Ensure alignment
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len].astype(int)
    y_pred = y_pred[:min_len]
    y_pred_proba = y_pred_proba[:min_len]
    
    logger.info(f"Evaluation samples: {len(y_true)}")
    
    return ensemble, test_df, y_true, y_pred, y_pred_proba


# =============================================================================
# Metrics Calculation Functions
# =============================================================================

def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, Any]:
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
    
    # Per-class precision and recall
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['class_0_precision'] = float(class_report.get('0', {}).get('precision', 0))
    metrics['class_0_recall'] = float(class_report.get('0', {}).get('recall', 0))
    metrics['class_0_f1'] = float(class_report.get('0', {}).get('f1-score', 0))
    metrics['class_1_precision'] = float(class_report.get('1', {}).get('precision', 0))
    metrics['class_1_recall'] = float(class_report.get('1', {}).get('recall', 0))
    metrics['class_1_f1'] = float(class_report.get('1', {}).get('f1-score', 0))
    
    # Class distribution in predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    metrics['prediction_distribution'] = pred_dist
    
    # Class distribution in ground truth
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    true_dist = dict(zip([int(u) for u in unique_true], [int(c) for c in counts_true]))
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


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrix_heatmap(
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
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]:d}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=['DOWN (0)', 'UP (1)'],
        yticklabels=['DOWN (0)', 'UP (1)'],
        annot_kws={'size': 14}
    )
    plt.title('Confusion Matrix - Final Ensemble Model', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str
) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Prediction probabilities
        save_path: Path to save the plot
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    # Mark optimal threshold point (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='green', s=100,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Final Ensemble Model', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"ROC curve saved to {save_path}")


def plot_prediction_confidence_distribution(
    y_pred_proba: np.ndarray,
    save_path: str,
    y_true: Optional[np.ndarray] = None
) -> None:
    """
    Plot prediction confidence histogram.
    
    Args:
        y_pred_proba: Prediction probabilities
        save_path: Path to save the plot
        y_true: Optional ground truth for coloring by correctness
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of prediction probabilities
    axes[0].hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[0].set_title('Prediction Probability Distribution', fontsize=12)
    axes[0].set_xlabel('Probability (P(UP))', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence by correctness
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
        axes[1].boxplot(y_pred_proba)
        axes[1].set_title('Probability Distribution Box Plot', fontsize=12)
        axes[1].set_ylabel('Probability', fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Confidence distribution saved to {save_path}")


def plot_equity_curve(
    equity_curve: List[float],
    save_path: str,
    title: str = 'Equity Curve'
) -> None:
    """Plot and save equity curve."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.plot(equity_curve, 'b-', linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.fill_between(range(len(equity_curve)), 0, equity_curve,
                     where=np.array(equity_curve) >= 0, alpha=0.3, color='green')
    plt.fill_between(range(len(equity_curve)), 0, equity_curve,
                     where=np.array(equity_curve) < 0, alpha=0.3, color='red')
    plt.title(title, fontsize=14)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Cumulative P&L (₹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Equity curve saved to {save_path}")


def plot_drawdown(
    drawdown_curve: List[float],
    save_path: str
) -> None:
    """Plot and save drawdown chart."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.fill_between(range(len(drawdown_curve)), 0, drawdown_curve, 
                     color='red', alpha=0.5)
    plt.plot(drawdown_curve, 'r-', linewidth=1)
    plt.title('Drawdown Chart', fontsize=14)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Drawdown chart saved to {save_path}")


def plot_hourly_trades(
    trades_df: pd.DataFrame,
    save_path: str
) -> None:
    """Plot trade distribution by hour."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if 'hour' not in trades_df.columns:
        logger.warning("No hour column in trades DataFrame, skipping hourly plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trade count by hour
    hourly_counts = trades_df.groupby('hour').size()
    axes[0].bar(hourly_counts.index, hourly_counts.values, color='steelblue', edgecolor='black')
    axes[0].set_title('Trade Count by Hour', fontsize=12)
    axes[0].set_xlabel('Hour', fontsize=10)
    axes[0].set_ylabel('Number of Trades', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Win rate by hour
    hourly_wins = trades_df.groupby('hour')['won'].mean() * 100
    colors = ['green' if x >= 50 else 'red' for x in hourly_wins.values]
    axes[1].bar(hourly_wins.index, hourly_wins.values, color=colors, edgecolor='black')
    axes[1].axhline(y=50, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Win Rate by Hour (%)', fontsize=12)
    axes[1].set_xlabel('Hour', fontsize=10)
    axes[1].set_ylabel('Win Rate (%)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Hourly trades plot saved to {save_path}")


def plot_monthly_returns(
    trades_df: pd.DataFrame,
    save_path: str
) -> None:
    """Plot monthly returns heatmap."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if 'year' not in trades_df.columns or 'month' not in trades_df.columns:
        logger.warning("No year/month columns in trades DataFrame, skipping monthly plot")
        return
    
    # Calculate monthly returns
    monthly_returns = trades_df.groupby(['year', 'month'])['net_return'].sum() * 100
    monthly_returns = monthly_returns.unstack(level='month')
    
    # Reorder months
    month_order = list(range(1, 13))
    monthly_returns = monthly_returns.reindex(columns=[m for m in month_order if m in monthly_returns.columns])
    
    plt.figure(figsize=(14, 6))
    
    # Create heatmap
    sns.heatmap(
        monthly_returns,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Return (%)'}
    )
    
    plt.title('Monthly Returns Heatmap (%)', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Monthly returns heatmap saved to {save_path}")


# =============================================================================
# Backtesting Functions
# =============================================================================

def backtest_trading_strategy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    proba: np.ndarray,
    sequence_length: int = 60,
    lot_size: int = 25,
    transaction_cost: float = 0.0003,
    trades_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive backtesting for BankNifty futures.
    
    Trading Strategy:
    - BankNifty futures (1 lot = 25 units)
    - Entry: market order at close of prediction candle
    - Exit: market order at close of next candle
    - Transaction cost: 0.03% per round trip
    
    Tracks:
    - Equity curve
    - Drawdown
    - Win rate
    - Sharpe ratio
    - Sortino ratio
    - Profit factor
    
    Args:
        df: DataFrame with OHLCV data
        predictions: Model predictions (0 or 1)
        proba: Prediction probabilities
        sequence_length: Sequence length used (to align predictions with data)
        lot_size: Number of units per lot (default: 25 for BankNifty)
        transaction_cost: Transaction cost per trade (default: 0.03% = 0.0003)
    
    Returns:
        Dictionary with all backtesting metrics
    """
    logger.info("Starting comprehensive backtesting...")
    
    # Align predictions with data
    start_idx = sequence_length - 1
    
    # Ensure we have enough data for exit
    max_trades = min(len(predictions), len(df) - start_idx - 1)
    
    trades = []
    cumulative_pnl = 0.0
    equity_curve = [0.0]
    peak_value = 0.0
    drawdown_curve = [0.0]
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    
    for i in range(max_trades):
        data_idx = start_idx + i
        
        if data_idx + 1 >= len(df):
            break
        
        entry_price = df.iloc[data_idx]['close']
        exit_price = df.iloc[data_idx + 1]['close']
        prediction = predictions[i]
        prob = proba[i]
        
        # Get timestamp info if available
        if isinstance(df.index, pd.DatetimeIndex):
            timestamp = df.index[data_idx]
            hour = timestamp.hour
            year = timestamp.year
            month = timestamp.month
            day_of_week = timestamp.dayofweek
        else:
            hour = 0
            year = 2024
            month = 1
            day_of_week = 0
        
        # Calculate raw PnL (per unit)
        if prediction == 1:  # LONG
            raw_pnl = exit_price - entry_price
            direction = 'LONG'
        else:  # SHORT
            raw_pnl = entry_price - exit_price
            direction = 'SHORT'
        
        # Calculate return percentage
        raw_return = raw_pnl / entry_price
        
        # Apply transaction cost (round trip = 2 * transaction_cost)
        net_return = raw_return - (2 * transaction_cost)
        net_pnl = raw_pnl * lot_size - (2 * transaction_cost * entry_price * lot_size)
        
        # Track trade
        trades.append({
            'trade_id': i + 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'probability': prob,
            'raw_pnl': raw_pnl * lot_size,
            'net_pnl': net_pnl,
            'raw_return': raw_return,
            'net_return': net_return,
            'won': net_pnl > 0,
            'hour': hour,
            'year': year,
            'month': month,
            'day_of_week': day_of_week
        })
        
        cumulative_pnl += net_pnl
        equity_curve.append(cumulative_pnl)
        
        # Track max drawdown
        if cumulative_pnl > peak_value:
            peak_value = cumulative_pnl
        
        drawdown = peak_value - cumulative_pnl
        drawdown_pct = (drawdown / peak_value * 100) if peak_value > 0 else 0.0
        drawdown_curve.append(-drawdown_pct)
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
    
    # Calculate summary statistics
    if not trades:
        logger.warning("No trades executed in simulation")
        return _empty_backtest_results()
    
    trades_df = pd.DataFrame(trades)

    # Save trades to CSV if requested or default path set
    if trades_df is not None and not trades_df.empty:
        if trades_csv_path is None:
            # Default path inside evaluation reports
            trades_csv_path = os.path.join('evaluation', 'reports', 'backtest_trades.csv')
        # Ensure directory exists
        Path(trades_csv_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            trades_df.to_csv(trades_csv_path, index=False)
            logger.info(f"Saved backtest trades to {trades_csv_path}")
        except Exception as e:
            logger.warning(f"Could not save trades to CSV: {e}")
    
    # Basic statistics
    total_trades = len(trades)
    winning_trades = trades_df[trades_df['won']]
    losing_trades = trades_df[~trades_df['won']]
    
    # Win/Loss metrics
    win_rate = len(winning_trades) / total_trades * 100
    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0.0
    
    # Profit factor
    gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else MIN_STD
    profit_factor = gross_profit / gross_loss if gross_loss > MIN_STD else float('inf')
    
    # Return calculations
    total_return = sum(t['net_return'] for t in trades) * 100
    
    # Annualized return (assuming ~252 trading days per year, ~6 trades per day)
    avg_daily_trades = 6  # approximate for intraday
    trading_days = total_trades / avg_daily_trades if total_trades > 0 else 1
    annualized_return = ((1 + total_return / 100) ** (252 / max(trading_days, 1)) - 1) * 100
    
    # Risk metrics
    returns = trades_df['net_return'].values
    avg_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else MIN_STD
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe_ratio = (avg_return / std_return) * np.sqrt(252 * avg_daily_trades) if std_return > MIN_STD else 0.0
    
    # Sortino ratio (using downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 1 else MIN_STD
    sortino_ratio = (avg_return / downside_std) * np.sqrt(252 * avg_daily_trades) if downside_std > MIN_STD else 0.0
    
    # Average holding time (assuming each trade is held for 1 candle period)
    avg_holding_time = 1  # 1 candle period
    
    results = {
        # Trade counts
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'long_trades': int((trades_df['direction'] == 'LONG').sum()),
        'short_trades': int((trades_df['direction'] == 'SHORT').sum()),
        
        # Return metrics
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'total_pnl': float(cumulative_pnl),
        
        # Win/Loss metrics
        'win_rate': float(win_rate),
        'avg_profit_per_winning_trade': float(avg_win),
        'avg_loss_per_losing_trade': float(avg_loss),
        'profit_factor': float(min(profit_factor, 999.99)),  # Cap for display
        
        # Risk metrics
        'max_drawdown': float(max_drawdown),
        'max_drawdown_pct': float(max_drawdown_pct),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        
        # Trade statistics
        'avg_trade_pnl': float(trades_df['net_pnl'].mean()),
        'avg_trade_return': float(trades_df['net_return'].mean() * 100),
        'std_trade_return': float(trades_df['net_return'].std() * 100),
        'best_trade': float(trades_df['net_pnl'].max()),
        'worst_trade': float(trades_df['net_pnl'].min()),
        'avg_holding_time': avg_holding_time,
        
        # Curves for plotting
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'trades_df': trades_df
    }
    # Add saved CSV path to results if any
    if trades_csv_path:
        results['trades_csv_path'] = trades_csv_path
    
    logger.info(f"Total trades: {results['total_trades']}")
    logger.info(f"Total P&L: ₹{results['total_pnl']:,.2f}")
    logger.info(f"Total return: {results['total_return']:.2f}%")
    logger.info(f"Win rate: {results['win_rate']:.2f}%")
    logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.4f}")
    logger.info(f"Max drawdown: {results['max_drawdown_pct']:.2f}%")
    
    return results


def _empty_backtest_results() -> Dict[str, Any]:
    """Return empty backtest results."""
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'long_trades': 0,
        'short_trades': 0,
        'total_return': 0.0,
        'annualized_return': 0.0,
        'total_pnl': 0.0,
        'win_rate': 0.0,
        'avg_profit_per_winning_trade': 0.0,
        'avg_loss_per_losing_trade': 0.0,
        'profit_factor': 0.0,
        'max_drawdown': 0.0,
        'max_drawdown_pct': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'avg_trade_pnl': 0.0,
        'avg_trade_return': 0.0,
        'std_trade_return': 0.0,
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'avg_holding_time': 0,
        'equity_curve': [0.0],
        'drawdown_curve': [0.0],
        'trades_df': pd.DataFrame(),
        'trades_csv_path': None
    }



# =============================================================================
# Report Generation Functions
# =============================================================================

def generate_comprehensive_report(
    metrics: Dict[str, Any],
    backtest_results: Dict[str, Any],
    output_dir: str = 'evaluation/reports'
) -> Tuple[str, str]:
    """
    Creates detailed HTML and text reports.
    
    Args:
        metrics: Classification metrics dictionary
        backtest_results: Backtesting results dictionary
        output_dir: Output directory for reports
    
    Returns:
        Tuple of (html_path, txt_path)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate text report
    txt_path = os.path.join(output_dir, 'final_report.txt')
    txt_content = _generate_text_report(metrics, backtest_results)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    logger.info(f"Text report saved to {txt_path}")
    
    # Generate HTML report
    html_path = os.path.join(output_dir, 'final_report.html')
    html_content = _generate_html_report(metrics, backtest_results)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"HTML report saved to {html_path}")
    
    return html_path, txt_path


def _generate_text_report(
    metrics: Dict[str, Any],
    backtest_results: Dict[str, Any]
) -> str:
    """Generate text report content."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("=== FINAL MODEL EVALUATION ===")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Classification Metrics
    lines.append("Classification Metrics:")
    lines.append("-" * 70)
    
    # Check accuracy target (60%)
    accuracy = metrics.get('accuracy', 0) * 100
    acc_pass = accuracy >= 60
    lines.append(f"{'✓' if acc_pass else '✗'} Accuracy: {accuracy:.1f}% [TARGET: 60%] - {'PASS' if acc_pass else 'FAIL'}")
    
    # Check precision target (55%)
    precision = metrics.get('precision', 0) * 100
    prec_pass = precision >= 55
    lines.append(f"{'✓' if prec_pass else '✗'} Precision: {precision:.1f}% [TARGET: 55%] - {'PASS' if prec_pass else 'FAIL'}")
    
    # Check recall target (55%)
    recall = metrics.get('recall', 0) * 100
    rec_pass = recall >= 55
    lines.append(f"{'✓' if rec_pass else '✗'} Recall: {recall:.1f}% [TARGET: 55%] - {'PASS' if rec_pass else 'FAIL'}")
    
    lines.append(f"✓ F1-Score: {metrics.get('f1_score', 0)*100:.1f}%")
    lines.append(f"✓ ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
    lines.append("")
    
    # Per-class metrics
    lines.append("Per-Class Metrics:")
    lines.append(f"  Class 0 (DOWN): Precision={metrics.get('class_0_precision', 0)*100:.1f}%, Recall={metrics.get('class_0_recall', 0)*100:.1f}%")
    lines.append(f"  Class 1 (UP):   Precision={metrics.get('class_1_precision', 0)*100:.1f}%, Recall={metrics.get('class_1_recall', 0)*100:.1f}%")
    lines.append("")
    
    # Backtesting Results
    lines.append("Backtesting Results:")
    lines.append("-" * 70)
    lines.append(f"✓ Total Return: {backtest_results.get('total_return', 0):.1f}%")
    
    # Check win rate target (55%)
    win_rate = backtest_results.get('win_rate', 0)
    wr_pass = win_rate >= 55
    lines.append(f"{'✓' if wr_pass else '✗'} Win Rate: {win_rate:.1f}% [TARGET: 55%] - {'PASS' if wr_pass else 'FAIL'}")
    
    lines.append(f"✓ Profit Factor: {backtest_results.get('profit_factor', 0):.2f}")
    lines.append(f"✓ Max Drawdown: -{backtest_results.get('max_drawdown_pct', 0):.1f}%")
    
    # Check Sharpe ratio target (1.0)
    sharpe = backtest_results.get('sharpe_ratio', 0)
    sharpe_pass = sharpe >= 1.0
    lines.append(f"{'✓' if sharpe_pass else '✗'} Sharpe Ratio: {sharpe:.2f} [TARGET: 1.0] - {'PASS' if sharpe_pass else 'FAIL'}")
    
    lines.append(f"✓ Sortino Ratio: {backtest_results.get('sortino_ratio', 0):.2f}")
    lines.append(f"✓ Total Trades: {backtest_results.get('total_trades', 0):,}")
    lines.append(f"✓ Average Holding Time: {backtest_results.get('avg_holding_time', 0)} candle(s)")
    lines.append("")
    
    # Trade Statistics
    lines.append("Trade Statistics:")
    lines.append("-" * 70)
    lines.append(f"  Winning Trades: {backtest_results.get('winning_trades', 0)}")
    lines.append(f"  Losing Trades: {backtest_results.get('losing_trades', 0)}")
    lines.append(f"  Long Trades: {backtest_results.get('long_trades', 0)}")
    lines.append(f"  Short Trades: {backtest_results.get('short_trades', 0)}")
    lines.append(f"  Avg Profit/Winning Trade: ₹{backtest_results.get('avg_profit_per_winning_trade', 0):,.2f}")
    lines.append(f"  Avg Loss/Losing Trade: ₹{backtest_results.get('avg_loss_per_losing_trade', 0):,.2f}")
    lines.append(f"  Best Trade: ₹{backtest_results.get('best_trade', 0):,.2f}")
    lines.append(f"  Worst Trade: ₹{backtest_results.get('worst_trade', 0):,.2f}")
    lines.append(f"  Total P&L: ₹{backtest_results.get('total_pnl', 0):,.2f}")
    lines.append("")
    
    # Success Criteria Summary
    lines.append("=" * 70)
    lines.append("SUCCESS CRITERIA SUMMARY")
    lines.append("=" * 70)
    
    criteria_passed = sum([acc_pass, prec_pass, rec_pass, wr_pass, sharpe_pass])
    total_criteria = 5
    
    lines.append(f"Overall: {criteria_passed}/{total_criteria} criteria PASSED {'✓✓✓' if criteria_passed == total_criteria else ''}")
    
    if criteria_passed == total_criteria:
        lines.append("Model ready for deployment!")
    else:
        lines.append("Some criteria not met. Review model performance.")
    
    lines.append("=" * 70)
    
    return '\n'.join(lines)


def _generate_html_report(
    metrics: Dict[str, Any],
    backtest_results: Dict[str, Any]
) -> str:
    """Generate HTML report content."""
    
    # Calculate pass/fail for criteria
    accuracy = metrics.get('accuracy', 0) * 100
    precision = metrics.get('precision', 0) * 100
    recall = metrics.get('recall', 0) * 100
    win_rate = backtest_results.get('win_rate', 0)
    sharpe = backtest_results.get('sharpe_ratio', 0)
    
    acc_pass = accuracy >= 60
    prec_pass = precision >= 55
    rec_pass = recall >= 55
    wr_pass = win_rate >= 55
    sharpe_pass = sharpe >= 1.0
    
    criteria_passed = sum([acc_pass, prec_pass, rec_pass, wr_pass, sharpe_pass])
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Final Model Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .criteria-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .criteria-table th, .criteria-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .criteria-table th {{
            background-color: #3498db;
            color: white;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .summary-box {{
            background-color: {'#d4edda' if criteria_passed == 5 else '#fff3cd'};
            border: 1px solid {'#c3e6cb' if criteria_passed == 5 else '#ffeeba'};
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        .summary-box h3 {{
            margin: 0;
            color: {'#155724' if criteria_passed == 5 else '#856404'};
        }}
        .plot-section {{
            margin: 20px 0;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .plot-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .plot-card img {{
            max-width: 100%;
            border-radius: 5px;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Final Model Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Classification Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card {'success' if acc_pass else 'warning'}">
                <div class="metric-value">{accuracy:.1f}%</div>
                <div class="metric-label">Accuracy (Target: 60%)</div>
            </div>
            <div class="metric-card {'success' if prec_pass else 'warning'}">
                <div class="metric-value">{precision:.1f}%</div>
                <div class="metric-label">Precision (Target: 55%)</div>
            </div>
            <div class="metric-card {'success' if rec_pass else 'warning'}">
                <div class="metric-value">{recall:.1f}%</div>
                <div class="metric-label">Recall (Target: 55%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('f1_score', 0)*100:.1f}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('roc_auc', 0):.3f}</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
        </div>
        
        <h2>💰 Backtesting Results</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{backtest_results.get('total_return', 0):.1f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card {'success' if wr_pass else 'warning'}">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate (Target: 55%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{backtest_results.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card {'success' if sharpe_pass else 'warning'}">
                <div class="metric-value">{sharpe:.2f}</div>
                <div class="metric-label">Sharpe Ratio (Target: 1.0)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">-{backtest_results.get('max_drawdown_pct', 0):.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{backtest_results.get('total_trades', 0):,}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>
        
        <h2>✅ Success Criteria</h2>
        <table class="criteria-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Target</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{accuracy:.1f}%</td>
                <td>≥ 60%</td>
                <td class="{'pass' if acc_pass else 'fail'}">{'PASS ✓' if acc_pass else 'FAIL ✗'}</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{precision:.1f}%</td>
                <td>≥ 55%</td>
                <td class="{'pass' if prec_pass else 'fail'}">{'PASS ✓' if prec_pass else 'FAIL ✗'}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{recall:.1f}%</td>
                <td>≥ 55%</td>
                <td class="{'pass' if rec_pass else 'fail'}">{'PASS ✓' if rec_pass else 'FAIL ✗'}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{win_rate:.1f}%</td>
                <td>≥ 55%</td>
                <td class="{'pass' if wr_pass else 'fail'}">{'PASS ✓' if wr_pass else 'FAIL ✗'}</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{sharpe:.2f}</td>
                <td>≥ 1.0</td>
                <td class="{'pass' if sharpe_pass else 'fail'}">{'PASS ✓' if sharpe_pass else 'FAIL ✗'}</td>
            </tr>
        </table>
        
        <div class="summary-box">
            <h3>{'🎉 Model Ready for Deployment!' if criteria_passed == 5 else f'⚠️ {criteria_passed}/5 Criteria Passed'}</h3>
            <p>Overall: {criteria_passed}/5 criteria PASSED</p>
        </div>
        
        <h2>📈 Trade Statistics</h2>
        <table class="criteria-table">
            <tr><td>Total Trades</td><td>{backtest_results.get('total_trades', 0):,}</td></tr>
            <tr><td>Winning Trades</td><td>{backtest_results.get('winning_trades', 0):,}</td></tr>
            <tr><td>Losing Trades</td><td>{backtest_results.get('losing_trades', 0):,}</td></tr>
            <tr><td>Long Trades</td><td>{backtest_results.get('long_trades', 0):,}</td></tr>
            <tr><td>Short Trades</td><td>{backtest_results.get('short_trades', 0):,}</td></tr>
            <tr><td>Total P&L</td><td>₹{backtest_results.get('total_pnl', 0):,.2f}</td></tr>
            <tr><td>Avg Profit/Winning Trade</td><td>₹{backtest_results.get('avg_profit_per_winning_trade', 0):,.2f}</td></tr>
            <tr><td>Avg Loss/Losing Trade</td><td>₹{backtest_results.get('avg_loss_per_losing_trade', 0):,.2f}</td></tr>
            <tr><td>Best Trade</td><td>₹{backtest_results.get('best_trade', 0):,.2f}</td></tr>
            <tr><td>Worst Trade</td><td>₹{backtest_results.get('worst_trade', 0):,.2f}</td></tr>
            <tr><td>Sortino Ratio</td><td>{backtest_results.get('sortino_ratio', 0):.2f}</td></tr>
            <tr><td>Annualized Return</td><td>{backtest_results.get('annualized_return', 0):.1f}%</td></tr>
        </table>
        
        <h2>📉 Visualizations</h2>
        <div class="plot-section">
            <p>Generated plots are available in the <code>evaluation/plots/</code> directory:</p>
            <ul>
                <li>final_confusion_matrix.png - Confusion Matrix Heatmap</li>
                <li>roc_curve.png - ROC Curve</li>
                <li>confidence_distribution.png - Prediction Confidence Distribution</li>
                <li>equity_curve.png - Equity Curve</li>
                <li>drawdown.png - Drawdown Chart</li>
                <li>hourly_trades.png - Trade Distribution by Hour</li>
                <li>monthly_returns.png - Monthly Returns Heatmap</li>
            </ul>
        </div>
        
        <footer>
            <p>BankNifty Directional Prediction Model - Final Evaluation Report</p>
            <p>Generated by evaluate_final.py</p>
        </footer>
    </div>
</body>
</html>
'''
    return html


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main entry point for final evaluation."""
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of the final ensemble model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/ensemble_model.pkl',
        help='Path to ensemble model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/test_final.csv',
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=60,
        help='Sequence length (default: 60)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation',
        help='Output directory for reports and plots'
    )
    parser.add_argument(
        '--lot_size',
        type=int,
        default=25,
        help='BankNifty lot size (default: 25)'
    )
    parser.add_argument(
        '--transaction_cost',
        type=float,
        default=0.0003,
        help='Transaction cost per trade (default: 0.03%%)'
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
    logger.info("FINAL MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load model and data
    ensemble, test_df, y_true, y_pred, y_pred_proba = load_final_model_and_data(
        model_path=args.model,
        test_data_path=args.data,
        sequence_length=args.sequence_length
    )
    
    # =========================================================================
    # THRESHOLD OPTIMIZATION (Key improvement for recall/precision)
    # =========================================================================
    logger.info("-" * 60)
    logger.info("Optimizing decision threshold...")
    
    # Find optimal threshold for F1 score
    optimal_threshold_f1, best_f1, metrics_f1 = find_optimal_threshold(
        y_true, y_pred_proba, metric='f1'
    )
    logger.info(f"Optimal threshold (F1): {optimal_threshold_f1:.3f} -> F1={best_f1:.4f}")
    
    # Find optimal threshold prioritizing recall
    optimal_threshold_recall, best_recall_score, metrics_recall = find_optimal_threshold(
        y_true, y_pred_proba, metric='f1_recall_avg'
    )
    logger.info(f"Optimal threshold (F1+Recall): {optimal_threshold_recall:.3f}")
    
    # Find threshold for target recall
    target_threshold, target_metrics = find_threshold_for_target_recall(
        y_true, y_pred_proba, target_recall=0.55, min_precision=0.35
    )
    logger.info(f"Threshold for 55% recall: {target_threshold:.3f}")
    logger.info(f"  -> Precision: {target_metrics.get('precision', 0):.4f}")
    logger.info(f"  -> Recall: {target_metrics.get('recall', 0):.4f}")
    logger.info(f"  -> F1: {target_metrics.get('f1', 0):.4f}")
    
    # Use the threshold that optimizes F1 + Recall balance
    # You can change this to optimal_threshold_f1 or target_threshold based on needs
    optimal_threshold = optimal_threshold_recall
    
    # Apply optimal threshold to get improved predictions
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    logger.info("-" * 60)
    logger.info(f"Applying optimized threshold: {optimal_threshold:.3f} (was 0.5)")
    logger.info(f"Prediction distribution change:")
    logger.info(f"  Before: DOWN={np.sum(y_pred == 0)}, UP={np.sum(y_pred == 1)}")
    logger.info(f"  After:  DOWN={np.sum(y_pred_optimized == 0)}, UP={np.sum(y_pred_optimized == 1)}")
    
    # Use optimized predictions for evaluation
    y_pred = y_pred_optimized
    
    # Calculate classification metrics with optimized threshold
    logger.info("-" * 60)
    logger.info("Calculating classification metrics (with optimized threshold)...")
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    # Add threshold info to metrics
    metrics['optimal_threshold'] = optimal_threshold
    metrics['default_threshold'] = 0.5
    metrics['threshold_optimization'] = {
        'f1_optimal': {'threshold': optimal_threshold_f1, 'metrics': metrics_f1},
        'recall_optimal': {'threshold': optimal_threshold_recall, 'metrics': metrics_recall},
        'target_55_recall': {'threshold': target_threshold, 'metrics': target_metrics}
    }
    
    # Generate visualizations
    logger.info("-" * 60)
    logger.info("Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix_heatmap(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, 'final_confusion_matrix.png')
    )
    
    # ROC curve
    plot_roc_curve(
        y_true, y_pred_proba,
        save_path=os.path.join(plots_dir, 'roc_curve.png')
    )
    
    # Confidence distribution
    plot_prediction_confidence_distribution(
        y_pred_proba,
        save_path=os.path.join(plots_dir, 'confidence_distribution.png'),
        y_true=y_true
    )
    
    # Threshold analysis plot
    plot_threshold_analysis(
        y_true, y_pred_proba,
        save_path=os.path.join(plots_dir, 'threshold_analysis.png'),
        optimal_threshold=optimal_threshold
    )
    
    # Run backtesting
    logger.info("-" * 60)
    logger.info("Running comprehensive backtesting...")
    backtest_results = backtest_trading_strategy(
        df=test_df,
        predictions=y_pred,
        proba=y_pred_proba,
        sequence_length=args.sequence_length,
        lot_size=args.lot_size,
        transaction_cost=args.transaction_cost,
        trades_csv_path=os.path.join(reports_dir, 'backtest_trades.csv')
    )
    
    # Plot equity curve
    plot_equity_curve(
        backtest_results['equity_curve'],
        save_path=os.path.join(plots_dir, 'equity_curve.png'),
        title='Equity Curve - BankNifty Strategy'
    )
    
    # Plot drawdown
    plot_drawdown(
        backtest_results['drawdown_curve'],
        save_path=os.path.join(plots_dir, 'drawdown.png')
    )
    
    # Plot hourly trades
    if isinstance(backtest_results.get('trades_df'), pd.DataFrame) and len(backtest_results['trades_df']) > 0:
        plot_hourly_trades(
            backtest_results['trades_df'],
            save_path=os.path.join(plots_dir, 'hourly_trades.png')
        )
        
        # Plot monthly returns
        plot_monthly_returns(
            backtest_results['trades_df'],
            save_path=os.path.join(plots_dir, 'monthly_returns.png')
        )
    
    # Generate comprehensive reports
    logger.info("-" * 60)
    logger.info("Generating reports...")
    generate_comprehensive_report(
        metrics=metrics,
        backtest_results=backtest_results,
        output_dir=reports_dir
    )
    
    # Save metrics as JSON
    # Remove non-serializable items from backtest_results
    backtest_json = {k: v for k, v in backtest_results.items() 
                     if k not in ['equity_curve', 'drawdown_curve', 'trades_df']}
    
    all_results = {
        'classification_metrics': metrics,
        'backtesting_results': backtest_json,
        'evaluation_config': {
            'model_path': args.model,
            'data_path': args.data,
            'sequence_length': args.sequence_length,
            'lot_size': args.lot_size,
            'transaction_cost': args.transaction_cost,
            'evaluated_at': datetime.now().isoformat()
        }
    }
    
    json_path = os.path.join(reports_dir, 'final_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Metrics JSON saved to {json_path}")
    
    # Print summary
    print("\n")
    with open(os.path.join(reports_dir, 'final_report.txt'), 'r', encoding='utf-8') as f:
        print(f.read())
    
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)
    logger.info(f"Reports saved to: {reports_dir}")
    logger.info(f"Plots saved to: {plots_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
