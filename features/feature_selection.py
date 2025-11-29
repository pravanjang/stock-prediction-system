#!/usr/bin/env python3
"""
Feature Selection Module for Stock Prediction System.

This module provides feature selection and importance analysis:
- Permutation importance calculation
- Correlation matrix analysis
- Multicollinearity removal (VIF-based)
- Top feature selection
- Feature importance visualization
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_permutation_importance(
    model: Any,
    test_df: pd.DataFrame,
    n_repeats: int = 10,
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Calculate permutation importance for each feature.
    
    Shuffles each feature independently, measures the drop in accuracy,
    and ranks features by importance.
    
    Args:
        model: Trained model with predict() method
        test_df: Test DataFrame with features and target
        n_repeats: Number of times to repeat shuffling (default: 10)
        feature_columns: List of feature columns to analyze. If None,
                        uses all columns except 'target' and 'datetime'
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    logger.info("=" * 60)
    logger.info("Calculating Permutation Importance")
    logger.info("=" * 60)
    
    np.random.seed(random_state)
    
    # Get feature columns
    if feature_columns is None:
        exclude_cols = ['target', 'datetime', 'date']
        feature_columns = [
            c for c in test_df.columns
            if c not in exclude_cols and not c.startswith('Unnamed')
        ]
    
    logger.info(f"Analyzing {len(feature_columns)} features with {n_repeats} repeats")
    
    # Get baseline accuracy
    try:
        predictions, _ = model.predict(test_df)
        _, _, y_test = model.prepare_data(test_df)
        
        if len(predictions) != len(y_test):
            # Align lengths
            min_len = min(len(predictions), len(y_test))
            predictions = predictions[:min_len]
            y_test = y_test[:min_len]
        
        baseline_acc = (predictions == y_test).mean()
        logger.info(f"Baseline accuracy: {baseline_acc:.4f}")
    except Exception as e:
        logger.error(f"Error getting baseline predictions: {e}")
        raise
    
    importance_scores: Dict[str, float] = {}
    
    for feature in tqdm(feature_columns, desc="Computing importance"):
        drop_scores = []
        
        for _ in range(n_repeats):
            # Create a copy with shuffled feature
            df_shuffled = test_df.copy()
            df_shuffled[feature] = np.random.permutation(
                df_shuffled[feature].values
            )
            
            try:
                # Get predictions on shuffled data
                preds_shuffled, _ = model.predict(df_shuffled)
                _, _, y_shuffled = model.prepare_data(df_shuffled)
                
                if len(preds_shuffled) != len(y_shuffled):
                    min_len = min(len(preds_shuffled), len(y_shuffled))
                    preds_shuffled = preds_shuffled[:min_len]
                    y_shuffled = y_shuffled[:min_len]
                
                shuffled_acc = (preds_shuffled == y_shuffled).mean()
                drop = baseline_acc - shuffled_acc
                drop_scores.append(drop)
            except Exception:
                # Skip this repeat if error
                continue
        
        if drop_scores:
            importance_scores[feature] = float(np.mean(drop_scores))
        else:
            importance_scores[feature] = 0.0
    
    # Sort by importance
    importance_scores = dict(
        sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    )
    
    logger.info("Top 10 most important features:")
    for i, (feat, score) in enumerate(list(importance_scores.items())[:10]):
        logger.info(f"  {i+1}. {feat}: {score:.6f}")
    
    return importance_scores


def calculate_correlation_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix for all features.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to analyze
    
    Returns:
        Correlation matrix as DataFrame
    """
    logger.info("Calculating correlation matrix...")
    
    if feature_columns is None:
        exclude_cols = ['target', 'datetime', 'date']
        feature_columns = [
            c for c in df.columns
            if c not in exclude_cols and not c.startswith('Unnamed')
        ]
    
    # Select only numeric columns
    numeric_cols = []
    for col in feature_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    corr_matrix = df[numeric_cols].corr()
    
    logger.info(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Log highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
        for col1, col2, corr_val in high_corr_pairs[:10]:
            logger.info(f"  {col1} <-> {col2}: {corr_val:.4f}")
    
    return corr_matrix


def calculate_vif(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor (VIF) for all features.
    
    VIF measures how much the variance of a coefficient is inflated
    due to multicollinearity.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to analyze
    
    Returns:
        Dictionary mapping feature names to VIF values
    """
    logger.info("Calculating VIF for multicollinearity detection...")
    
    if feature_columns is None:
        exclude_cols = ['target', 'datetime', 'date']
        feature_columns = [
            c for c in df.columns
            if c not in exclude_cols and not c.startswith('Unnamed')
        ]
    
    # Filter to numeric columns and remove NaN
    numeric_cols = []
    for col in feature_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if not df[col].isna().all():
                numeric_cols.append(col)
    
    # Prepare data
    X = df[numeric_cols].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    vif_values: Dict[str, float] = {}
    
    for i, col in enumerate(tqdm(numeric_cols, desc="Computing VIF")):
        try:
            # Get the feature column
            y = X[col].values
            
            # Get all other columns
            other_cols = [c for c in numeric_cols if c != col]
            X_other = X[other_cols].values
            
            # Add constant term
            X_other = np.column_stack([np.ones(len(y)), X_other])
            
            # Calculate R-squared using OLS with least squares
            # R^2 = 1 - (SS_res / SS_tot)
            try:
                # Use lstsq with explicit rcond for numerical stability
                coeffs = np.linalg.lstsq(X_other, y, rcond=1e-15)[0]
                y_pred = X_other @ coeffs
                
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                
                if ss_tot > 0:
                    r_squared = 1 - (ss_res / ss_tot)
                    r_squared = max(0, min(r_squared, 0.9999))  # Clip to avoid inf
                    vif = 1 / (1 - r_squared)
                else:
                    vif = 1.0
            except np.linalg.LinAlgError:
                vif = float('inf')
            
            vif_values[col] = float(vif)
        except Exception as e:
            logger.warning(f"Error calculating VIF for {col}: {e}")
            vif_values[col] = float('inf')
    
    # Sort by VIF
    vif_values = dict(sorted(vif_values.items(), key=lambda x: x[1], reverse=True))
    
    # Log high VIF features
    high_vif = [(k, v) for k, v in vif_values.items() if v > 10]
    if high_vif:
        logger.info(f"Found {len(high_vif)} features with VIF > 10:")
        for feat, vif in high_vif[:10]:
            logger.info(f"  {feat}: VIF = {vif:.2f}")
    
    return vif_values


def remove_multicollinear_features(
    df: pd.DataFrame,
    vif_threshold: float = 10,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with VIF greater than threshold.
    
    Iteratively removes features with highest VIF until all
    remaining features have VIF below threshold.
    
    Args:
        df: DataFrame with features
        vif_threshold: VIF threshold for removal (default: 10)
        feature_columns: List of feature columns to analyze
    
    Returns:
        Tuple of (filtered DataFrame, list of removed feature names)
    """
    logger.info("=" * 60)
    logger.info(f"Removing Multicollinear Features (VIF > {vif_threshold})")
    logger.info("=" * 60)
    
    if feature_columns is None:
        exclude_cols = ['target', 'datetime', 'date']
        feature_columns = [
            c for c in df.columns
            if c not in exclude_cols and not c.startswith('Unnamed')
        ]
    
    removed_features: List[str] = []
    remaining_features = feature_columns.copy()
    
    iteration = 0
    max_iterations = len(feature_columns)
    
    while iteration < max_iterations:
        iteration += 1
        
        # Calculate VIF for remaining features
        vif_values = calculate_vif(df, remaining_features)
        
        # Find max VIF
        if not vif_values:
            break
        
        max_vif_feature = max(vif_values, key=vif_values.get)
        max_vif = vif_values[max_vif_feature]
        
        if max_vif <= vif_threshold:
            logger.info(
                f"All remaining features have VIF <= {vif_threshold}"
            )
            break
        
        # Remove feature with highest VIF
        logger.info(
            f"Iteration {iteration}: Removing {max_vif_feature} "
            f"(VIF = {max_vif:.2f})"
        )
        removed_features.append(max_vif_feature)
        remaining_features.remove(max_vif_feature)
        
        if len(remaining_features) < 2:
            logger.warning("Only 1 feature remaining. Stopping.")
            break
    
    logger.info(f"Removed {len(removed_features)} multicollinear features")
    logger.info(f"Remaining features: {len(remaining_features)}")
    
    # Create filtered DataFrame
    keep_cols = [c for c in df.columns if c not in removed_features]
    df_filtered = df[keep_cols].copy()
    
    return df_filtered, removed_features


def select_top_features(
    importance_dict: Dict[str, float],
    top_k: int = 40
) -> List[str]:
    """
    Select top K most important features.
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        top_k: Number of top features to select (default: 40)
    
    Returns:
        List of top K feature names
    """
    logger.info(f"Selecting top {top_k} features...")
    
    # Sort by importance
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top K
    top_features = [feat for feat, _ in sorted_features[:top_k]]
    
    logger.info(f"Selected {len(top_features)} top features")
    
    return top_features


def save_feature_importance_plot(
    importance_dict: Dict[str, float],
    save_path: str,
    top_n: int = 20
) -> None:
    """
    Create and save feature importance horizontal bar chart.
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        save_path: Path to save the plot
        top_n: Number of top features to plot (default: 20)
    """
    logger.info(f"Creating feature importance plot (top {top_n})...")
    
    # Sort and get top N
    sorted_items = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Reverse for horizontal bar chart (top at top)
    features = features[::-1]
    scores = scores[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(features, scores, color=colors)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(
            width + max(scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{score:.4f}',
            ha='left',
            va='center',
            fontsize=9
        )
    
    ax.set_xlabel('Importance Score (Accuracy Drop)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance (Permutation)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {save_path}")


def save_feature_importance_csv(
    importance_dict: Dict[str, float],
    save_path: str
) -> None:
    """
    Save feature importance scores to CSV.
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        save_path: Path to save the CSV
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([
        {'feature': feat, 'importance': score}
        for feat, score in importance_dict.items()
    ])
    df = df.sort_values('importance', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    df = df[['rank', 'feature', 'importance']]
    
    df.to_csv(save_path, index=False)
    logger.info(f"Feature importance saved to {save_path}")


def save_feature_list(
    features: List[str],
    save_path: str
) -> None:
    """
    Save feature list to text file.
    
    Args:
        features: List of feature names
        save_path: Path to save the text file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    
    logger.info(f"Feature list saved to {save_path}")


def load_model(model_path: str) -> Any:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
    
    Returns:
        Loaded model instance
    """
    # Try loading as HybridBGRUModel first
    try:
        from models.bgru_hybrid import HybridBGRUModel
        model = HybridBGRUModel()
        model.load_model(model_path)
        logger.info(f"Loaded HybridBGRUModel from {model_path}")
        return model
    except Exception:
        pass
    
    # Try loading as BGRUPredictor
    try:
        from models.bgru_base import BGRUPredictor
        model = BGRUPredictor()
        model.load_model(model_path)
        logger.info(f"Loaded BGRUPredictor from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Could not load model from {model_path}: {e}")
        raise


def run_feature_analysis(
    model_path: str,
    data_path: str,
    output_dir: str = 'features/',
    n_repeats: int = 10,
    vif_threshold: float = 10,
    top_k: int = 40
) -> Dict[str, Any]:
    """
    Run complete feature analysis pipeline.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to test data CSV
        output_dir: Directory to save outputs
        n_repeats: Number of repeats for permutation importance
        vif_threshold: VIF threshold for multicollinearity removal
        top_k: Number of top features to select
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info("Feature Selection & Importance Analysis")
    logger.info("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # 1. Calculate permutation importance
    importance_dict = calculate_permutation_importance(
        model, df, n_repeats=n_repeats
    )
    
    # Save importance results
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    save_feature_importance_csv(importance_dict, importance_csv_path)
    
    # Save importance plot
    plot_path = os.path.join(output_dir, 'feature_importance_plot.png')
    save_feature_importance_plot(importance_dict, plot_path)
    
    # 2. Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df)
    corr_path = os.path.join(output_dir, 'correlation_matrix.csv')
    corr_matrix.to_csv(corr_path)
    logger.info(f"Correlation matrix saved to {corr_path}")
    
    # 3. Remove multicollinear features
    _, removed_features = remove_multicollinear_features(
        df, vif_threshold=vif_threshold
    )
    
    removed_path = os.path.join(output_dir, 'removed_features.txt')
    save_feature_list(removed_features, removed_path)
    
    # 4. Select top features (excluding removed multicollinear ones)
    filtered_importance = {
        k: v for k, v in importance_dict.items()
        if k not in removed_features
    }
    top_features = select_top_features(filtered_importance, top_k=top_k)
    
    selected_path = os.path.join(output_dir, 'selected_features.txt')
    save_feature_list(top_features, selected_path)
    
    logger.info("=" * 60)
    logger.info("Feature Analysis Complete")
    logger.info("=" * 60)
    logger.info(f"Total features analyzed: {len(importance_dict)}")
    logger.info(f"Removed (VIF > {vif_threshold}): {len(removed_features)}")
    logger.info(f"Selected top features: {len(top_features)}")
    logger.info("=" * 60)
    
    return {
        'importance': importance_dict,
        'correlation_matrix': corr_matrix,
        'removed_features': removed_features,
        'selected_features': top_features
    }


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Feature selection and importance analysis'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/bgru_hybrid.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/test_final.csv',
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='features/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--n_repeats',
        type=int,
        default=10,
        help='Number of repeats for permutation importance (default: 10)'
    )
    parser.add_argument(
        '--vif_threshold',
        type=float,
        default=10.0,
        help='VIF threshold for multicollinearity removal (default: 10)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=40,
        help='Number of top features to select (default: 40)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        logger.error(f"Model checkpoint not found: {args.model}")
        return 1
    
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return 1
    
    # Run analysis
    try:
        run_feature_analysis(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output_dir,
            n_repeats=args.n_repeats,
            vif_threshold=args.vif_threshold,
            top_k=args.top_k
        )
        return 0
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
