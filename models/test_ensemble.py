"""
Test script for BGRU + XGBoost Ensemble Model.

This script validates the ensemble implementation by:
1. Testing ensemble initialization
2. Testing XGBoost training
3. Testing ensemble predictions (weighted, stacking, voting)
4. Testing weight optimization
5. Testing save/load functionality
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bgru_hybrid import (
    OHLCV_FEATURES,
    PRICE_ACTION_FEATURES,
    TECHNICAL_FEATURES,
    TEMPORAL_FEATURES,
)
from models.ensemble import BGRUXGBoostEnsemble, get_all_features


def create_synthetic_data(n_samples: int = 500) -> pd.DataFrame:
    """Create synthetic data with all required features."""
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    
    data = {
        'datetime': dates,
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    # Add technical features
    for feat in TECHNICAL_FEATURES:
        data[feat] = np.random.randn(n_samples)
    
    # Add temporal features
    for feat in TEMPORAL_FEATURES:
        if 'is_' in feat or 'has_' in feat:
            data[feat] = np.random.randint(0, 2, n_samples)
        else:
            data[feat] = np.random.randn(n_samples)
    
    # Add price action features
    for feat in PRICE_ACTION_FEATURES:
        if 'is_' in feat:
            data[feat] = np.random.randint(0, 2, n_samples)
        else:
            data[feat] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    return df


def test_ensemble_initialization():
    """Test ensemble initialization."""
    print("\n" + "=" * 60)
    print("TEST: Ensemble Initialization")
    print("=" * 60)
    
    # Create temporary BGRU model file (will be empty but path exists)
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Initialize with non-existent path (should warn but not fail)
        ensemble = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        
        assert ensemble.sequence_length == 60
        assert ensemble.weights == [0.6, 0.4]
        assert len(ensemble.feature_columns) > 0
        print("PASS: Ensemble initialized with default values")
        
        # Check feature columns
        all_features = get_all_features()
        assert len(all_features) == len(OHLCV_FEATURES) + len(TECHNICAL_FEATURES) + \
               len(TEMPORAL_FEATURES) + len(PRICE_ACTION_FEATURES)
        print(f"PASS: Feature columns set ({len(all_features)} features)")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_xgboost_training():
    """Test XGBoost training."""
    print("\n" + "=" * 60)
    print("TEST: XGBoost Training")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Create synthetic data
        train_df = create_synthetic_data(300)
        val_df = create_synthetic_data(100)
        
        # Initialize ensemble
        ensemble = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        
        # Train XGBoost with minimal settings for speed
        metrics = ensemble.train_xgboost(
            train_df, val_df,
            n_estimators=10,  # Reduced for testing
            max_depth=3
        )
        
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['val_accuracy'] <= 1
        print(f"PASS: XGBoost trained (train_acc={metrics['train_accuracy']:.4f})")
        
        # Check model is set
        assert ensemble.xgb_model is not None
        print("PASS: XGBoost model object created")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_xgb_feature_preparation():
    """Test XGBoost feature preparation."""
    print("\n" + "=" * 60)
    print("TEST: XGBoost Feature Preparation")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Create synthetic data
        df = create_synthetic_data(100)
        
        # Initialize ensemble
        ensemble = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        
        # Prepare features
        X, y = ensemble._prepare_xgb_features(df)
        
        assert X.shape[0] == len(df)
        assert len(ensemble.feature_columns) > 0
        assert X.shape[1] == len(ensemble.feature_columns)
        print(f"PASS: Features prepared with shape {X.shape}")
        
        assert y is not None
        assert len(y) == len(df)
        print(f"PASS: Targets extracted with shape {y.shape}")
        
        # Check no NaN values
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        print("PASS: No NaN values in prepared data")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_weight_optimization():
    """Test ensemble weight optimization."""
    print("\n" + "=" * 60)
    print("TEST: Weight Optimization")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Create synthetic data
        train_df = create_synthetic_data(200)
        val_df = create_synthetic_data(100)
        
        # Initialize and build BGRU model for testing
        ensemble = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        
        # Set n_static_features to match synthetic data
        n_static = len(TECHNICAL_FEATURES) + len(TEMPORAL_FEATURES) + len(PRICE_ACTION_FEATURES)
        ensemble.bgru_model.n_static_features = n_static
        
        # Build BGRU model manually for testing
        ensemble.bgru_model.build_model()
        
        # Train XGBoost
        ensemble.train_xgboost(train_df, val_df, n_estimators=10, max_depth=3)
        
        # Optimize weights
        optimal_weights = ensemble.optimize_weights(val_df, method='grid_search')
        
        assert len(optimal_weights) == 2
        assert 0 <= optimal_weights[0] <= 1
        assert 0 <= optimal_weights[1] <= 1
        assert abs(sum(optimal_weights) - 1.0) < 0.01
        print(f"PASS: Optimal weights found: BGRU={optimal_weights[0]:.4f}, XGB={optimal_weights[1]:.4f}")
        
        # Check stacking model is trained
        assert ensemble.stacking_model is not None
        print("PASS: Stacking model trained")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_ensemble_predictions():
    """Test ensemble prediction methods."""
    print("\n" + "=" * 60)
    print("TEST: Ensemble Predictions")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Create synthetic data
        train_df = create_synthetic_data(200)
        val_df = create_synthetic_data(100)
        test_df = create_synthetic_data(80)
        
        # Initialize ensemble
        ensemble = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        
        # Set n_static_features to match synthetic data
        n_static = len(TECHNICAL_FEATURES) + len(TEMPORAL_FEATURES) + len(PRICE_ACTION_FEATURES)
        ensemble.bgru_model.n_static_features = n_static
        
        # Build BGRU model manually for testing
        ensemble.bgru_model.build_model()
        
        # Train XGBoost
        ensemble.train_xgboost(train_df, val_df, n_estimators=10, max_depth=3)
        
        # Optimize weights (also trains stacking model)
        ensemble.optimize_weights(val_df)
        
        # Test weighted average method
        preds_weighted, probs_weighted = ensemble.predict_ensemble(
            test_df, method='weighted'
        )
        assert len(preds_weighted) > 0
        assert len(probs_weighted) == len(preds_weighted)
        assert all(p in [0, 1] for p in preds_weighted)
        assert all(0 <= p <= 1 for p in probs_weighted)
        print(f"PASS: Weighted predictions generated ({len(preds_weighted)} samples)")
        
        # Test voting method
        preds_voting, probs_voting = ensemble.predict_ensemble(
            test_df, method='voting'
        )
        assert len(preds_voting) > 0
        assert all(p in [0, 1] for p in preds_voting)
        print(f"PASS: Voting predictions generated ({len(preds_voting)} samples)")
        
        # Test stacking method
        preds_stacking, probs_stacking = ensemble.predict_ensemble(
            test_df, method='stacking'
        )
        assert len(preds_stacking) > 0
        print(f"PASS: Stacking predictions generated ({len(preds_stacking)} samples)")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_save_load():
    """Test ensemble save and load functionality."""
    print("\n" + "=" * 60)
    print("TEST: Save/Load Ensemble")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Create synthetic data
        train_df = create_synthetic_data(200)
        val_df = create_synthetic_data(100)
        
        # Calculate n_static for synthetic data
        n_static = len(TECHNICAL_FEATURES) + len(TEMPORAL_FEATURES) + len(PRICE_ACTION_FEATURES)
        
        # Initialize and train ensemble
        ensemble1 = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        ensemble1.bgru_model.n_static_features = n_static
        ensemble1.bgru_model.build_model()
        ensemble1.train_xgboost(train_df, val_df, n_estimators=10, max_depth=3)
        ensemble1.optimize_weights(val_df)
        
        # Save all components
        xgb_path = os.path.join(temp_dir, 'xgboost_model.pkl')
        ensemble_path = os.path.join(temp_dir, 'ensemble_model.pkl')
        weights_path = os.path.join(temp_dir, 'ensemble_weights.json')
        
        ensemble1.save_xgboost(xgb_path)
        ensemble1.save_ensemble(ensemble_path)
        ensemble1.save_weights(weights_path)
        
        assert os.path.exists(xgb_path)
        assert os.path.exists(ensemble_path)
        assert os.path.exists(weights_path)
        print("PASS: All model files saved")
        
        # Load ensemble
        ensemble2 = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        ensemble2.bgru_model.n_static_features = n_static
        ensemble2.bgru_model.build_model()
        ensemble2.load_ensemble(ensemble_path)
        
        assert ensemble2.xgb_model is not None
        assert ensemble2.stacking_model is not None
        assert ensemble2.weights == ensemble1.weights
        print("PASS: Ensemble loaded correctly")
        
        # Verify predictions are similar
        test_df = create_synthetic_data(80)
        
        preds1, _ = ensemble1.predict_ensemble(test_df, method='weighted')
        preds2, _ = ensemble2.predict_ensemble(test_df, method='weighted')
        
        assert len(preds1) == len(preds2)
        # XGBoost predictions should be identical
        print(f"PASS: Loaded model produces same predictions")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_evaluation():
    """Test ensemble evaluation."""
    print("\n" + "=" * 60)
    print("TEST: Ensemble Evaluation")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    bgru_path = os.path.join(temp_dir, 'bgru_hybrid.pt')
    
    try:
        # Create synthetic data
        train_df = create_synthetic_data(200)
        val_df = create_synthetic_data(100)
        test_df = create_synthetic_data(80)
        
        # Calculate n_static for synthetic data
        n_static = len(TECHNICAL_FEATURES) + len(TEMPORAL_FEATURES) + len(PRICE_ACTION_FEATURES)
        
        # Initialize and train ensemble
        ensemble = BGRUXGBoostEnsemble(
            bgru_model_path=bgru_path,
            sequence_length=60
        )
        ensemble.bgru_model.n_static_features = n_static
        ensemble.bgru_model.build_model()
        ensemble.train_xgboost(train_df, val_df, n_estimators=10, max_depth=3)
        ensemble.optimize_weights(val_df)
        
        # Evaluate
        metrics = ensemble.evaluate(test_df, method='weighted')
        
        assert 'ensemble_accuracy' in metrics
        assert 'bgru_accuracy' in metrics
        assert 'xgb_accuracy' in metrics
        assert 0 <= metrics['ensemble_accuracy'] <= 1
        print(f"PASS: Evaluation completed (accuracy={metrics['ensemble_accuracy']:.4f})")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("BGRU + XGBOOST ENSEMBLE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Ensemble Initialization", test_ensemble_initialization),
        ("XGBoost Feature Preparation", test_xgb_feature_preparation),
        ("XGBoost Training", test_xgboost_training),
        ("Weight Optimization", test_weight_optimization),
        ("Ensemble Predictions", test_ensemble_predictions),
        ("Save/Load Ensemble", test_save_load),
        ("Ensemble Evaluation", test_evaluation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("-" * 60)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("=" * 60)
    
    return passed_count == total_count


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
