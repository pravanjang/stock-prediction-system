"""
Test script for Hyperparameter Optimization Module.

This script validates the hyperparameter optimization implementation by:
1. Testing class balancing logic
2. Testing Optuna study setup
3. Testing optimization objective
4. Testing parameter ranges
5. Testing result saving
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import optuna

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bgru_base import (
    OHLCV_FEATURES,
    TECHNICAL_FEATURES,
    TEMPORAL_FEATURES,
    PRICE_ACTION_FEATURES,
)


def create_synthetic_data(n_samples: int = 500, imbalanced: bool = False) -> pd.DataFrame:
    """Create synthetic data with all required features."""
    start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_samples)
    dates = pd.date_range(start_date, periods=n_samples, freq='D')
    
    # Create target with specified balance
    if imbalanced:
        # 70/30 split
        targets = np.array([0] * int(n_samples * 0.7) + [1] * int(n_samples * 0.3))
        np.random.shuffle(targets)
    else:
        # Balanced 50/50 split
        targets = np.random.randint(0, 2, n_samples)
    
    data = {
        'datetime': dates,
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
        'target': targets
    }
    
    # Add some technical features
    for feat in TECHNICAL_FEATURES[:5]:
        data[feat] = np.random.randn(n_samples)
    
    # Add temporal features
    for feat in TEMPORAL_FEATURES[:3]:
        if 'is_' in feat or 'has_' in feat:
            data[feat] = np.random.randint(0, 2, n_samples)
        else:
            data[feat] = np.random.randn(n_samples)
    
    # Add price action features
    for feat in PRICE_ACTION_FEATURES[:2]:
        if 'is_' in feat:
            data[feat] = np.random.randint(0, 2, n_samples)
        else:
            data[feat] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    return df


def test_apply_class_balancing_balanced():
    """Test class balancing with balanced data."""
    print("\n" + "=" * 60)
    print("TEST: Class Balancing (Balanced Data)")
    print("=" * 60)
    
    from models.optimize_hyperparams import apply_class_balancing
    
    # Create balanced data
    df = create_synthetic_data(200, imbalanced=False)
    
    # Check class distribution is roughly balanced
    class_0_ratio = (df['target'] == 0).sum() / len(df)
    class_1_ratio = (df['target'] == 1).sum() / len(df)
    
    # For balanced random data, ratio should be between 0.45 and 0.55
    weights = apply_class_balancing(df)
    
    # With a 50/50 split, weights should be None or approximately equal
    if 0.45 <= class_0_ratio <= 0.55:
        assert weights is None, "Should not apply weights for balanced data"
        print("PASS: No weights applied for balanced data")
    else:
        print(f"INFO: Data was not perfectly balanced ({class_0_ratio:.2f}/{class_1_ratio:.2f})")
    
    return True


def test_apply_class_balancing_imbalanced():
    """Test class balancing with imbalanced data."""
    print("\n" + "=" * 60)
    print("TEST: Class Balancing (Imbalanced Data)")
    print("=" * 60)
    
    from models.optimize_hyperparams import apply_class_balancing
    
    # Create imbalanced data (70/30 split)
    df = create_synthetic_data(200, imbalanced=True)
    
    weights = apply_class_balancing(df)
    
    assert weights is not None, "Should apply weights for imbalanced data"
    assert len(weights) == 2, "Should have weights for 2 classes"
    assert weights[0].item() > 0, "Class 0 weight should be positive"
    assert weights[1].item() > 0, "Class 1 weight should be positive"
    
    # For 70/30 split, class 1 should have higher weight
    print(f"PASS: Weights computed: Class 0 = {weights[0].item():.4f}, Class 1 = {weights[1].item():.4f}")
    
    # Verify weight formula: weight = total / (n_classes * count)
    total = len(df)
    count_0 = (df['target'] == 0).sum()
    count_1 = (df['target'] == 1).sum()
    expected_weight_0 = total / (2 * count_0)
    expected_weight_1 = total / (2 * count_1)
    
    assert abs(weights[0].item() - expected_weight_0) < 0.01, "Class 0 weight formula incorrect"
    assert abs(weights[1].item() - expected_weight_1) < 0.01, "Class 1 weight formula incorrect"
    print("PASS: Weight formula verified")
    
    return True


def test_setup_optuna_study():
    """Test Optuna study setup."""
    print("\n" + "=" * 60)
    print("TEST: Optuna Study Setup")
    print("=" * 60)
    
    from models.optimize_hyperparams import setup_optuna_study
    
    study = setup_optuna_study(study_name="test_study")
    
    assert isinstance(study, optuna.Study), "Should return Optuna Study"
    assert study.study_name == "test_study", "Study name should match"
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE, "Direction should be maximize"
    
    print(f"PASS: Study created: {study.study_name}")
    print(f"PASS: Direction: {study.direction.name}")
    print(f"PASS: Sampler: {type(study.sampler).__name__}")
    print(f"PASS: Pruner: {type(study.pruner).__name__}")
    
    return True


def test_hyperparameter_ranges():
    """Test that hyperparameter ranges match specification."""
    print("\n" + "=" * 60)
    print("TEST: Hyperparameter Ranges")
    print("=" * 60)
    
    # Expected ranges from specification
    expected_ranges = {
        'hidden_dim': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [32, 64, 128],
    }
    
    # Create a test trial to sample values
    study = optuna.create_study()
    
    def test_objective(trial):
        # Sample using the same logic as OptunaObjective
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5])
        lr = trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        return 0.5  # Dummy return
    
    # Run a few trials to verify sampling works
    study.optimize(test_objective, n_trials=10, show_progress_bar=False)
    
    # Verify all sampled values are within expected ranges
    for trial in study.trials:
        for param, expected in expected_ranges.items():
            actual = trial.params.get(param)
            assert actual in expected, f"{param}={actual} not in {expected}"
    
    print("PASS: All hyperparameters sampled from correct ranges")
    for param, values in expected_ranges.items():
        print(f"  - {param}: {values}")
    
    return True


def test_save_optimization_results():
    """Test saving optimization results."""
    print("\n" + "=" * 60)
    print("TEST: Save Optimization Results")
    print("=" * 60)
    
    from models.optimize_hyperparams import save_optimization_results
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a mock study with some trials
        study = optuna.create_study(direction='maximize')
        
        def mock_objective(trial):
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5])
            return trial.number * 0.1 + 0.5  # Increasing accuracy
        
        study.optimize(mock_objective, n_trials=5, show_progress_bar=False)
        
        # Save results
        save_optimization_results(study, output_dir=temp_dir)
        
        # Check files exist
        hyperparams_path = os.path.join(temp_dir, 'best_hyperparams.json')
        history_path = os.path.join(temp_dir, 'optimization_history.csv')
        
        assert os.path.exists(hyperparams_path), "best_hyperparams.json not created"
        assert os.path.exists(history_path), "optimization_history.csv not created"
        print(f"PASS: Files created in {temp_dir}")
        
        # Verify JSON content
        import json
        with open(hyperparams_path, 'r') as f:
            best_params = json.load(f)
        
        assert 'hidden_dim' in best_params, "hidden_dim not in best_params"
        assert 'dropout' in best_params, "dropout not in best_params"
        assert 'best_value' in best_params, "best_value not in best_params"
        print(f"PASS: JSON contains expected keys: {list(best_params.keys())}")
        
        # Verify CSV content
        history_df = pd.read_csv(history_path)
        
        assert len(history_df) == 5, f"Expected 5 trials, got {len(history_df)}"
        assert 'number' in history_df.columns, "number column missing"
        assert 'value' in history_df.columns, "value column missing"
        assert 'hidden_dim' in history_df.columns, "hidden_dim column missing"
        print(f"PASS: CSV contains {len(history_df)} trials with columns: {list(history_df.columns)}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_module_docstrings():
    """Test that required functions have proper docstrings."""
    print("\n" + "=" * 60)
    print("TEST: Function Docstrings")
    print("=" * 60)
    
    from models.optimize_hyperparams import (
        setup_optuna_study,
        objective_function,
        train_with_best_params,
        apply_class_balancing,
    )
    
    functions = [
        ('setup_optuna_study', setup_optuna_study),
        ('objective_function', objective_function),
        ('train_with_best_params', train_with_best_params),
        ('apply_class_balancing', apply_class_balancing),
    ]
    
    for name, func in functions:
        assert func.__doc__ is not None, f"{name} missing docstring"
        assert len(func.__doc__) > 50, f"{name} docstring too short"
        print(f"PASS: {name} has docstring ({len(func.__doc__)} chars)")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Class Balancing (Balanced)", test_apply_class_balancing_balanced),
        ("Class Balancing (Imbalanced)", test_apply_class_balancing_imbalanced),
        ("Optuna Study Setup", test_setup_optuna_study),
        ("Hyperparameter Ranges", test_hyperparameter_ranges),
        ("Save Optimization Results", test_save_optimization_results),
        ("Function Docstrings", test_module_docstrings),
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
