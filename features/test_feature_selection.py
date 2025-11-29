"""
Test script for Feature Selection Module.

This script validates the feature selection implementation by:
1. Testing correlation matrix calculation
2. Testing VIF calculation
3. Testing feature selection functions
4. Testing feature importance plotting
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_selection import (
    calculate_correlation_matrix,
    calculate_vif,
    remove_multicollinear_features,
    select_top_features,
    save_feature_importance_plot,
    save_feature_importance_csv,
    save_feature_list
)


def create_synthetic_data(n_samples: int = 500) -> pd.DataFrame:
    """Create synthetic data for testing."""
    np.random.seed(42)
    
    data = {
        'datetime': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
        'target': np.random.randint(0, 2, n_samples),
        # Independent features
        'feature_a': np.random.randn(n_samples),
        'feature_b': np.random.randn(n_samples),
        'feature_c': np.random.randn(n_samples),
        # Highly correlated features
        'feature_d': np.random.randn(n_samples),
    }
    
    # Make feature_e highly correlated with feature_d
    data['feature_e'] = data['feature_d'] * 0.95 + np.random.randn(n_samples) * 0.1
    
    # Make feature_f highly correlated with feature_d
    data['feature_f'] = data['feature_d'] * 0.9 + np.random.randn(n_samples) * 0.2
    
    return pd.DataFrame(data)


def test_correlation_matrix():
    """Test correlation matrix calculation."""
    print("\n" + "=" * 60)
    print("TEST: Correlation Matrix")
    print("=" * 60)
    
    df = create_synthetic_data()
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df)
    
    # Check output type
    assert isinstance(corr_matrix, pd.DataFrame), "Should return a DataFrame"
    print("PASS: Returns DataFrame")
    
    # Check shape (should be square)
    assert corr_matrix.shape[0] == corr_matrix.shape[1], "Correlation matrix should be square"
    print(f"PASS: Square matrix: {corr_matrix.shape}")
    
    # Check diagonal is 1
    diag_vals = np.diag(corr_matrix.values)
    assert np.allclose(diag_vals, 1.0), "Diagonal should be 1.0"
    print("PASS: Diagonal values are 1.0")
    
    # Check for high correlation between feature_d and feature_e
    if 'feature_d' in corr_matrix.columns and 'feature_e' in corr_matrix.columns:
        corr_de = abs(corr_matrix.loc['feature_d', 'feature_e'])
        assert corr_de > 0.9, f"Expected high correlation between feature_d and feature_e, got {corr_de}"
        print(f"PASS: Detected high correlation: {corr_de:.4f}")
    
    return True


def test_vif_calculation():
    """Test VIF calculation."""
    print("\n" + "=" * 60)
    print("TEST: VIF Calculation")
    print("=" * 60)
    
    df = create_synthetic_data()
    
    feature_cols = ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f']
    
    vif_values = calculate_vif(df, feature_cols)
    
    # Check output type
    assert isinstance(vif_values, dict), "Should return a dictionary"
    print("PASS: Returns dictionary")
    
    # Check all features have VIF
    for feat in feature_cols:
        assert feat in vif_values, f"Missing VIF for {feat}"
    print(f"PASS: VIF calculated for all {len(feature_cols)} features")
    
    # VIF for independent features should be lower
    independent_vifs = [vif_values['feature_a'], vif_values['feature_b'], vif_values['feature_c']]
    print(f"INFO: Independent feature VIFs: {independent_vifs}")
    
    # Correlated features should have higher VIF
    correlated_vifs = [vif_values['feature_d'], vif_values['feature_e'], vif_values['feature_f']]
    print(f"INFO: Correlated feature VIFs: {correlated_vifs}")
    
    return True


def test_multicollinearity_removal():
    """Test multicollinearity removal."""
    print("\n" + "=" * 60)
    print("TEST: Multicollinearity Removal")
    print("=" * 60)
    
    df = create_synthetic_data()
    
    feature_cols = ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f']
    
    df_filtered, removed = remove_multicollinear_features(
        df, vif_threshold=10, feature_columns=feature_cols
    )
    
    # Check output types
    assert isinstance(df_filtered, pd.DataFrame), "Should return a DataFrame"
    assert isinstance(removed, list), "Should return a list of removed features"
    print(f"PASS: Correct return types")
    
    # Check that removed features are not in filtered DataFrame
    for feat in removed:
        assert feat not in df_filtered.columns, f"{feat} should not be in filtered DataFrame"
    print(f"PASS: Removed features not in filtered DataFrame")
    
    # Check that non-removed features are preserved
    remaining = [f for f in feature_cols if f not in removed]
    for feat in remaining:
        assert feat in df_filtered.columns, f"{feat} should be in filtered DataFrame"
    print(f"PASS: Remaining features preserved")
    
    print(f"INFO: Removed {len(removed)} features: {removed}")
    print(f"INFO: Remaining {len(remaining)} features")
    
    return True


def test_top_feature_selection():
    """Test top feature selection."""
    print("\n" + "=" * 60)
    print("TEST: Top Feature Selection")
    print("=" * 60)
    
    # Create synthetic importance scores
    importance_dict = {
        'feature_1': 0.15,
        'feature_2': 0.12,
        'feature_3': 0.10,
        'feature_4': 0.08,
        'feature_5': 0.05,
        'feature_6': 0.03,
        'feature_7': 0.02,
        'feature_8': 0.01,
    }
    
    # Select top 3
    top_3 = select_top_features(importance_dict, top_k=3)
    
    assert len(top_3) == 3, f"Expected 3 features, got {len(top_3)}"
    print(f"PASS: Selected {len(top_3)} features")
    
    assert top_3[0] == 'feature_1', f"Expected feature_1 first, got {top_3[0]}"
    assert top_3[1] == 'feature_2', f"Expected feature_2 second, got {top_3[1]}"
    assert top_3[2] == 'feature_3', f"Expected feature_3 third, got {top_3[2]}"
    print(f"PASS: Features selected in correct order: {top_3}")
    
    # Select top 5
    top_5 = select_top_features(importance_dict, top_k=5)
    assert len(top_5) == 5
    print(f"PASS: Top 5 selection works: {top_5}")
    
    return True


def test_save_functions():
    """Test save functions."""
    print("\n" + "=" * 60)
    print("TEST: Save Functions")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test save_feature_importance_csv
        importance_dict = {
            'feature_1': 0.15,
            'feature_2': 0.12,
            'feature_3': 0.10,
        }
        
        csv_path = os.path.join(temp_dir, 'importance.csv')
        save_feature_importance_csv(importance_dict, csv_path)
        
        assert os.path.exists(csv_path), "CSV file not created"
        df = pd.read_csv(csv_path)
        assert 'feature' in df.columns
        assert 'importance' in df.columns
        assert len(df) == 3
        print("PASS: Feature importance CSV saved correctly")
        
        # Test save_feature_list
        feature_list = ['feature_1', 'feature_2', 'feature_3']
        txt_path = os.path.join(temp_dir, 'features.txt')
        save_feature_list(feature_list, txt_path)
        
        assert os.path.exists(txt_path), "Text file not created"
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        assert lines == feature_list
        print("PASS: Feature list saved correctly")
        
        # Test save_feature_importance_plot
        plot_path = os.path.join(temp_dir, 'plot.png')
        save_feature_importance_plot(importance_dict, plot_path, top_n=3)
        
        assert os.path.exists(plot_path), "Plot file not created"
        print("PASS: Feature importance plot saved correctly")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Correlation Matrix", test_correlation_matrix),
        ("VIF Calculation", test_vif_calculation),
        ("Multicollinearity Removal", test_multicollinearity_removal),
        ("Top Feature Selection", test_top_feature_selection),
        ("Save Functions", test_save_functions),
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
