"""
Test script for Technical Indicators Module

This script validates the technical indicators implementation by:
1. Loading sample data
2. Applying all indicators
3. Verifying no NaN values remain
4. Checking indicator value ranges
5. Verifying scaler save/load functionality
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.technical import (
    add_technical_indicators,
    get_indicator_columns,
    add_momentum_indicators,
    add_trend_indicators,
    add_volume_indicators,
    add_volatility_indicators,
    handle_nan_values,
    normalize_indicators
)


def test_indicator_calculation():
    """Test that all indicators are calculated correctly."""
    print("\n" + "=" * 60)
    print("TEST: Indicator Calculation")
    print("=" * 60)
    
    # Load sample data
    data_path = os.path.join('data', 'processed', 'train.csv')
    if not os.path.exists(data_path):
        print(f"SKIP: Test data not found at {data_path}")
        return True
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Add indicators without normalization first
    df_with_indicators = add_technical_indicators(df, normalize=False)
    
    # Get expected indicator columns
    expected_indicators = get_indicator_columns()
    
    # Check that all expected indicators are present
    missing_indicators = [col for col in expected_indicators if col not in df_with_indicators.columns]
    if missing_indicators:
        print(f"FAIL: Missing indicators: {missing_indicators}")
        return False
    
    print(f"PASS: All {len(expected_indicators)} expected indicators are present")
    
    # Check indicator count
    original_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'target']
    new_cols = [col for col in df_with_indicators.columns if col not in original_cols]
    print(f"INFO: Added {len(new_cols)} indicator columns")
    
    return True


def test_no_nan_values():
    """Test that NaN values are properly handled."""
    print("\n" + "=" * 60)
    print("TEST: NaN Value Handling")
    print("=" * 60)
    
    # Load sample data
    data_path = os.path.join('data', 'processed', 'train.csv')
    if not os.path.exists(data_path):
        print(f"SKIP: Test data not found at {data_path}")
        return True
    
    df = pd.read_csv(data_path)
    
    # Add indicators with normalization
    df_with_indicators = add_technical_indicators(df, normalize=True)
    
    # Check for NaN values
    nan_counts = df_with_indicators.isna().sum()
    total_nan = nan_counts.sum()
    
    if total_nan > 0:
        print(f"FAIL: Found {total_nan} NaN values")
        print(f"Columns with NaN: {nan_counts[nan_counts > 0].to_dict()}")
        return False
    
    print(f"PASS: No NaN values in output")
    
    # Check for inf values
    inf_counts = np.isinf(df_with_indicators.select_dtypes(include=[np.number])).sum()
    total_inf = inf_counts.sum()
    
    if total_inf > 0:
        print(f"FAIL: Found {total_inf} Inf values")
        return False
    
    print(f"PASS: No Inf values in output")
    
    return True


def test_normalization_range():
    """Test that normalized indicators are in [0, 1] range."""
    print("\n" + "=" * 60)
    print("TEST: Normalization Range")
    print("=" * 60)
    
    # Load sample data
    data_path = os.path.join('data', 'processed', 'train.csv')
    if not os.path.exists(data_path):
        print(f"SKIP: Test data not found at {data_path}")
        return True
    
    df = pd.read_csv(data_path)
    
    # Create temp directory for scaler
    temp_dir = tempfile.mkdtemp()
    scaler_path = os.path.join(temp_dir, 'test_scaler.pkl')
    
    try:
        # Add indicators with normalization
        df_with_indicators = add_technical_indicators(
            df, normalize=True, scaler_path=scaler_path
        )
        
        # Get indicator columns
        indicator_cols = get_indicator_columns()
        existing_indicators = [col for col in indicator_cols if col in df_with_indicators.columns]
        
        # Check range (with small tolerance for floating point errors)
        epsilon = 1e-9
        out_of_range = []
        for col in existing_indicators:
            min_val = df_with_indicators[col].min()
            max_val = df_with_indicators[col].max()
            
            if min_val < -epsilon or max_val > 1 + epsilon:
                out_of_range.append((col, min_val, max_val))
        
        if out_of_range:
            print(f"FAIL: {len(out_of_range)} columns out of [0, 1] range:")
            for col, min_val, max_val in out_of_range[:5]:  # Show first 5
                print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")
            return False
        
        print(f"PASS: All {len(existing_indicators)} indicator columns are in [0, 1] range")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
    
    return True


def test_scaler_save_load():
    """Test scaler save and load functionality."""
    print("\n" + "=" * 60)
    print("TEST: Scaler Save/Load")
    print("=" * 60)
    
    # Load sample data
    data_path = os.path.join('data', 'processed', 'train.csv')
    if not os.path.exists(data_path):
        print(f"SKIP: Test data not found at {data_path}")
        return True
    
    df = pd.read_csv(data_path)
    
    # Create temp directory for scaler
    temp_dir = tempfile.mkdtemp()
    scaler_path = os.path.join(temp_dir, 'test_scaler.pkl')
    
    try:
        # Add indicators and fit scaler
        df_fitted = add_technical_indicators(
            df.copy(), normalize=True, scaler_path=scaler_path, fit_scaler=True
        )
        
        # Check scaler file exists
        if not os.path.exists(scaler_path):
            print(f"FAIL: Scaler not saved to {scaler_path}")
            return False
        
        print(f"PASS: Scaler saved to {scaler_path}")
        
        # Load scaler and transform same data
        df_loaded = add_technical_indicators(
            df.copy(), normalize=True, scaler_path=scaler_path, fit_scaler=False
        )
        
        # Compare results
        indicator_cols = get_indicator_columns()
        existing_indicators = [col for col in indicator_cols if col in df_fitted.columns]
        
        for col in existing_indicators:
            diff = (df_fitted[col] - df_loaded[col]).abs().max()
            if diff > 1e-6:
                print(f"FAIL: Column {col} differs after scaler load (max diff: {diff})")
                return False
        
        print(f"PASS: Scaler load produces identical results")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
    
    return True


def test_individual_indicator_groups():
    """Test each indicator group separately."""
    print("\n" + "=" * 60)
    print("TEST: Individual Indicator Groups")
    print("=" * 60)
    
    # Load sample data
    data_path = os.path.join('data', 'processed', 'train.csv')
    if not os.path.exists(data_path):
        print(f"SKIP: Test data not found at {data_path}")
        return True
    
    df = pd.read_csv(data_path)
    
    # Test momentum indicators
    df_momentum = add_momentum_indicators(df.copy())
    momentum_cols = ['rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_hist',
                     'stoch_k', 'stoch_d', 'adx_14', 'momentum_10']
    missing_momentum = [col for col in momentum_cols if col not in df_momentum.columns]
    if missing_momentum:
        print(f"FAIL: Momentum - missing: {missing_momentum}")
        return False
    print(f"PASS: Momentum indicators ({len(momentum_cols)} columns)")
    
    # Test trend indicators
    df_trend = add_trend_indicators(df.copy())
    trend_cols = ['ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_20', 'sma_50',
                  'supertrend_10_3', 'supertrend_7_2', 'bb_upper', 'bb_middle',
                  'bb_lower', 'bb_width', 'psar']
    missing_trend = [col for col in trend_cols if col not in df_trend.columns]
    if missing_trend:
        print(f"FAIL: Trend - missing: {missing_trend}")
        return False
    print(f"PASS: Trend indicators ({len(trend_cols)} columns)")
    
    # Test volume indicators
    df_volume = add_volume_indicators(df.copy())
    volume_cols = ['volume_sma_20', 'volume_roc_10', 'obv', 'vwap', 'mfi_14']
    missing_volume = [col for col in volume_cols if col not in df_volume.columns]
    if missing_volume:
        print(f"FAIL: Volume - missing: {missing_volume}")
        return False
    print(f"PASS: Volume indicators ({len(volume_cols)} columns)")
    
    # Test volatility indicators
    df_volatility = add_volatility_indicators(df.copy())
    volatility_cols = ['atr_14', 'hist_vol_20', 'bb_width_pct']
    missing_volatility = [col for col in volatility_cols if col not in df_volatility.columns]
    if missing_volatility:
        print(f"FAIL: Volatility - missing: {missing_volatility}")
        return False
    print(f"PASS: Volatility indicators ({len(volatility_cols)} columns)")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("TECHNICAL INDICATORS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Indicator Calculation", test_indicator_calculation),
        ("NaN Value Handling", test_no_nan_values),
        ("Normalization Range", test_normalization_range),
        ("Scaler Save/Load", test_scaler_save_load),
        ("Individual Indicator Groups", test_individual_indicator_groups),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
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
