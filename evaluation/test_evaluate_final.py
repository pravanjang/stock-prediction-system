"""
Test script for Final Evaluation & Backtesting.

This script validates the evaluate_final.py implementation by:
1. Testing metrics calculation functions
2. Testing backtesting functions
3. Testing report generation
4. Testing visualization functions
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate_final import (
    calculate_all_metrics,
    plot_confusion_matrix_heatmap,
    plot_roc_curve,
    plot_prediction_confidence_distribution,
    plot_equity_curve,
    plot_drawdown,
    plot_hourly_trades,
    plot_monthly_returns,
    backtest_trading_strategy,
    generate_comprehensive_report,
    _empty_backtest_results,
)


def create_synthetic_predictions(n_samples: int = 500) -> tuple:
    """Create synthetic prediction data for testing."""
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.clip(np.random.randn(n_samples) * 0.2 + 0.5, 0.1, 0.9)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return y_true, y_pred, y_pred_proba


def create_synthetic_ohlcv_data(n_samples: int = 200) -> pd.DataFrame:
    """Create synthetic OHLCV data for backtesting."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01 09:15', periods=n_samples, freq='5min')
    
    # Simulate price movement
    base_price = 45000
    price_changes = np.random.randn(n_samples) * 50
    close = base_price + np.cumsum(price_changes)
    
    data = {
        'open': close - np.random.rand(n_samples) * 20,
        'high': close + np.random.rand(n_samples) * 30,
        'low': close - np.random.rand(n_samples) * 30,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_calculate_all_metrics():
    """Test metrics calculation function."""
    print("\n" + "=" * 60)
    print("TEST: Calculate All Metrics")
    print("=" * 60)
    
    y_true, y_pred, y_pred_proba = create_synthetic_predictions(500)
    
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
    
    # Check all required metrics are present
    required_keys = [
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
        'true_negatives', 'false_positives', 'false_negatives', 'true_positives',
        'specificity', 'class_0_precision', 'class_0_recall', 'class_1_precision',
        'class_1_recall', 'prediction_distribution', 'ground_truth_distribution',
        'total_samples', 'correct_predictions', 'incorrect_predictions'
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
    print("PASS: All required metrics present")
    
    # Check metric ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1
    print("PASS: Metric values within valid ranges")
    
    # Check sample counts
    assert metrics['total_samples'] == 500
    assert metrics['correct_predictions'] + metrics['incorrect_predictions'] == 500
    print("PASS: Sample counts correct")
    
    return True


def test_backtest_trading_strategy():
    """Test backtesting function."""
    print("\n" + "=" * 60)
    print("TEST: Backtest Trading Strategy")
    print("=" * 60)
    
    df = create_synthetic_ohlcv_data(200)
    
    # Create predictions (aligned with sequence_length=60)
    n_predictions = len(df) - 60
    predictions = np.random.randint(0, 2, n_predictions)
    proba = np.random.rand(n_predictions)
    
    results = backtest_trading_strategy(
        df=df,
        predictions=predictions,
        proba=proba,
        sequence_length=60,
        lot_size=25,
        transaction_cost=0.0003
    )
    
    # Check all required metrics are present
    required_keys = [
        'total_trades', 'winning_trades', 'losing_trades',
        'long_trades', 'short_trades', 'total_return', 'annualized_return',
        'total_pnl', 'win_rate', 'avg_profit_per_winning_trade',
        'avg_loss_per_losing_trade', 'profit_factor', 'max_drawdown',
        'max_drawdown_pct', 'sharpe_ratio', 'sortino_ratio',
        'avg_trade_pnl', 'avg_trade_return', 'std_trade_return',
        'best_trade', 'worst_trade', 'avg_holding_time',
        'equity_curve', 'drawdown_curve', 'trades_df'
    ]
    
    for key in required_keys:
        assert key in results, f"Missing backtest result: {key}"
    print("PASS: All required backtest metrics present")
    
    # Check trade counts
    assert results['total_trades'] > 0
    assert results['total_trades'] == results['winning_trades'] + results['losing_trades']
    assert results['total_trades'] == results['long_trades'] + results['short_trades']
    print(f"PASS: Trade counts consistent ({results['total_trades']} trades)")
    
    # Check win rate range
    assert 0 <= results['win_rate'] <= 100
    print(f"PASS: Win rate within valid range ({results['win_rate']:.1f}%)")
    
    # Check equity curve length
    assert len(results['equity_curve']) == results['total_trades'] + 1
    print("PASS: Equity curve has correct length")
    
    # Check trades dataframe
    assert isinstance(results['trades_df'], pd.DataFrame)
    assert len(results['trades_df']) == results['total_trades']
    print("PASS: Trades DataFrame has correct length")
    
    return True


def test_empty_backtest_results():
    """Test empty backtest results function."""
    print("\n" + "=" * 60)
    print("TEST: Empty Backtest Results")
    print("=" * 60)
    
    results = _empty_backtest_results()
    
    assert results['total_trades'] == 0
    assert results['win_rate'] == 0.0
    assert results['total_pnl'] == 0.0
    assert isinstance(results['trades_df'], pd.DataFrame)
    print("PASS: Empty results have correct default values")
    
    return True


def test_plot_functions():
    """Test plotting functions."""
    print("\n" + "=" * 60)
    print("TEST: Plotting Functions")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        y_true, y_pred, y_pred_proba = create_synthetic_predictions(100)
        
        # Test confusion matrix
        cm_path = os.path.join(temp_dir, 'confusion_matrix.png')
        plot_confusion_matrix_heatmap(y_true, y_pred, cm_path)
        assert os.path.exists(cm_path)
        print("PASS: Confusion matrix plot generated")
        
        # Test ROC curve
        roc_path = os.path.join(temp_dir, 'roc_curve.png')
        plot_roc_curve(y_true, y_pred_proba, roc_path)
        assert os.path.exists(roc_path)
        print("PASS: ROC curve plot generated")
        
        # Test confidence distribution
        conf_path = os.path.join(temp_dir, 'confidence.png')
        plot_prediction_confidence_distribution(y_pred_proba, conf_path, y_true)
        assert os.path.exists(conf_path)
        print("PASS: Confidence distribution plot generated")
        
        # Test equity curve
        equity_curve = [0, 100, 150, 80, 200, 250]
        equity_path = os.path.join(temp_dir, 'equity.png')
        plot_equity_curve(equity_curve, equity_path)
        assert os.path.exists(equity_path)
        print("PASS: Equity curve plot generated")
        
        # Test drawdown chart
        drawdown_curve = [0, -2, -5, -3, -8, -4]
        dd_path = os.path.join(temp_dir, 'drawdown.png')
        plot_drawdown(drawdown_curve, dd_path)
        assert os.path.exists(dd_path)
        print("PASS: Drawdown chart generated")
        
        # Test hourly trades
        trades_df = pd.DataFrame({
            'hour': [9, 10, 11, 12, 9, 10, 11, 12],
            'won': [True, False, True, True, False, True, False, True],
            'net_return': [0.01, -0.02, 0.015, 0.02, -0.01, 0.01, -0.015, 0.02]
        })
        hourly_path = os.path.join(temp_dir, 'hourly.png')
        plot_hourly_trades(trades_df, hourly_path)
        assert os.path.exists(hourly_path)
        print("PASS: Hourly trades plot generated")
        
        # Test monthly returns
        trades_df_monthly = pd.DataFrame({
            'year': [2024, 2024, 2024, 2024],
            'month': [1, 1, 2, 2],
            'net_return': [0.01, -0.02, 0.015, 0.02]
        })
        monthly_path = os.path.join(temp_dir, 'monthly.png')
        plot_monthly_returns(trades_df_monthly, monthly_path)
        assert os.path.exists(monthly_path)
        print("PASS: Monthly returns heatmap generated")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_generate_reports():
    """Test report generation."""
    print("\n" + "=" * 60)
    print("TEST: Report Generation")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample metrics
        metrics = {
            'accuracy': 0.62,
            'precision': 0.60,
            'recall': 0.65,
            'f1_score': 0.62,
            'roc_auc': 0.68,
            'class_0_precision': 0.58,
            'class_0_recall': 0.55,
            'class_0_f1': 0.56,
            'class_1_precision': 0.60,
            'class_1_recall': 0.65,
            'class_1_f1': 0.62,
        }
        
        backtest_results = {
            'total_trades': 1000,
            'winning_trades': 580,
            'losing_trades': 420,
            'long_trades': 550,
            'short_trades': 450,
            'total_return': 25.5,
            'annualized_return': 35.2,
            'total_pnl': 125000.0,
            'win_rate': 58.0,
            'avg_profit_per_winning_trade': 500.0,
            'avg_loss_per_losing_trade': -350.0,
            'profit_factor': 1.65,
            'max_drawdown': 15000.0,
            'max_drawdown_pct': 8.5,
            'sharpe_ratio': 1.45,
            'sortino_ratio': 2.1,
            'avg_trade_pnl': 125.0,
            'avg_trade_return': 0.025,
            'std_trade_return': 0.5,
            'best_trade': 5000.0,
            'worst_trade': -3500.0,
            'avg_holding_time': 1,
        }
        
        html_path, txt_path = generate_comprehensive_report(
            metrics, backtest_results, temp_dir
        )
        
        assert os.path.exists(html_path)
        assert os.path.exists(txt_path)
        print("PASS: Both HTML and text reports generated")
        
        # Check HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        assert 'Final Model Evaluation Report' in html_content
        assert 'Classification Metrics' in html_content
        assert 'Backtesting Results' in html_content
        print("PASS: HTML report has expected content")
        
        # Check text content
        with open(txt_path, 'r') as f:
            txt_content = f.read()
        assert 'FINAL MODEL EVALUATION' in txt_content
        assert 'Classification Metrics' in txt_content
        assert 'Backtesting Results' in txt_content
        print("PASS: Text report has expected content")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("EVALUATE_FINAL.PY TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Calculate All Metrics", test_calculate_all_metrics),
        ("Backtest Trading Strategy", test_backtest_trading_strategy),
        ("Empty Backtest Results", test_empty_backtest_results),
        ("Plotting Functions", test_plot_functions),
        ("Report Generation", test_generate_reports),
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
