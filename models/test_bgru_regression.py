"""
Test script for BGRU Regression tasks.

This script validates the BGRU model behavior in regression mode by:
1. Testing model and predictor prepare_sequences for regression targets
2. Testing forward pass for raw continuous outputs
3. Testing training loop with MSE loss for a few epochs
4. Testing save/load functionality and prediction outputs
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bgru_base import BGRUPredictor


def test_regression_prepare_and_forward():
    print("\n" + "=" * 60)
    print("TEST: Regression Prepare and Forward")
    print("=" * 60)

    # Create synthetic dataset with continuous next_close
    n_samples = 400
    dates = pd.date_range('2025-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
    })
    # Create next_close as continuous target
    df['next_close'] = df['close'].shift(-1).fillna(method='ffill')

    predictor = BGRUPredictor(feature_groups=['ohlcv'], regression=True)
    X, y = predictor.prepare_sequences(df, sequence_length=60)

    assert X.shape[0] == n_samples - 60
    assert y.shape[0] == n_samples - 60
    print("PASS: prepare_sequences returns X and y with correct shapes")

    # Build model and forward pass
    predictor.build_model()
    sample_X = torch.FloatTensor(X[:8]).to(predictor.device)
    outputs = predictor.model(sample_X)
    assert outputs.shape == (8, 1)
    print("PASS: Forward pass for regression returns shape (batch, 1)")

    return True


def test_regression_train_save_load():
    print("\n" + "=" * 60)
    print("TEST: Regression Training, Save and Load")
    print("=" * 60)

    # Create small synthetic dataset with continuous next_close
    n_samples = 400
    dates = pd.date_range('2025-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
    })
    df['next_close'] = df['close'].shift(-1).fillna(method='ffill')

    # Split into train/val/test by simple index split
    # Ensure split sizes produce enough sequences (> sequence_length)
    train_df = df.iloc[:250].copy()
    val_df = df.iloc[250:330].copy()
    test_df = df.iloc[330:].copy()

    predictor = BGRUPredictor(feature_groups=['ohlcv'], regression=True)
    # Build to ensure consistent behavior
    predictor.build_model()

    # Prepare sequences and assert there is enough data for training
    print(f"Data lengths -> Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    X_train, y_train = predictor.prepare_sequences(train_df, sequence_length=60)
    X_val, y_val = predictor.prepare_sequences(val_df, sequence_length=60)
    print(f"Sequenced -> Train: {len(X_train)} sequences, Val: {len(X_val)} sequences")
    assert len(X_train) > 0 and len(X_val) > 0, "Not enough sequences prepared for train/val in test splits"

    # Train for a few epochs to check regression training
    history = predictor.train(
        train_df=train_df,
        val_df=val_df,
        epochs=3,
        batch_size=32,
        lr=0.001,
        sequence_length=60,
        checkpoint_dir=os.path.join('models', 'checkpoints', 'test_regression')
    )

    assert 'train_rmse' in history and 'val_rmse' in history
    assert len(history['train_rmse']) >= 1
    print("PASS: Training returned RMSE metrics in history")

    # Save model, then load
    temp_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(temp_dir, 'reg_model.pt')
    predictor.save_model(ckpt_path)
    assert os.path.exists(ckpt_path)
    print(f"PASS: Model saved to {ckpt_path}")

    # Load into a new predictor instance
    p2 = BGRUPredictor(feature_groups=['ohlcv'], regression=True)
    p2.load_model(ckpt_path)
    assert p2.regression is True
    print("PASS: Model loaded and regression flag restored")

    # Predict using loaded model
    preds, probs = p2.predict(test_df, sequence_length=60, batch_size=32)
    assert len(preds) > 0
    print("PASS: Predictions generated for regression model")

    shutil.rmtree(temp_dir)
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("BGRU REGRESSION MODEL TEST SUITE")
    print("=" * 60)

    tests = [
        ("Prepare & Forward", test_regression_prepare_and_forward),
        ("Train / Save / Load", test_regression_train_save_load),
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

    print("\n" + "=" * 60)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print("=" * 60)
    return passed_count == total_count


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
