"""
Test script for Hybrid BGRU Model.

This script validates the Hybrid BGRU implementation by:
1. Testing model architecture
2. Testing forward pass
3. Testing data preparation
4. Testing save/load functionality
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

from models.bgru_hybrid import (
    HybridBGRUModel,
    HybridBGRUNetwork,
    OHLCV_FEATURES,
    TECHNICAL_FEATURES,
    TEMPORAL_FEATURES,
    PRICE_ACTION_FEATURES,
    get_static_features
)


def test_network_architecture():
    """Test that network architecture matches specification."""
    print("\n" + "=" * 60)
    print("TEST: Network Architecture")
    print("=" * 60)
    
    # Create network with specified dimensions
    n_price_features = 5
    n_static_features = 50
    network = HybridBGRUNetwork(
        n_price_features=n_price_features,
        n_static_features=n_static_features,
        hidden_dim=128,
        dropout=0.3
    )
    
    # Check BGRU layers (using dynamic ModuleList)
    assert hasattr(network, 'gru_layers'), "Missing gru_layers"
    assert len(network.gru_layers) >= 2, "Should have at least 2 GRU layers"
    for gru in network.gru_layers:
        assert gru.bidirectional, "All GRU layers should be bidirectional"
    print("PASS: BGRU layers are bidirectional")
    
    # Check first GRU layer hidden dimension
    assert network.gru_layers[0].hidden_size == 128, f"gru1 hidden size should be 128, got {network.gru_layers[0].hidden_size}"
    print("PASS: BGRU first layer hidden dimension correct (128)")
    
    # Check static path layers
    assert hasattr(network, 'static_fc1'), "Missing static_fc1 layer"
    assert hasattr(network, 'static_fc2'), "Missing static_fc2 layer"
    assert network.static_fc1.in_features == n_static_features
    assert network.static_fc1.out_features == 64
    assert network.static_fc2.out_features == 32
    print("PASS: Static path layers correct (64, 32)")
    
    # Check fusion layers
    assert hasattr(network, 'fusion_fc1'), "Missing fusion_fc1 layer"
    assert hasattr(network, 'fusion_fc2'), "Missing fusion_fc2 layer"
    assert network.fusion_fc1.out_features == 64
    assert network.fusion_fc2.out_features == 32
    print("PASS: Fusion layers correct (->64->32)")
    
    # Check output layer
    assert hasattr(network, 'output_fc'), "Missing output_fc layer"
    assert network.output_fc.out_features == 1
    print("PASS: Output layer correct (1 unit)")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"INFO: Total parameters: {total_params:,}")
    
    return True


def test_forward_pass():
    """Test forward pass with sample data."""
    print("\n" + "=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)
    
    n_price_features = 5
    n_static_features = 50
    sequence_length = 60
    batch_size = 32
    
    network = HybridBGRUNetwork(
        n_price_features=n_price_features,
        n_static_features=n_static_features
    )
    
    # Create sample inputs
    seq_input = torch.randn(batch_size, sequence_length, n_price_features)
    static_input = torch.randn(batch_size, n_static_features)
    
    # Forward pass
    output = network(seq_input, static_input)
    
    # Check output shape
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print(f"PASS: Output shape correct: {output.shape}")
    
    # For regression, output is unbounded (no sigmoid activation)
    # Just verify output is a valid tensor (not NaN or Inf)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    print(f"PASS: Output is valid tensor (no NaN/Inf), range: [{output.min():.4f}, {output.max():.4f}]")
    
    return True


def test_model_initialization():
    """Test HybridBGRUModel initialization."""
    print("\n" + "=" * 60)
    print("TEST: Model Initialization")
    print("=" * 60)
    
    model = HybridBGRUModel(
        sequence_length=60,
        n_price_features=5,
        n_static_features=50
    )
    
    assert model.sequence_length == 60
    assert model.n_price_features == 5
    assert model.n_static_features == 50
    print("PASS: Model parameters initialized correctly")
    
    # Check feature columns
    assert len(model.ohlcv_columns) == 5
    assert model.ohlcv_columns == OHLCV_FEATURES
    print(f"PASS: OHLCV columns: {model.ohlcv_columns}")
    
    static_features = get_static_features()
    expected_count = len(TECHNICAL_FEATURES) + len(TEMPORAL_FEATURES) + len(PRICE_ACTION_FEATURES)
    assert len(static_features) == expected_count
    print(f"PASS: Static features count: {len(static_features)}")
    
    return True


def test_data_preparation():
    """Test data preparation with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: Data Preparation")
    print("=" * 60)
    
    # Create synthetic data
    n_samples = 200
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
    
    # Add some static features
    for feat in TECHNICAL_FEATURES[:5]:
        data[feat] = np.random.randn(n_samples)
    for feat in TEMPORAL_FEATURES[:3]:
        data[feat] = np.random.randn(n_samples)
    for feat in PRICE_ACTION_FEATURES[:2]:
        data[feat] = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(data)
    
    model = HybridBGRUModel(sequence_length=60)
    
    X_seq, X_static, y = model.prepare_data(df)
    
    # Check shapes
    expected_samples = n_samples - 60
    assert X_seq.shape[0] == expected_samples, f"Expected {expected_samples} samples, got {X_seq.shape[0]}"
    assert X_seq.shape[1] == 60, f"Expected sequence length 60, got {X_seq.shape[1]}"
    assert X_seq.shape[2] == 5, f"Expected 5 price features, got {X_seq.shape[2]}"
    print(f"PASS: Sequential data shape: {X_seq.shape}")
    
    assert X_static.shape[0] == expected_samples
    print(f"PASS: Static data shape: {X_static.shape}")
    
    assert len(y) == expected_samples
    print(f"PASS: Target shape: {y.shape}")
    
    # Check for NaN values
    assert not np.isnan(X_seq).any(), "X_seq contains NaN values"
    assert not np.isnan(X_static).any(), "X_static contains NaN values"
    assert not np.isnan(y).any(), "y contains NaN values"
    print("PASS: No NaN values in prepared data")
    
    return True


def test_save_load():
    """Test model save and load functionality."""
    print("\n" + "=" * 60)
    print("TEST: Save/Load Model")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, 'test_model.pt')
    
    try:
        # Create and build model
        model1 = HybridBGRUModel(
            sequence_length=60,
            n_price_features=5,
            n_static_features=50
        )
        model1.build_model()
        
        # Save model
        model1.save_model(checkpoint_path)
        assert os.path.exists(checkpoint_path), "Checkpoint file not created"
        print(f"PASS: Model saved to {checkpoint_path}")
        
        # Load model
        model2 = HybridBGRUModel()
        model2.load_model(checkpoint_path)
        
        assert model2.sequence_length == model1.sequence_length
        assert model2.n_price_features == model1.n_price_features
        assert model2.n_static_features == model1.n_static_features
        print("PASS: Model parameters loaded correctly")
        
        # Check weights are the same
        for (name1, param1), (name2, param2) in zip(
            model1.model.named_parameters(),
            model2.model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2), f"Parameter {name1} differs after load"
        print("PASS: Model weights loaded correctly")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("HYBRID BGRU MODEL TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Network Architecture", test_network_architecture),
        ("Forward Pass", test_forward_pass),
        ("Model Initialization", test_model_initialization),
        ("Data Preparation", test_data_preparation),
        ("Save/Load Model", test_save_load),
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
