"""
Test module for price action features.

Tests cover:
1. Candlestick pattern detection
2. Return feature calculations
3. Range metrics
4. Volume-price relationships
5. Market microstructure features
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from price_action import (
    detect_bullish_engulfing,
    detect_bearish_engulfing,
    detect_doji,
    detect_hammer,
    detect_shooting_star,
    detect_inside_bar,
    add_candlestick_patterns,
    add_return_features,
    add_range_features,
    add_volume_price_features,
    add_microstructure_features,
    add_price_action_features,
    get_price_action_columns,
)


class TestBullishEngulfing(unittest.TestCase):
    """Test bullish engulfing pattern detection."""
    
    def test_bullish_engulfing_detected(self):
        """Detect valid bullish engulfing pattern."""
        # Previous candle: bearish (open=100, close=95)
        # Current candle: bullish engulfs (open=94, close=101)
        df = pd.DataFrame({
            'open': [100, 94],
            'high': [102, 102],
            'low': [94, 93],
            'close': [95, 101],
        })
        result = detect_bullish_engulfing(df)
        self.assertEqual(result.iloc[1], 1)
        
    def test_bullish_engulfing_not_detected(self):
        """Don't detect when criteria not met."""
        # Both candles bullish - not engulfing
        df = pd.DataFrame({
            'open': [100, 102],
            'high': [105, 108],
            'low': [99, 101],
            'close': [104, 107],
        })
        result = detect_bullish_engulfing(df)
        self.assertEqual(result.iloc[1], 0)


class TestBearishEngulfing(unittest.TestCase):
    """Test bearish engulfing pattern detection."""
    
    def test_bearish_engulfing_detected(self):
        """Detect valid bearish engulfing pattern."""
        # Previous candle: bullish (open=95, close=100)
        # Current candle: bearish engulfs (open=101, close=94)
        df = pd.DataFrame({
            'open': [95, 101],
            'high': [101, 102],
            'low': [94, 93],
            'close': [100, 94],
        })
        result = detect_bearish_engulfing(df)
        self.assertEqual(result.iloc[1], 1)
        
    def test_bearish_engulfing_not_detected(self):
        """Don't detect when criteria not met."""
        # Both candles bearish - not engulfing
        df = pd.DataFrame({
            'open': [105, 102],
            'high': [106, 103],
            'low': [100, 98],
            'close': [101, 99],
        })
        result = detect_bearish_engulfing(df)
        self.assertEqual(result.iloc[1], 0)


class TestDoji(unittest.TestCase):
    """Test doji pattern detection."""
    
    def test_doji_detected(self):
        """Detect doji when body is very small."""
        # Open and close nearly equal, but high-low range is large
        # Body = 0.005 (100.005 - 100), Range = 10 (105 - 95)
        # Body/Range = 0.0005 which is < 0.001 threshold
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [100.005],  # Very small body (0.005)
        })
        result = detect_doji(df)
        self.assertEqual(result.iloc[0], 1)
        
    def test_doji_not_detected(self):
        """Don't detect doji when body is large."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [104.0],  # Large body
        })
        result = detect_doji(df)
        self.assertEqual(result.iloc[0], 0)


class TestHammer(unittest.TestCase):
    """Test hammer pattern detection."""
    
    def test_hammer_detected(self):
        """Detect hammer pattern with long lower shadow."""
        # Long lower shadow, small upper shadow, body at top
        df = pd.DataFrame({
            'open': [98.0],
            'high': [100.0],  # Small upper shadow
            'low': [90.0],    # Long lower shadow
            'close': [99.0],  # Close near high (bullish hammer)
        })
        result = detect_hammer(df)
        self.assertEqual(result.iloc[0], 1)
        
    def test_hammer_not_detected(self):
        """Don't detect when lower shadow is short."""
        df = pd.DataFrame({
            'open': [95.0],
            'high': [100.0],
            'low': [94.0],   # Short lower shadow
            'close': [99.0],
        })
        result = detect_hammer(df)
        self.assertEqual(result.iloc[0], 0)


class TestShootingStar(unittest.TestCase):
    """Test shooting star pattern detection."""
    
    def test_shooting_star_detected(self):
        """Detect shooting star with long upper shadow."""
        # Long upper shadow, small lower shadow, body at bottom
        df = pd.DataFrame({
            'open': [92.0],
            'high': [100.0],  # Long upper shadow
            'low': [90.0],    # Small lower shadow
            'close': [91.0],  # Close near low
        })
        result = detect_shooting_star(df)
        self.assertEqual(result.iloc[0], 1)


class TestInsideBar(unittest.TestCase):
    """Test inside bar pattern detection."""
    
    def test_inside_bar_detected(self):
        """Detect inside bar when current bar within previous range."""
        df = pd.DataFrame({
            'open': [100, 97],
            'high': [105, 99],   # Current high below previous
            'low': [95, 96],     # Current low above previous
            'close': [102, 98],
        })
        result = detect_inside_bar(df)
        self.assertEqual(result.iloc[1], 1)
        
    def test_inside_bar_not_detected(self):
        """Don't detect when bar breaks previous range."""
        df = pd.DataFrame({
            'open': [100, 97],
            'high': [105, 108],  # Current high above previous
            'low': [95, 94],
            'close': [102, 106],
        })
        result = detect_inside_bar(df)
        self.assertEqual(result.iloc[1], 0)


class TestReturnFeatures(unittest.TestCase):
    """Test return calculations."""
    
    def test_return_1_calculation(self):
        """Test 1-period return calculation."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [102, 104, 103],
            'volume': [1000, 1100, 1050],
        })
        result = add_return_features(df.copy())
        
        # return_1 for row 1 = (104 - 102) / 102
        expected = (104 - 102) / 102
        self.assertAlmostEqual(result['return_1'].iloc[1], expected, places=6)
        
    def test_return_columns_exist(self):
        """Test that all return columns are created."""
        df = pd.DataFrame({
            'open': list(range(100, 200)),
            'high': list(range(105, 205)),
            'low': list(range(95, 195)),
            'close': list(range(102, 202)),
            'volume': [1000] * 100,
        })
        result = add_return_features(df.copy())
        
        expected_cols = ['return_1', 'return_5', 'return_15', 'return_30', 'return_60']
        for col in expected_cols:
            self.assertIn(col, result.columns)


class TestRangeFeatures(unittest.TestCase):
    """Test range metric calculations."""
    
    def test_range_pct_calculation(self):
        """Test range percentage calculation."""
        df = pd.DataFrame({
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000],
        })
        result = add_range_features(df.copy())
        
        # range_pct = (110 - 90) / 105
        expected = 20 / 105
        self.assertAlmostEqual(result['range_pct'].iloc[0], expected, places=6)
        
    def test_close_position_calculation(self):
        """Test close position within range."""
        df = pd.DataFrame({
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [100],  # Middle of range
            'volume': [1000],
        })
        result = add_range_features(df.copy())
        
        # close_position = (100 - 90) / (110 - 90) = 0.5
        self.assertAlmostEqual(result['close_position'].iloc[0], 0.5, places=6)
        
    def test_body_to_range_calculation(self):
        """Test body to range ratio."""
        df = pd.DataFrame({
            'open': [95],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000],
        })
        result = add_range_features(df.copy())
        
        # body_to_range = abs(105 - 95) / (110 - 90) = 10 / 20 = 0.5
        self.assertAlmostEqual(result['body_to_range'].iloc[0], 0.5, places=6)


class TestVolumePriceFeatures(unittest.TestCase):
    """Test volume-price relationship features."""
    
    def test_volume_price_trend(self):
        """Test volume-price trend calculation."""
        df = pd.DataFrame({
            'open': [100, 102],
            'high': [105, 107],
            'low': [99, 101],
            'close': [102, 105],  # 3-point rise
            'volume': [1000, 1500],
        })
        result = add_volume_price_features(df.copy())
        
        # For row 1: volume_price_trend = 1500 * (105 - 102) / 102
        expected = 1500 * (105 - 102) / 102
        self.assertAlmostEqual(result['volume_price_trend'].iloc[1], expected, places=4)
        
    def test_price_volume_correlation_exists(self):
        """Test that price-volume correlation column is created."""
        df = pd.DataFrame({
            'open': list(range(100, 120)),
            'high': list(range(105, 125)),
            'low': list(range(95, 115)),
            'close': list(range(102, 122)),
            'volume': [1000 + i*10 for i in range(20)],
        })
        result = add_volume_price_features(df.copy())
        self.assertIn('price_volume_correlation', result.columns)


class TestMicrostructureFeatures(unittest.TestCase):
    """Test market microstructure features."""
    
    def test_tick_direction(self):
        """Test tick direction calculation."""
        df = pd.DataFrame({
            'open': [100, 102, 101],
            'high': [105, 107, 106],
            'low': [99, 101, 100],
            'close': [102, 105, 103],  # Up, then down
            'volume': [1000, 1100, 1050],
        })
        result = add_microstructure_features(df.copy())
        
        self.assertEqual(result['tick_direction'].iloc[1], 1)   # Up
        self.assertEqual(result['tick_direction'].iloc[2], -1)  # Down
        
    def test_volatility_regime_values(self):
        """Test volatility regime is 0, 1, or 2."""
        np.random.seed(42)
        df = pd.DataFrame({
            'open': 100 + np.random.randn(200) * 2,
            'high': 105 + np.random.randn(200) * 2,
            'low': 95 + np.random.randn(200) * 2,
            'close': 100 + np.random.randn(200) * 2,
            'volume': 1000 + np.random.randn(200) * 100,
        })
        # Ensure high > open,close and low < open,close
        df['high'] = df[['open', 'close', 'high']].max(axis=1) + 1
        df['low'] = df[['open', 'close', 'low']].min(axis=1) - 1
        
        result = add_microstructure_features(df.copy())
        
        unique_regimes = result['volatility_regime'].unique()
        for regime in unique_regimes:
            self.assertIn(regime, [0, 1, 2])


class TestAddPriceActionFeatures(unittest.TestCase):
    """Test main price action feature function."""
    
    def test_all_columns_added(self):
        """Test that all price action columns are added."""
        np.random.seed(42)
        df = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 105 + np.random.randn(100) * 2,
            'low': 95 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.abs(1000 + np.random.randn(100) * 100),
        })
        # Ensure high > open,close and low < open,close
        df['high'] = df[['open', 'close', 'high']].max(axis=1) + 1
        df['low'] = df[['open', 'close', 'low']].min(axis=1) - 1
        
        result = add_price_action_features(df.copy())
        
        expected_cols = get_price_action_columns()
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")
            
    def test_no_nan_values(self):
        """Test that no NaN values remain after feature addition."""
        np.random.seed(42)
        df = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 105 + np.random.randn(100) * 2,
            'low': 95 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.abs(1000 + np.random.randn(100) * 100),
        })
        # Ensure high > open,close and low < open,close
        df['high'] = df[['open', 'close', 'high']].max(axis=1) + 1
        df['low'] = df[['open', 'close', 'low']].min(axis=1) - 1
        
        result = add_price_action_features(df.copy())
        
        price_action_cols = get_price_action_columns()
        existing_cols = [col for col in price_action_cols if col in result.columns]
        nan_count = result[existing_cols].isna().sum().sum()
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaN values")
        
    def test_required_columns_validation(self):
        """Test that missing columns raise an error."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            # Missing 'low', 'close', 'volume'
        })
        
        with self.assertRaises(ValueError):
            add_price_action_features(df)


class TestGetPriceActionColumns(unittest.TestCase):
    """Test column list function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        cols = get_price_action_columns()
        self.assertIsInstance(cols, list)
        
    def test_expected_count(self):
        """Test that expected number of columns are returned."""
        cols = get_price_action_columns()
        # 6 patterns + 5 returns + 4 range + 2 volume-price + 5 microstructure = 22
        self.assertGreaterEqual(len(cols), 20)


if __name__ == '__main__':
    unittest.main()
