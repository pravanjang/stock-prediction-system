"""
Price Action Features Module for Stock Prediction System

This module extracts candlestick patterns and price momentum features:
- Candlestick patterns (bullish/bearish engulfing, doji, hammer, shooting star, inside bar)
- Price momentum (returns at multiple lags)
- Range metrics (range %, close position, gap, body-to-range ratio)
- Volume-price relationships (volume-price trend, price-volume correlation)
- Market microstructure features (volume imbalance, price impact, volatility regime)

All features are designed for BankNifty intraday/daily prediction.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Candlestick Pattern Detection Functions
# ============================================================================

def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect bullish engulfing pattern.
    
    Criteria:
    - Previous candle is bearish (close < open)
    - Current candle is bullish (close > open)
    - Current body completely engulfs previous body
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series of binary indicators (1 = pattern detected, 0 = not detected)
    """
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)
    curr_bullish = df['close'] > df['open']
    
    # Current open below previous close, current close above previous open
    engulfs = (df['open'] <= df['close'].shift(1)) & (df['close'] >= df['open'].shift(1))
    
    return (prev_bearish & curr_bullish & engulfs).astype(int)


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect bearish engulfing pattern.
    
    Criteria:
    - Previous candle is bullish (close > open)
    - Current candle is bearish (close < open)
    - Current body completely engulfs previous body
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series of binary indicators (1 = pattern detected, 0 = not detected)
    """
    prev_bullish = df['close'].shift(1) > df['open'].shift(1)
    curr_bearish = df['close'] < df['open']
    
    # Current open above previous close, current close below previous open
    engulfs = (df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1))
    
    return (prev_bullish & curr_bearish & engulfs).astype(int)


def detect_doji(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """
    Detect doji pattern (indecision candle).
    
    Criteria:
    - Body size is less than threshold (0.1%) of the candle range
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum body-to-range ratio to qualify as doji
    
    Returns:
        Series of binary indicators
    """
    body_size = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    
    # Avoid division by zero
    candle_range = candle_range.replace(0, np.nan)
    body_to_range = body_size / candle_range
    
    return (body_to_range < threshold).fillna(False).astype(int)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Detect hammer pattern (potential bullish reversal).
    
    Criteria:
    - Lower shadow is at least 2x the body size
    - Upper shadow is small (less than body size)
    - Body is in upper third of candle range
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series of binary indicators
    """
    body_size = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    
    # Calculate shadows
    body_top = df[['open', 'close']].max(axis=1)
    body_bottom = df[['open', 'close']].min(axis=1)
    upper_shadow = df['high'] - body_top
    lower_shadow = body_bottom - df['low']
    
    # Avoid division by zero
    body_size_safe = body_size.replace(0, 0.0001)
    
    # Hammer criteria
    long_lower_shadow = lower_shadow >= 2 * body_size_safe
    small_upper_shadow = upper_shadow <= body_size_safe
    
    # Body in upper portion of range
    body_position = (body_bottom - df['low']) / candle_range.replace(0, np.nan)
    body_in_upper = body_position >= 0.6
    
    return (long_lower_shadow & small_upper_shadow & body_in_upper.fillna(False)).astype(int)


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Detect shooting star pattern (potential bearish reversal).
    
    Criteria:
    - Upper shadow is at least 2x the body size
    - Lower shadow is small (less than body size)
    - Body is in lower third of candle range
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series of binary indicators
    """
    body_size = abs(df['close'] - df['open'])
    candle_range = df['high'] - df['low']
    
    # Calculate shadows
    body_top = df[['open', 'close']].max(axis=1)
    body_bottom = df[['open', 'close']].min(axis=1)
    upper_shadow = df['high'] - body_top
    lower_shadow = body_bottom - df['low']
    
    # Avoid division by zero
    body_size_safe = body_size.replace(0, 0.0001)
    
    # Shooting star criteria
    long_upper_shadow = upper_shadow >= 2 * body_size_safe
    small_lower_shadow = lower_shadow <= body_size_safe
    
    # Body in lower portion of range
    body_position = (body_top - df['low']) / candle_range.replace(0, np.nan)
    body_in_lower = body_position <= 0.4
    
    return (long_upper_shadow & small_lower_shadow & body_in_lower.fillna(False)).astype(int)


def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Detect inside bar pattern (consolidation/compression).
    
    Criteria:
    - Current high is below previous high
    - Current low is above previous low
    - Current bar is completely within previous bar's range
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Series of binary indicators
    """
    high_below_prev = df['high'] < df['high'].shift(1)
    low_above_prev = df['low'] > df['low'].shift(1)
    
    return (high_below_prev & low_above_prev).astype(int)


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all candlestick pattern features to dataframe.
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with pattern columns added
    """
    logger.info("Adding candlestick pattern features...")
    
    df['is_bullish_engulf'] = detect_bullish_engulfing(df)
    df['is_bearish_engulf'] = detect_bearish_engulfing(df)
    df['is_doji'] = detect_doji(df)
    df['is_hammer'] = detect_hammer(df)
    df['is_shooting_star'] = detect_shooting_star(df)
    df['is_inside_bar'] = detect_inside_bar(df)
    
    pattern_cols = ['is_bullish_engulf', 'is_bearish_engulf', 'is_doji', 
                    'is_hammer', 'is_shooting_star', 'is_inside_bar']
    
    # Log pattern occurrence counts
    for col in pattern_cols:
        count = df[col].sum()
        pct = 100 * count / len(df)
        logger.info(f"  {col}: {count} occurrences ({pct:.2f}%)")
    
    return df


# ============================================================================
# Price Momentum Features
# ============================================================================

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return features at multiple lag periods.
    
    Features:
    - return_1: 1-period return (previous close to current close)
    - return_5: 5-period return
    - return_15: 15-period return
    - return_30: 30-period return
    - return_60: 60-period return
    
    Args:
        df: DataFrame with close price
    
    Returns:
        DataFrame with return columns added
    """
    logger.info("Adding return features...")
    
    lags = [1, 5, 15, 30, 60]
    
    for lag in lags:
        col_name = f'return_{lag}'
        df[col_name] = (df['close'] - df['close'].shift(lag)) / df['close'].shift(lag)
        
        # Log statistics
        mean_return = df[col_name].mean()
        std_return = df[col_name].std()
        logger.info(f"  {col_name}: mean={mean_return:.6f}, std={std_return:.6f}")
    
    return df


# ============================================================================
# Range Metrics
# ============================================================================

def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add range-based features.
    
    Features:
    - range_pct: (high - low) / close (percentage range)
    - close_position: (close - low) / (high - low) [0 to 1 scale]
    - gap_at_open: (open - prev_close) / prev_close
    - body_to_range: abs(close - open) / (high - low)
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with range metric columns added
    """
    logger.info("Adding range features...")
    
    # Range percentage
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Close position within range [0=at low, 1=at high]
    candle_range = df['high'] - df['low']
    candle_range_safe = candle_range.replace(0, np.nan)
    df['close_position'] = (df['close'] - df['low']) / candle_range_safe
    df['close_position'] = df['close_position'].fillna(0.5)  # Default to middle for doji
    
    # Gap at open
    df['gap_at_open'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Body to range ratio
    body_size = abs(df['close'] - df['open'])
    df['body_to_range'] = body_size / candle_range_safe
    df['body_to_range'] = df['body_to_range'].fillna(0)  # Default to 0 for flat candles
    
    # Clip extreme values
    df['body_to_range'] = df['body_to_range'].clip(0, 1)
    df['close_position'] = df['close_position'].clip(0, 1)
    
    # Log statistics
    for col in ['range_pct', 'close_position', 'gap_at_open', 'body_to_range']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        logger.info(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    return df


# ============================================================================
# Volume-Price Relationship Features
# ============================================================================

def add_volume_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-price relationship features.
    
    Features:
    - volume_price_trend: volume * price_change / prev_close
    - price_volume_correlation: rolling correlation of price change and volume
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with volume-price columns added
    """
    logger.info("Adding volume-price relationship features...")
    
    # Price change
    price_change = df['close'] - df['close'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Volume Price Trend (VPT component)
    df['volume_price_trend'] = df['volume'] * (price_change / prev_close)
    
    # Price-Volume Correlation (rolling 10-period)
    # Calculate rolling correlation between price changes and volume
    returns = df['close'].pct_change()
    df['price_volume_correlation'] = returns.rolling(window=10).corr(df['volume'])
    
    # Fill NaN values
    df['volume_price_trend'] = df['volume_price_trend'].fillna(0)
    df['price_volume_correlation'] = df['price_volume_correlation'].fillna(0)
    
    # Log statistics
    for col in ['volume_price_trend', 'price_volume_correlation']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        logger.info(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    return df


# ============================================================================
# Market Microstructure Features
# ============================================================================

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market microstructure features.
    
    Features:
    - volume_imbalance: (volume - volume_SMA20) / volume_SMA20
    - price_impact: abs(returns) / (volume / volume_SMA20)
    - volatility_regime: classify ATR into low/medium/high (0/1/2)
    - volume_trend: (volume_SMA5 - volume_SMA20) / volume_SMA20
    - tick_direction: sign(close - prev_close) [-1, 0, 1]
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with microstructure columns added
    """
    logger.info("Adding market microstructure features...")
    
    # Volume SMA calculations
    volume_sma20 = df['volume'].rolling(window=20).mean()
    volume_sma5 = df['volume'].rolling(window=5).mean()
    
    # Avoid division by zero
    volume_sma20_safe = volume_sma20.replace(0, np.nan)
    
    # Volume imbalance
    df['volume_imbalance'] = (df['volume'] - volume_sma20) / volume_sma20_safe
    df['volume_imbalance'] = df['volume_imbalance'].fillna(0)
    
    # Price impact (how much price moves per unit of relative volume)
    returns = df['close'].pct_change().abs()
    relative_volume = df['volume'] / volume_sma20_safe
    relative_volume_safe = relative_volume.replace(0, np.nan)
    df['price_impact'] = returns / relative_volume_safe
    df['price_impact'] = df['price_impact'].fillna(0)
    # Clip extreme values
    df['price_impact'] = df['price_impact'].clip(0, df['price_impact'].quantile(0.99))
    
    # Volatility regime (based on rolling ATR percentile)
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Classify volatility regime using quartiles
    atr_25 = atr.rolling(window=100).quantile(0.25)
    atr_75 = atr.rolling(window=100).quantile(0.75)
    
    df['volatility_regime'] = 1  # Default to medium
    df.loc[atr <= atr_25, 'volatility_regime'] = 0  # Low volatility
    df.loc[atr >= atr_75, 'volatility_regime'] = 2  # High volatility
    df['volatility_regime'] = df['volatility_regime'].fillna(1)
    
    # Volume trend
    df['volume_trend'] = (volume_sma5 - volume_sma20) / volume_sma20_safe
    df['volume_trend'] = df['volume_trend'].fillna(0)
    
    # Tick direction
    price_change = df['close'] - df['close'].shift(1)
    df['tick_direction'] = np.sign(price_change)
    df['tick_direction'] = df['tick_direction'].fillna(0).astype(int)
    
    # Log statistics
    for col in ['volume_imbalance', 'price_impact', 'volatility_regime', 'volume_trend', 'tick_direction']:
        if col == 'volatility_regime':
            counts = df[col].value_counts()
            logger.info(f"  {col} distribution: {counts.to_dict()}")
        elif col == 'tick_direction':
            counts = df[col].value_counts()
            logger.info(f"  {col} distribution: {counts.to_dict()}")
        else:
            mean_val = df[col].mean()
            std_val = df[col].std()
            logger.info(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    return df


# ============================================================================
# Main Feature Addition Function
# ============================================================================

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all candlestick patterns and price momentum features.
    
    This is the main function that combines all price action features:
    1. Candlestick patterns (6 patterns)
    2. Return features (5 lags)
    3. Range metrics (4 features)
    4. Volume-price relationships (2 features)
    5. Market microstructure features (5 features)
    
    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
    
    Returns:
        DataFrame with 22+ new price action columns
    """
    logger.info("=" * 60)
    logger.info("Adding Price Action Features")
    logger.info("=" * 60)
    
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Store original columns to track new additions
    original_columns = list(df.columns)
    
    # Add all feature groups
    df = add_candlestick_patterns(df)
    df = add_return_features(df)
    df = add_range_features(df)
    df = add_volume_price_features(df)
    df = add_microstructure_features(df)
    
    # Get list of new columns
    new_columns = [col for col in df.columns if col not in original_columns]
    logger.info(f"\nAdded {len(new_columns)} price action columns: {new_columns}")
    
    # Check for NaN values
    nan_counts = df[new_columns].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        logger.warning(f"NaN values found in price action features:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  {col}: {count} NaNs")
        
        # Forward fill remaining NaNs
        logger.info("Forward-filling remaining NaN values...")
        df[new_columns] = df[new_columns].ffill()
        
        # Backward fill any remaining (first rows)
        df[new_columns] = df[new_columns].bfill()
        
        remaining_nans = df[new_columns].isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"Still have {remaining_nans} NaN values after filling")
            df[new_columns] = df[new_columns].fillna(0)
    else:
        logger.info("No NaN values in price action features")
    
    logger.info("-" * 60)
    logger.info(f"Final DataFrame shape: {df.shape}")
    logger.info("-" * 60)
    
    return df


def get_price_action_columns() -> list:
    """
    Returns list of all price action column names.
    
    Returns:
        List of price action column names
    """
    columns = [
        # Candlestick patterns
        'is_bullish_engulf', 'is_bearish_engulf', 'is_doji',
        'is_hammer', 'is_shooting_star', 'is_inside_bar',
        # Return features
        'return_1', 'return_5', 'return_15', 'return_30', 'return_60',
        # Range metrics
        'range_pct', 'close_position', 'gap_at_open', 'body_to_range',
        # Volume-price relationships
        'volume_price_trend', 'price_volume_correlation',
        # Market microstructure
        'volume_imbalance', 'price_impact', 'volatility_regime',
        'volume_trend', 'tick_direction',
    ]
    return columns


def main():
    """Main function for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Add price action features to OHLCV data'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path with OHLCV data'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path for data with price action features'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    import os
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)[:10]}...")
    
    # Add price action features
    df_with_features = add_price_action_features(df)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save output
    df_with_features.to_csv(args.output, index=False)
    logger.info(f"Saved data with price action features to {args.output}")
    
    # Print summary
    price_action_cols = get_price_action_columns()
    existing_cols = [col for col in price_action_cols if col in df_with_features.columns]
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input rows: {len(df)}")
    logger.info(f"Output rows: {len(df_with_features)}")
    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"Final columns: {len(df_with_features.columns)}")
    logger.info(f"Price action columns added: {len(existing_cols)}")
    logger.info(f"NaN values in output: {df_with_features.isna().sum().sum()}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
