"""
Technical Indicators Module for Stock Prediction System

This module implements various technical indicators for stock price analysis:
- Momentum Indicators: RSI, MACD, Stochastic, ADX, Momentum
- Trend Indicators: EMA, SMA, Supertrend, Bollinger Bands, Parabolic SAR
- Volume Indicators: Volume SMA, Volume ROC, OBV, VWAP, MFI
- Volatility Indicators: ATR, Historical Volatility, BB Width

All indicators are normalized to [0, 1] range using MinMaxScaler.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
    """
    Calculate Supertrend indicator.
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        Series with Supertrend values
    """
    try:
        supertrend = ta.supertrend(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            length=period,
            multiplier=multiplier
        )
        # Return the trend direction (-1 or 1) converted to (0 or 1)
        col_name = f'SUPERTd_{period}_{multiplier}'
        if supertrend is not None and col_name in supertrend.columns:
            # Convert -1 to 0 and 1 to 1
            return (supertrend[col_name] + 1) / 2
        else:
            # Fallback: return the supertrend line
            col_name = f'SUPERT_{period}_{multiplier}'
            if supertrend is not None and col_name in supertrend.columns:
                return supertrend[col_name]
            return pd.Series([np.nan] * len(df), index=df.index)
    except Exception as e:
        logger.warning(f"Supertrend calculation failed: {e}")
        return pd.Series([np.nan] * len(df), index=df.index)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    For daily data without intraday reset, this calculates cumulative VWAP
    or rolling VWAP based on available data.
    
    Args:
        df: DataFrame with OHLC and volume data
    
    Returns:
        Series with VWAP values
    """
    try:
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP as cumulative volume-weighted average
        cumulative_tpv = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()
        
        vwap = cumulative_tpv / cumulative_volume
        
        # Handle division by zero
        vwap = vwap.replace([np.inf, -np.inf], np.nan)
        
        return vwap
    except Exception as e:
        logger.warning(f"VWAP calculation failed: {e}")
        return pd.Series([np.nan] * len(df), index=df.index)


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators to the dataframe.
    
    Indicators:
    - RSI (14, 21)
    - MACD (12, 26, 9)
    - Stochastic (14, 3, 3)
    - ADX (14)
    - Momentum (10)
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with momentum indicators added
    """
    logger.info("Adding momentum indicators...")
    
    # RSI (Relative Strength Index)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_21'] = ta.rsi(df['close'], length=21)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
    else:
        df['stoch_k'] = np.nan
        df['stoch_d'] = np.nan
    
    # ADX (Average Directional Index)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None:
        df['adx_14'] = adx['ADX_14']
    else:
        df['adx_14'] = np.nan
    
    # Momentum
    df['momentum_10'] = ta.mom(df['close'], length=10)
    
    return df


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators to the dataframe.
    
    Indicators:
    - EMA (9, 21, 50, 200)
    - SMA (20, 50)
    - Supertrend (10, 3) and (7, 2)
    - Bollinger Bands (20, 2)
    - Parabolic SAR
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with trend indicators added
    """
    logger.info("Adding trend indicators...")
    
    # EMA (Exponential Moving Average)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200)
    
    # SMA (Simple Moving Average)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    
    # Supertrend
    df['supertrend_10_3'] = calculate_supertrend(df, period=10, multiplier=3.0)
    df['supertrend_7_2'] = calculate_supertrend(df, period=7, multiplier=2.0)
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        # Find column names dynamically (handles different pandas-ta versions)
        bbu_col = [col for col in bbands.columns if col.startswith('BBU_')]
        bbm_col = [col for col in bbands.columns if col.startswith('BBM_')]
        bbl_col = [col for col in bbands.columns if col.startswith('BBL_')]
        bbb_col = [col for col in bbands.columns if col.startswith('BBB_')]
        
        df['bb_upper'] = bbands[bbu_col[0]] if bbu_col else np.nan
        df['bb_middle'] = bbands[bbm_col[0]] if bbm_col else np.nan
        df['bb_lower'] = bbands[bbl_col[0]] if bbl_col else np.nan
        df['bb_width'] = bbands[bbb_col[0]] if bbb_col else np.nan  # Bandwidth percentage
    else:
        df['bb_upper'] = np.nan
        df['bb_middle'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_width'] = np.nan
    
    # Parabolic SAR
    psar = ta.psar(df['high'], df['low'], df['close'])
    if psar is not None:
        # Get the long/short SAR values
        psar_long_col = [col for col in psar.columns if 'PSARl_' in col]
        psar_short_col = [col for col in psar.columns if 'PSARs_' in col]
        
        if psar_long_col and psar_short_col:
            # Combine long and short SAR values
            psar_values = psar[psar_long_col[0]].fillna(psar[psar_short_col[0]])
            df['psar'] = psar_values
        else:
            df['psar'] = np.nan
    else:
        df['psar'] = np.nan
    
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume indicators to the dataframe.
    
    Indicators:
    - Volume SMA (20)
    - Volume Rate of Change (10)
    - On-Balance Volume (OBV)
    - VWAP
    - Money Flow Index (14)
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with volume indicators added
    """
    logger.info("Adding volume indicators...")
    
    # Volume SMA
    df['volume_sma_20'] = ta.sma(df['volume'], length=20)
    
    # Volume Rate of Change
    df['volume_roc_10'] = ta.roc(df['volume'], length=10)
    
    # On-Balance Volume
    df['obv'] = ta.obv(df['close'], df['volume'])
    
    # VWAP (Volume Weighted Average Price)
    df['vwap'] = calculate_vwap(df)
    
    # Money Flow Index
    df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to the dataframe.
    
    Indicators:
    - ATR (14)
    - Historical Volatility (20-period rolling std)
    - Bollinger Band Width Percentage
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with volatility indicators added
    """
    logger.info("Adding volatility indicators...")
    
    # ATR (Average True Range)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Historical Volatility (20-period rolling standard deviation of returns)
    returns = df['close'].pct_change()
    df['hist_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Bollinger Band Width Percentage (relative to middle band)
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
        df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    else:
        # Calculate if not already present
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            bbu_col = [col for col in bbands.columns if col.startswith('BBU_')]
            bbm_col = [col for col in bbands.columns if col.startswith('BBM_')]
            bbl_col = [col for col in bbands.columns if col.startswith('BBL_')]
            if bbu_col and bbm_col and bbl_col:
                df['bb_width_pct'] = (bbands[bbu_col[0]] - bbands[bbl_col[0]]) / bbands[bbm_col[0]] * 100
            else:
                df['bb_width_pct'] = np.nan
        else:
            df['bb_width_pct'] = np.nan
    
    return df


def add_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum-based signal features to better capture UP moves.
    
    These features are specifically designed to improve recall for
    bullish (UP) predictions by identifying momentum-based patterns.
    
    Features:
    - RSI Divergence: Detects divergence between price and RSI
    - MACD Crossover: Identifies MACD line crossing above signal
    - Volume Spike: Detects unusual volume relative to average
    - Price Momentum Score: Combined momentum indicator
    - Trend Strength: Measures strength of current trend
    - Breakout Signal: Identifies potential breakout conditions
    
    Args:
        df: DataFrame with OHLC and volume data
    
    Returns:
        DataFrame with momentum signal features added
    """
    logger.info("Adding momentum signal features...")
    
    # Ensure required indicators exist
    if 'rsi_14' not in df.columns:
        df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # 1. RSI Divergence Detection
    # Bullish divergence: price makes lower low, RSI makes higher low
    price_change_5 = df['close'].diff(5)
    rsi_change_5 = df['rsi_14'].diff(5)
    
    # Bullish divergence (price down, RSI up) - potential UP move
    df['rsi_bullish_divergence'] = (
        (price_change_5 < 0) & (rsi_change_5 > 0) & (df['rsi_14'] < 40)
    ).astype(int)
    
    # Bearish divergence (price up, RSI down) - potential reversal
    df['rsi_bearish_divergence'] = (
        (price_change_5 > 0) & (rsi_change_5 < 0) & (df['rsi_14'] > 60)
    ).astype(int)
    
    # 2. MACD Crossover Detection
    if 'macd' not in df.columns:
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
    
    # MACD crosses above signal line (bullish)
    macd_prev = df['macd'].shift(1)
    signal_prev = df['macd_signal'].shift(1)
    df['macd_bullish_cross'] = (
        (macd_prev <= signal_prev) & (df['macd'] > df['macd_signal'])
    ).astype(int)
    
    # MACD crosses below signal line (bearish)
    df['macd_bearish_cross'] = (
        (macd_prev >= signal_prev) & (df['macd'] < df['macd_signal'])
    ).astype(int)
    
    # 3. Volume Spike Detection
    if 'volume_sma_20' not in df.columns:
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
    
    # Volume spike: volume > 1.5x average
    df['volume_spike'] = (
        df['volume'] > df['volume_sma_20'] * 1.5
    ).astype(int)
    
    # High volume breakout: volume spike with positive close
    df['high_volume_up'] = (
        (df['volume'] > df['volume_sma_20'] * 1.5) & 
        (df['close'] > df['open'])
    ).astype(int)
    
    # 4. Price Momentum Score (composite indicator)
    # Combine multiple momentum factors
    momentum_10 = df['close'].diff(10) / df['close'].shift(10) * 100
    momentum_5 = df['close'].diff(5) / df['close'].shift(5) * 100
    momentum_3 = df['close'].diff(3) / df['close'].shift(3) * 100
    
    # Weighted momentum score
    df['momentum_score'] = (
        0.5 * momentum_3 + 0.3 * momentum_5 + 0.2 * momentum_10
    )
    
    # 5. Trend Strength Indicator
    # Based on EMA alignment and ADX
    if 'ema_9' not in df.columns:
        df['ema_9'] = ta.ema(df['close'], length=9)
    if 'ema_21' not in df.columns:
        df['ema_21'] = ta.ema(df['close'], length=21)
    if 'ema_50' not in df.columns:
        df['ema_50'] = ta.ema(df['close'], length=50)
    
    # Bullish trend: EMA9 > EMA21 > EMA50
    df['ema_bullish_alignment'] = (
        (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
    ).astype(int)
    
    # Bearish trend: EMA9 < EMA21 < EMA50
    df['ema_bearish_alignment'] = (
        (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
    ).astype(int)
    
    # 6. Breakout Signal
    # Price breaks above recent high with volume confirmation
    rolling_high_20 = df['high'].rolling(window=20).max().shift(1)
    rolling_low_20 = df['low'].rolling(window=20).min().shift(1)
    
    df['breakout_up'] = (
        (df['close'] > rolling_high_20) & 
        (df['volume'] > df['volume_sma_20'])
    ).astype(int)
    
    df['breakout_down'] = (
        (df['close'] < rolling_low_20) & 
        (df['volume'] > df['volume_sma_20'])
    ).astype(int)
    
    # 7. RSI Momentum Zones
    # Oversold recovery potential
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # RSI entering momentum zone (30-70 crossing)
    rsi_prev = df['rsi_14'].shift(1)
    df['rsi_momentum_entry'] = (
        (rsi_prev < 30) & (df['rsi_14'] >= 30)
    ).astype(int)
    
    # 8. Combined UP Signal Score (for model training)
    # Higher score = more bullish signals
    df['bullish_signal_count'] = (
        df['rsi_bullish_divergence'] +
        df['macd_bullish_cross'] +
        df['high_volume_up'] +
        df['ema_bullish_alignment'] +
        df['breakout_up'] +
        df['rsi_momentum_entry']
    )
    
    logger.info("Added 15 momentum signal features")
    
    return df


def handle_nan_values(df: pd.DataFrame, indicator_columns: list) -> pd.DataFrame:
    """
    Handle NaN values in indicator columns.
    
    Strategy:
    - Forward-fill for warmup period (first ~200 rows)
    - Then backward-fill for any remaining NaN at start
    - Fill remaining NaN with column median
    
    Args:
        df: DataFrame with indicators
        indicator_columns: List of indicator column names
    
    Returns:
        DataFrame with NaN values handled
    """
    logger.info("Handling NaN values in indicators...")
    
    for col in indicator_columns:
        if col in df.columns:
            # First forward-fill
            df[col] = df[col].ffill()
            # Then backward-fill for remaining NaN at the start
            df[col] = df[col].bfill()
            
            # If still NaN (edge case), fill with column median or 0
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
    
    return df


def normalize_indicators(df: pd.DataFrame, indicator_columns: list, 
                         scaler_path: str = None, fit_scaler: bool = True) -> tuple:
    """
    Normalize indicator columns to [0, 1] range using MinMaxScaler.
    
    Args:
        df: DataFrame with indicators
        indicator_columns: List of indicator column names to normalize
        scaler_path: Path to save/load scaler
        fit_scaler: If True, fit new scaler; if False, load existing scaler
    
    Returns:
        Tuple of (normalized DataFrame, scaler object)
    """
    logger.info("Normalizing indicators to [0, 1] range...")
    
    if fit_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Get indicator data
        indicator_data = df[indicator_columns].values
        
        # Fit and transform
        normalized_data = scaler.fit_transform(indicator_data)
        
        # Update DataFrame
        for i, col in enumerate(indicator_columns):
            df[col] = normalized_data[:, i]
        
        # Save scaler if path provided
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump({'scaler': scaler, 'columns': indicator_columns}, f)
            logger.info(f"Scaler saved to {scaler_path}")
    else:
        # Load existing scaler
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
            saved_columns = scaler_data['columns']
            
            # Verify columns match
            if set(saved_columns) != set(indicator_columns):
                logger.warning("Loaded scaler columns don't match current columns. Using saved columns.")
                indicator_columns = saved_columns
            
            # Transform
            indicator_data = df[indicator_columns].values
            normalized_data = scaler.transform(indicator_data)
            
            for i, col in enumerate(indicator_columns):
                df[col] = normalized_data[:, i]
            
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    return df, scaler


def add_technical_indicators(df: pd.DataFrame, normalize: bool = True,
                            scaler_path: str = None, fit_scaler: bool = True) -> pd.DataFrame:
    """
    Adds all technical indicators to dataframe.
    
    Uses pandas-ta library for indicator calculations.
    Handles NaN values (forward-fill for initial rows).
    Normalizes indicators to [0, 1] using MinMaxScaler.
    
    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        normalize: Whether to normalize indicators to [0, 1]
        scaler_path: Path to save/load scaler (default: models/scalers/technical_scaler.pkl)
        fit_scaler: If True, fit new scaler; if False, load existing scaler
    
    Returns:
        DataFrame with 40-50 new indicator columns, normalized to [0, 1]
    """
    logger.info("=" * 60)
    logger.info("Adding Technical Indicators")
    logger.info("=" * 60)
    
    # Validate input columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Store original columns
    original_columns = list(df.columns)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Add all indicators
    df = add_momentum_indicators(df)
    df = add_trend_indicators(df)
    df = add_volume_indicators(df)
    df = add_volatility_indicators(df)
    df = add_momentum_signals(df)  # New momentum-based features for better recall
    
    # Get list of new indicator columns
    indicator_columns = [col for col in df.columns if col not in original_columns]
    logger.info(f"Added {len(indicator_columns)} indicator columns")
    
    # Handle NaN values
    df = handle_nan_values(df, indicator_columns)
    
    # Check for remaining NaN values
    nan_counts = df[indicator_columns].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Remaining NaN values:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.info("All NaN values handled successfully")
    
    # Normalize if requested
    if normalize:
        if scaler_path is None:
            scaler_path = os.path.join('models', 'scalers', 'technical_scaler.pkl')
        
        df, scaler = normalize_indicators(
            df, indicator_columns, 
            scaler_path=scaler_path, 
            fit_scaler=fit_scaler
        )
    
    # Replace any remaining inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    logger.info("-" * 60)
    logger.info(f"Final DataFrame shape: {df.shape}")
    logger.info(f"Indicator columns: {len(indicator_columns)}")
    logger.info("-" * 60)
    
    return df


def get_indicator_columns() -> list:
    """
    Returns list of all indicator column names.
    
    Returns:
        List of indicator column names
    """
    return [
        # Momentum
        'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d',
        'adx_14',
        'momentum_10',
        # Trend
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'sma_20', 'sma_50',
        'supertrend_10_3', 'supertrend_7_2',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'psar',
        # Volume
        'volume_sma_20', 'volume_roc_10',
        'obv', 'vwap', 'mfi_14',
        # Volatility
        'atr_14', 'hist_vol_20', 'bb_width_pct',
        # Momentum Signals (new features for improved recall)
        'rsi_bullish_divergence', 'rsi_bearish_divergence',
        'macd_bullish_cross', 'macd_bearish_cross',
        'volume_spike', 'high_volume_up',
        'momentum_score',
        'ema_bullish_alignment', 'ema_bearish_alignment',
        'breakout_up', 'breakout_down',
        'rsi_oversold', 'rsi_overbought', 'rsi_momentum_entry',
        'bullish_signal_count'
    ]


def main():
    """Main function for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Add technical indicators to OHLCV data'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path with OHLCV data'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path for data with indicators'
    )
    parser.add_argument(
        '--scaler', '-s',
        default='models/scalers/technical_scaler.pkl',
        help='Path to save/load scaler (default: models/scalers/technical_scaler.pkl)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip normalization'
    )
    parser.add_argument(
        '--load-scaler',
        action='store_true',
        help='Load existing scaler instead of fitting new one'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Add technical indicators
    df_with_indicators = add_technical_indicators(
        df,
        normalize=not args.no_normalize,
        scaler_path=args.scaler,
        fit_scaler=not args.load_scaler
    )
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save output
    df_with_indicators.to_csv(args.output, index=False)
    logger.info(f"Saved data with indicators to {args.output}")
    
    # Print summary
    indicator_cols = get_indicator_columns()
    existing_indicators = [col for col in indicator_cols if col in df_with_indicators.columns]
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input rows: {len(df)}")
    logger.info(f"Output rows: {len(df_with_indicators)}")
    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"Final columns: {len(df_with_indicators.columns)}")
    logger.info(f"Indicator columns added: {len(existing_indicators)}")
    logger.info(f"NaN values in output: {df_with_indicators.isna().sum().sum()}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
