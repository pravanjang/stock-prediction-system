#!/usr/bin/env python3
"""
Data Collection Module for BankNifty Stock Prediction System (NSE Source).

This module provides functionality to:
- Fetch historical BankNifty data from NSE (using NSEMasterData) or local CSV
- Clean and validate the data
- Split data into train/validation/test sets
- Save processed data with statistics
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytz

# Add NSE directory to path to import NSEMasterData
current_dir = os.path.dirname(os.path.abspath(__file__))
nse_dir = os.path.join(current_dir, 'NSE')
if nse_dir not in sys.path:
    sys.path.append(nse_dir)

try:
    from NSEMasterData import NSEMasterData
except ImportError:
    # Fallback if running from root
    try:
        from data.NSE.NSEMasterData import NSEMasterData
    except ImportError:
        print("Could not import NSEMasterData. Make sure it is in data/NSE/")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BANKNIFTY_SYMBOL = "NIFTY BANK"  # NSE symbol for Bank Nifty
IST = pytz.timezone('Asia/Kolkata')
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def fetch_banknifty_data(
    start_date: str,
    end_date: str,
    interval: str = '15m',
    source: str = 'nse',
    csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical BankNifty data from specified source.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (default: 15m)
        source: Data source - 'nse' or 'csv'
        csv_path: Path to local CSV file (required if source is 'csv')
    
    Returns:
        DataFrame with OHLCV data indexed by datetime
    
    Raises:
        ValueError: If invalid source or missing csv_path
    """
    logger.info(f"Fetching BankNifty data from {start_date} to {end_date}")
    
    if source == 'nse':
        return _fetch_from_nse(start_date, end_date, interval)
    elif source == 'csv':
        if csv_path is None:
            raise ValueError("csv_path is required when source is 'csv'")
        return _fetch_from_csv(csv_path)
    else:
        raise ValueError(f"Unsupported source: {source}. Use 'nse' or 'csv'")


def _fetch_from_nse(start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Fetch data from NSE using NSEMasterData."""
    
    logger.info(f"Downloading data from NSE for {BANKNIFTY_SYMBOL}")
    
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    try:
        nse = NSEMasterData()
        nse.download_symbol_master()
        
        df = nse.get_history(
            symbol=BANKNIFTY_SYMBOL,
            exchange='NSE',
            start=start_dt,
            end=end_dt,
            interval=interval
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch data from NSE: {e}")
        return pd.DataFrame()
    
    if df.empty:
        logger.warning("No data returned from NSE")
        return pd.DataFrame()
    
    # NSEMasterData returns columns: Open, High, Low, Close, Volume (and index is Timestamp)
    # We need to standardize column names to lowercase
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Select only OHLCV columns
    columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    
    # Ensure index is datetime and timezone-aware
    # NSEMasterData returns timezone-naive timestamps (but adjusted to IST implicitly or explicitly)
    # We should localize to IST if naive
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)
    
    df.index.name = 'datetime'
    
    logger.info(f"Successfully fetched {len(df)} rows from NSE")
    return df


def _fetch_from_csv(csv_path: str) -> pd.DataFrame:
    """Load data from local CSV file."""
    logger.info(f"Loading data from CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Try to identify and set datetime column
    datetime_columns = ['datetime', 'date', 'timestamp', 'Date', 'Datetime', 'Timestamp']
    datetime_col = None
    
    for col in datetime_columns:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col is None:
        # Assume first column is datetime
        datetime_col = df.columns[0]
    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)
    df.index.name = 'datetime'
    
    # Make timezone-aware if not already
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure required OHLCV columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in CSV. Setting to NaN.")
            df[col] = float('nan')
    
    df = df[required_columns]
    
    logger.info(f"Successfully loaded {len(df)} rows from CSV")
    return df


def clean_data(df: pd.DataFrame, log_file: Optional[str] = None) -> pd.DataFrame:
    """
    Clean and validate the data with quality checks.
    
    Args:
        df: Raw DataFrame with OHLCV data
        log_file: Optional path to write cleaning log
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame received for cleaning")
        return df
    
    cleaning_log = []
    original_rows = len(df)
    
    def log_step(message: str):
        cleaning_log.append(f"{datetime.now().isoformat()} - {message}")
        logger.info(message)
    
    log_step(f"Starting data cleaning. Initial rows: {original_rows}")
    
    # 1. Remove duplicate timestamps
    duplicates_before = df.index.duplicated().sum()
    if duplicates_before > 0:
        df = df[~df.index.duplicated(keep='last')]
        log_step(f"Removed {duplicates_before} duplicate timestamps")
    
    # 2. Sort by datetime index
    df = df.sort_index()
    log_step("Data sorted by datetime index")
    
    # 3. Filter to trading hours (9:15 AM - 3:30 PM IST)
    if df.index.tz is not None:
        trading_hours_mask = (
            (df.index.time >= MARKET_OPEN) & 
            (df.index.time <= MARKET_CLOSE)
        )
        non_trading_rows = (~trading_hours_mask).sum()
        if non_trading_rows > 0:
            df = df[trading_hours_mask]
            log_step(f"Removed {non_trading_rows} rows outside trading hours (9:15 AM - 3:30 PM IST)")
    
    # 4. Flag and handle abnormal price gaps (>5% moves in 15min)
    df = df.copy()
    df['price_change_pct'] = df['close'].pct_change().abs() * 100
    abnormal_gaps = df['price_change_pct'] > 5
    abnormal_count = abnormal_gaps.sum()
    
    if abnormal_count > 0:
        # We flag but don't remove these (they could be legitimate)
        log_step(f"Flagged {abnormal_count} abnormal price gaps (>5% change)")
        # Store the flags for reference
        df['abnormal_gap'] = abnormal_gaps
    else:
        df['abnormal_gap'] = False
    
    # 5. Handle missing values with forward-fill strategy
    missing_before = df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
    if missing_before > 0:
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].ffill()
        # Also backward fill for any remaining NaNs at the start
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].bfill()
        log_step(f"Forward-filled {missing_before} missing values in OHLCV columns")
    
    # 6. Validate price relationships (high >= low, etc.)
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    invalid_count = invalid_ohlc.sum()
    if invalid_count > 0:
        # Fix invalid OHLC by recalculating
        log_step(f"Found {invalid_count} rows with invalid OHLC relationships")
        df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
        df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
        log_step("Fixed invalid OHLC relationships")
    
    # 7. Remove temporary columns used for analysis
    df = df.drop(columns=['price_change_pct', 'abnormal_gap'], errors='ignore')
    
    # Final summary
    final_rows = len(df)
    rows_removed = original_rows - final_rows
    log_step(f"Cleaning complete. Final rows: {final_rows}. Removed: {rows_removed}")
    
    # Write cleaning log to file if specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write('\n'.join(cleaning_log))
        logger.info(f"Cleaning log written to {log_file}")
    
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets (chronologically).
    
    Args:
        df: Clean DataFrame with OHLCV data
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    
    Raises:
        ValueError: If ratios don't sum to <= 1.0
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Data split - Train: {len(train_df)} rows ({train_ratio*100:.0f}%), "
                f"Val: {len(val_df)} rows ({val_ratio*100:.0f}%), "
                f"Test: {len(test_df)} rows ({test_ratio*100:.0f}%)")
    
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'data/processed/'
) -> None:
    """
    Save train, validation, and test splits to CSV files.
    
    Also generates data_stats.json with statistics about the data.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory path
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV files
    train_df.to_csv(output_path / 'train.csv')
    val_df.to_csv(output_path / 'val.csv')
    test_df.to_csv(output_path / 'test.csv')
    
    logger.info(f"Saved train.csv ({len(train_df)} rows)")
    logger.info(f"Saved val.csv ({len(val_df)} rows)")
    logger.info(f"Saved test.csv ({len(test_df)} rows)")
    
    # Generate and save statistics
    all_data = pd.concat([train_df, val_df, test_df])
    
    stats = {
        'date_range': {
            'start': str(all_data.index.min()),
            'end': str(all_data.index.max())
        },
        'total_rows': len(all_data),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'train_date_range': {
            'start': str(train_df.index.min()) if not train_df.empty else None,
            'end': str(train_df.index.max()) if not train_df.empty else None
        },
        'val_date_range': {
            'start': str(val_df.index.min()) if not val_df.empty else None,
            'end': str(val_df.index.max()) if not val_df.empty else None
        },
        'test_date_range': {
            'start': str(test_df.index.min()) if not test_df.empty else None,
            'end': str(test_df.index.max()) if not test_df.empty else None
        },
        'missing_values': {
            col: int(all_data[col].isna().sum()) 
            for col in ['open', 'high', 'low', 'close', 'volume']
            if col in all_data.columns
        },
        'generated_at': datetime.now().isoformat()
    }
    
    with open(output_path / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved data_stats.json to {output_path}")


def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Fetch and process BankNifty historical data (NSE Source)'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='15m',
        help='Data interval (default: 15m)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='nse',
        choices=['nse', 'csv'],
        help='Data source (default: nse)'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='Path to local CSV file (required if source is csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/',
        help='Output directory for processed data (default: data/processed/)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training data ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation data ratio (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return 1
    
    # Ensure start_date is before end_date
    if start_dt >= end_dt:
        logger.error("start_date must be before end_date")
        return 1
    
    # Validate ratio arguments
    if args.train_ratio < 0 or args.train_ratio > 1:
        logger.error("train_ratio must be between 0 and 1")
        return 1
    if args.val_ratio < 0 or args.val_ratio > 1:
        logger.error("val_ratio must be between 0 and 1")
        return 1
    if args.train_ratio + args.val_ratio > 1.0:
        logger.error("train_ratio + val_ratio must not exceed 1.0")
        return 1
    
    # Determine cleaning log path
    cleaning_log_path = os.path.join(args.output_dir, 'cleaning_log.txt')
    
    try:
        # Fetch data
        df = fetch_banknifty_data(
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            source=args.source,
            csv_path=args.csv_path
        )
        
        if df.empty:
            logger.error("No data fetched. Please check your parameters and try again.")
            return 1
        
        # Clean data
        df = clean_data(df, log_file=cleaning_log_path)
        
        if df.empty:
            logger.error("No data remaining after cleaning.")
            return 1
        
        # Split data
        train_df, val_df, test_df = split_data(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        
        # Save splits
        save_splits(train_df, val_df, test_df, output_dir=args.output_dir)
        
        logger.info("Data processing complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
