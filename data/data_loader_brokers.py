#!/usr/bin/env python3
"""
Data Collection Module for BankNifty Stock Prediction System (Broker API Source).

This module provides functionality to:
- Fetch historical BankNifty data from configured Broker (e.g., Zerodha)
- Clean and validate the data
- Split data into train/validation/test sets
- Save processed data with statistics
"""

import argparse
import json
import logging
import os
import sys
import yaml
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytz

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.brokers.zerodha import ZerodhaClient
from data.brokers.sharekhan import SharekhanClient
from data.brokers.interactive_brokers import InteractiveBrokersClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# For Zerodha, we usually need the instrument token or a resolvable symbol
# "NSE:NIFTY BANK" is the trading symbol for Bank Nifty Index
BANKNIFTY_SYMBOL = "NSE:NIFTY BANK" 
IST = pytz.timezone('Asia/Kolkata')
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

def load_config(config_path: str) -> dict:
    """Load broker configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_broker_client(config: dict):
    """Factory method to get the active broker client."""
    active_broker = config.get('active_broker')
    
    if active_broker == 'zerodha':
        broker_config = config['brokers'].get('zerodha', {})
        return ZerodhaClient(broker_config)
    elif active_broker == 'sharekhan':
        broker_config = config['brokers'].get('sharekhan', {})
        return SharekhanClient(broker_config)
    elif active_broker == 'interactive_brokers':
        broker_config = config['brokers'].get('interactive_brokers', {})
        return InteractiveBrokersClient(broker_config)
    else:
        raise ValueError(f"Unsupported broker: {active_broker}")

def fetch_banknifty_data(
    start_date: str,
    end_date: str,
    interval: str = '15m',
    config_path: str = 'configs/brokers_config.yaml'
) -> pd.DataFrame:
    """
    Fetch historical BankNifty data from configured broker.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (default: 15m)
        config_path: Path to broker config file
    
    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    logger.info(f"Fetching BankNifty data from {start_date} to {end_date} using broker")
    
    try:
        # Load config
        full_config_path = os.path.join(project_root, config_path)
        config = load_config(full_config_path)
        
        # Initialize broker client
        client = get_broker_client(config)
        
        # Authenticate
        client.authenticate()
        
        # Convert dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch data
        df = client.fetch_historical_data(
            symbol=BANKNIFTY_SYMBOL,
            start_date=start_dt,
            end_date=end_dt,
            interval=interval
        )
        
        if df.empty:
            logger.warning("No data returned from Broker")
            return pd.DataFrame()
        
        # Ensure index is datetime and timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(IST)
        else:
            df.index = df.index.tz_convert(IST)
        
        df.index.name = 'datetime'
        
        logger.info(f"Successfully fetched {len(df)} rows from Broker")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data from broker: {e}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame, log_file: Optional[str] = None) -> pd.DataFrame:
    """
    Clean and validate the data with quality checks.
    (Same implementation as original data_loader)
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
        log_step(f"Flagged {abnormal_count} abnormal price gaps (>5% change)")
        df['abnormal_gap'] = abnormal_gaps
    else:
        df['abnormal_gap'] = False
    
    # 5. Handle missing values with forward-fill strategy
    missing_before = df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
    if missing_before > 0:
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].ffill()
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].bfill()
        log_step(f"Forward-filled {missing_before} missing values in OHLCV columns")
    
    # 6. Validate price relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    invalid_count = invalid_ohlc.sum()
    if invalid_count > 0:
        log_step(f"Found {invalid_count} rows with invalid OHLC relationships")
        df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
        df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
        log_step("Fixed invalid OHLC relationships")
    
    # 7. Remove temporary columns
    df = df.drop(columns=['price_change_pct', 'abnormal_gap'], errors='ignore')
    
    final_rows = len(df)
    rows_removed = original_rows - final_rows
    log_step(f"Cleaning complete. Final rows: {final_rows}. Removed: {rows_removed}")
    
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
    """Split data into train, validation, and test sets."""
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
    """Save train, validation, and test splits to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / 'train.csv')
    val_df.to_csv(output_path / 'val.csv')
    test_df.to_csv(output_path / 'test.csv')
    
    logger.info(f"Saved train.csv ({len(train_df)} rows)")
    logger.info(f"Saved val.csv ({len(val_df)} rows)")
    logger.info(f"Saved test.csv ({len(test_df)} rows)")
    
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
        'generated_at': datetime.now().isoformat()
    }
    
    with open(output_path / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved data_stats.json to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fetch and process BankNifty data from Brokers')
    parser.add_argument('--start_date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--interval', type=str, default='15m', help='Data interval')
    parser.add_argument('--output_dir', type=str, default='data/processed/', help='Output directory')
    parser.add_argument('--config', type=str, default='configs/brokers_config.yaml', help='Path to broker config')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    
    args = parser.parse_args()
    
    # Determine cleaning log path
    cleaning_log_path = os.path.join(args.output_dir, 'cleaning_log.txt')
    
    try:
        df = fetch_banknifty_data(
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            config_path=args.config
        )
        
        if df.empty:
            logger.error("No data fetched.")
            return 1
        
        df = clean_data(df, log_file=cleaning_log_path)
        
        if df.empty:
            logger.error("No data remaining after cleaning.")
            return 1
        
        train_df, val_df, test_df = split_data(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        
        save_splits(train_df, val_df, test_df, output_dir=args.output_dir)
        
        logger.info("Data processing complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
