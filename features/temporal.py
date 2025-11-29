"""
Temporal Features Module for Stock Prediction System

This module extracts time-based features from timestamp column:
- Day of week (cyclical encoding)
- Week of month
- Month of year (cyclical encoding)
- Expiry-related features (BankNifty weekly/monthly expiry)

For intraday data, it also includes:
- Hour of day (cyclical encoding)
- Is first/last hour of trading

All periodic features use cyclical encoding (sin/cos) to preserve continuity.

BankNifty Expiry Timeline:
-------------------------
1. June 13, 2008: BankNifty F&O launched with monthly expiry on last Thursday
2. May 27, 2016: Weekly expiry introduced (every Thursday)
3. September 4, 2023: Weekly expiry moved to Wednesday
4. March 1, 2024: Monthly/Quarterly expiry moved to last Wednesday
5. January 1, 2025: Monthly/Quarterly reverted to last Thursday (weekly stays Wednesday)
6. November 20, 2024: Weekly expiry discontinued (last weekly was Nov 13, 2024)
7. April 4, 2025 onwards: All expiries moved to Tuesday
"""

import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime, timedelta, date
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cyclical_encode(values: pd.Series, max_value: float) -> Tuple[pd.Series, pd.Series]:
    """
    Apply cyclical encoding using sin/cos transformation.
    
    Args:
        values: Series of values to encode
        max_value: Maximum value in the cycle (e.g., 7 for days, 12 for months)
    
    Returns:
        Tuple of (sin_encoded, cos_encoded) Series
    """
    normalized = values / max_value
    sin_encoded = np.sin(2 * np.pi * normalized)
    cos_encoded = np.cos(2 * np.pi * normalized)
    return sin_encoded, cos_encoded


# ============================================================================
# BankNifty Expiry Timeline Constants
# ============================================================================

# Key dates for expiry rule changes
BANKNIFTY_LAUNCH_DATE = date(2008, 6, 13)
WEEKLY_EXPIRY_START = date(2016, 5, 27)  # Weekly expiry introduced
WEEKLY_TO_WEDNESDAY = date(2023, 9, 4)   # Weekly moved to Wednesday
MONTHLY_TO_WEDNESDAY = date(2024, 3, 1)  # Monthly/Quarterly to Wednesday
MONTHLY_BACK_TO_THURSDAY = date(2025, 1, 1)  # Monthly back to Thursday
WEEKLY_EXPIRY_END = date(2024, 11, 13)   # Last weekly expiry
ALL_TO_TUESDAY = date(2025, 4, 4)        # All expiries to Tuesday


def get_last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """
    Get the last occurrence of a specific weekday in a month.
    
    Args:
        year: Year
        month: Month (1-12)
        weekday: Weekday (0=Monday, 1=Tuesday, ..., 6=Sunday)
    
    Returns:
        Date of last occurrence of that weekday
    """
    # Find the last day of the month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    last_day = next_month - timedelta(days=1)
    
    # Find last occurrence of the weekday
    days_since_weekday = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_since_weekday)


def get_monthly_expiry_date(dt: date) -> date:
    """
    Get the monthly expiry date for the given date's month.
    
    BankNifty Monthly Expiry Rules:
    - Before March 1, 2024: Last Thursday
    - March 1, 2024 to Dec 31, 2024: Last Wednesday
    - Jan 1, 2025 to April 3, 2025: Last Thursday
    - April 4, 2025 onwards: Last Tuesday
    
    Args:
        dt: Date to get monthly expiry for
    
    Returns:
        Monthly expiry date for that month
    """
    if dt >= ALL_TO_TUESDAY:
        # Last Tuesday
        return get_last_weekday_of_month(dt.year, dt.month, 1)  # Tuesday = 1
    elif dt >= MONTHLY_BACK_TO_THURSDAY:
        # Last Thursday
        return get_last_weekday_of_month(dt.year, dt.month, 3)  # Thursday = 3
    elif dt >= MONTHLY_TO_WEDNESDAY:
        # Last Wednesday
        return get_last_weekday_of_month(dt.year, dt.month, 2)  # Wednesday = 2
    else:
        # Last Thursday (original rule)
        return get_last_weekday_of_month(dt.year, dt.month, 3)  # Thursday = 3


def get_weekly_expiry_day(dt: date) -> Optional[int]:
    """
    Get the weekday for weekly expiry based on the date.
    
    BankNifty Weekly Expiry Rules:
    - Before May 27, 2016: No weekly expiry
    - May 27, 2016 to Sep 3, 2023: Thursday (weekday=3)
    - Sep 4, 2023 to Nov 13, 2024: Wednesday (weekday=2)
    - After Nov 13, 2024: No weekly expiry (discontinued)
    
    Args:
        dt: Date to check
    
    Returns:
        Weekday number (0-6) or None if no weekly expiry
    """
    if dt < WEEKLY_EXPIRY_START:
        return None  # No weekly expiry before May 27, 2016
    elif dt > WEEKLY_EXPIRY_END:
        return None  # Weekly expiry discontinued after Nov 13, 2024
    elif dt >= WEEKLY_TO_WEDNESDAY:
        return 2  # Wednesday
    else:
        return 3  # Thursday


def is_weekly_expiry_active(dt: date) -> bool:
    """
    Check if weekly expiry is active on the given date.
    
    Args:
        dt: Date to check
    
    Returns:
        True if weekly expiry exists for this date
    """
    return WEEKLY_EXPIRY_START <= dt <= WEEKLY_EXPIRY_END


def get_next_weekly_expiry(dt: date) -> Optional[date]:
    """
    Get the next weekly expiry date from the given date.
    
    Args:
        dt: Current date
    
    Returns:
        Next weekly expiry date, or None if no weekly expiry
    """
    weekly_day = get_weekly_expiry_day(dt)
    if weekly_day is None:
        return None
    
    current_weekday = dt.weekday()
    
    if current_weekday <= weekly_day:
        # Expiry is later this week
        days_until = weekly_day - current_weekday
    else:
        # Expiry is next week
        days_until = 7 - current_weekday + weekly_day
    
    next_expiry = dt + timedelta(days=days_until)
    
    # Check if next expiry is still within weekly expiry period
    if next_expiry > WEEKLY_EXPIRY_END:
        return None
    
    return next_expiry


def get_next_monthly_expiry(dt: date) -> date:
    """
    Get the next monthly expiry date from the given date.
    
    Args:
        dt: Current date
    
    Returns:
        Next monthly expiry date
    """
    # Get this month's expiry
    monthly_expiry = get_monthly_expiry_date(dt)
    
    if dt <= monthly_expiry:
        return monthly_expiry
    else:
        # Get next month's expiry
        if dt.month == 12:
            next_month_date = date(dt.year + 1, 1, 1)
        else:
            next_month_date = date(dt.year, dt.month + 1, 1)
        return get_monthly_expiry_date(next_month_date)


def is_monthly_expiry_day(dt: date) -> bool:
    """
    Check if the given date is a monthly expiry day.
    
    Args:
        dt: Date to check
    
    Returns:
        True if this is a monthly expiry day
    """
    monthly_expiry = get_monthly_expiry_date(dt)
    return dt == monthly_expiry


def is_weekly_expiry_day(dt: date) -> bool:
    """
    Check if the given date is a weekly expiry day.
    
    Args:
        dt: Date to check
    
    Returns:
        True if this is a weekly expiry day
    """
    if not is_weekly_expiry_active(dt):
        return False
    
    weekly_day = get_weekly_expiry_day(dt)
    return dt.weekday() == weekly_day


def get_days_to_weekly_expiry(dt: date) -> int:
    """
    Calculate trading days to next weekly expiry.
    
    Args:
        dt: Current date
    
    Returns:
        Days to weekly expiry (0 = expiry day), or -1 if no weekly expiry
    """
    if not is_weekly_expiry_active(dt):
        return -1
    
    next_expiry = get_next_weekly_expiry(dt)
    if next_expiry is None:
        return -1
    
    return (next_expiry - dt).days


def get_days_to_monthly_expiry(dt: date) -> int:
    """
    Calculate calendar days to next monthly expiry.
    
    Args:
        dt: Current date
    
    Returns:
        Days to monthly expiry (0 = expiry day)
    """
    next_expiry = get_next_monthly_expiry(dt)
    return (next_expiry - dt).days


def detect_data_frequency(df: pd.DataFrame, datetime_col: str = 'datetime') -> str:
    """
    Detect if data is intraday or daily.
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
    
    Returns:
        'intraday' or 'daily'
    """
    if datetime_col not in df.columns:
        logger.warning(f"Column '{datetime_col}' not found, assuming daily data")
        return 'daily'
    
    # Parse datetime if not already
    dt_series = pd.to_datetime(df[datetime_col])
    
    # Check time component
    times = dt_series.dt.time
    unique_times = times.unique()
    
    # If all times are 00:00:00 or there's only one unique time, it's daily
    if len(unique_times) == 1:
        return 'daily'
    
    # Check if there are multiple timestamps on the same day
    dates = dt_series.dt.date
    samples_per_day = dates.value_counts()
    
    if samples_per_day.mean() > 1.5:
        return 'intraday'
    else:
        return 'daily'


def add_hour_features(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """
    Add hour-related features for intraday data.
    
    Trading hours: 9:15 AM to 3:30 PM IST
    Hour index 0-6 representing market hours.
    
    Args:
        df: DataFrame to add features to
        dt_series: Datetime series
    
    Returns:
        DataFrame with hour features added
    """
    logger.info("Adding hour features for intraday data...")
    
    # Extract hour and minute
    hours = dt_series.dt.hour
    minutes = dt_series.dt.minute
    
    # Calculate market hour (0-6)
    # 9:15 AM = 0, 10:15 AM = 1, ..., 3:15 PM = 6
    market_hour = (hours - 9) + (minutes - 15) / 60
    market_hour = market_hour.clip(0, 6)
    
    # Cyclical encoding for hour
    df['hour_sin'], df['hour_cos'] = cyclical_encode(market_hour, 6.0)
    
    # Is first hour of trading (9:15-10:15)
    df['is_first_hour'] = ((hours == 9) & (minutes >= 15)) | ((hours == 10) & (minutes < 15))
    df['is_first_hour'] = df['is_first_hour'].astype(int)
    
    # Is last hour of trading (2:30-3:30)
    df['is_last_hour'] = ((hours == 14) & (minutes >= 30)) | ((hours == 15) & (minutes <= 30))
    df['is_last_hour'] = df['is_last_hour'].astype(int)
    
    return df


def add_day_features(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """
    Add day of week features with cyclical encoding.
    
    Args:
        df: DataFrame to add features to
        dt_series: Datetime series
    
    Returns:
        DataFrame with day features added
    """
    logger.info("Adding day of week features...")
    
    # Day of week (0 = Monday, 4 = Friday)
    day_of_week = dt_series.dt.dayofweek
    
    # Cyclical encoding (5 trading days)
    df['day_sin'], df['day_cos'] = cyclical_encode(day_of_week, 5.0)
    
    return df


def add_week_features(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """
    Add week of month feature.
    
    Args:
        df: DataFrame to add features to
        dt_series: Datetime series
    
    Returns:
        DataFrame with week features added
    """
    logger.info("Adding week of month features...")
    
    # Week of month (1-5)
    day_of_month = dt_series.dt.day
    week_of_month = ((day_of_month - 1) // 7) + 1
    
    df['week_of_month'] = week_of_month
    
    return df


def add_month_features(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """
    Add month of year features with cyclical encoding.
    
    Args:
        df: DataFrame to add features to
        dt_series: Datetime series
    
    Returns:
        DataFrame with month features added
    """
    logger.info("Adding month of year features...")
    
    # Month of year (1-12)
    month = dt_series.dt.month
    
    # Cyclical encoding
    df['month_sin'], df['month_cos'] = cyclical_encode(month, 12.0)
    
    return df


def add_expiry_features(df: pd.DataFrame, dt_series: pd.Series) -> pd.DataFrame:
    """
    Add expiry-related features for BankNifty with accurate historical timeline.
    
    BankNifty Expiry Timeline:
    - June 13, 2008: Launch with monthly expiry on last Thursday
    - May 27, 2016: Weekly expiry introduced (Thursday)
    - Sep 4, 2023: Weekly expiry moved to Wednesday
    - Mar 1, 2024: Monthly expiry moved to last Wednesday
    - Jan 1, 2025: Monthly expiry back to last Thursday
    - Nov 13, 2024: Weekly expiry discontinued
    - Apr 4, 2025: All expiries moved to Tuesday
    
    Args:
        df: DataFrame to add features to
        dt_series: Datetime series
    
    Returns:
        DataFrame with expiry features added
    """
    logger.info("Adding expiry-related features (with historical timeline)...")
    
    # Convert to date objects for processing
    dates = pd.to_datetime(dt_series)
    date_values = dates.dt.date
    
    # --- Monthly Expiry Features ---
    
    # Days to monthly expiry
    df['days_to_monthly_expiry'] = date_values.apply(get_days_to_monthly_expiry)
    
    # Is monthly expiry day
    df['is_monthly_expiry'] = date_values.apply(
        lambda x: 1 if is_monthly_expiry_day(x) else 0
    )
    
    # --- Weekly Expiry Features ---
    
    # Is weekly expiry active (between May 27, 2016 and Nov 13, 2024)
    df['has_weekly_expiry'] = date_values.apply(
        lambda x: 1 if is_weekly_expiry_active(x) else 0
    )
    
    # Days to weekly expiry (-1 if no weekly expiry)
    df['days_to_weekly_expiry'] = date_values.apply(get_days_to_weekly_expiry)
    
    # Is weekly expiry day
    df['is_weekly_expiry'] = date_values.apply(
        lambda x: 1 if is_weekly_expiry_day(x) else 0
    )
    
    # --- Combined Expiry Features ---
    
    # Is any expiry day (weekly or monthly)
    df['is_expiry_day'] = ((df['is_weekly_expiry'] == 1) | (df['is_monthly_expiry'] == 1)).astype(int)
    
    # Days to nearest expiry (weekly if active, otherwise monthly)
    df['days_to_expiry'] = df.apply(
        lambda row: row['days_to_weekly_expiry'] if row['days_to_weekly_expiry'] >= 0 
                    else row['days_to_monthly_expiry'],
        axis=1
    )
    
    # Is expiry week (within 4 days of any expiry)
    df['is_expiry_week'] = (df['days_to_expiry'] <= 4).astype(int)
    
    return df


def add_temporal_features(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Extracts time-based features from timestamp column.
    Uses cyclical encoding (sin/cos) for periodic features.
    
    Features added:
    - For intraday data: hour_sin, hour_cos, is_first_hour, is_last_hour
    - day_sin, day_cos (day of week)
    - week_of_month
    - month_sin, month_cos
    - Weekly expiry: days_to_weekly_expiry, is_weekly_expiry, has_weekly_expiry
    - Monthly expiry: days_to_monthly_expiry, is_monthly_expiry
    - Combined: days_to_expiry, is_expiry_day, is_expiry_week
    
    Args:
        df: DataFrame with OHLCV data and datetime column
        datetime_col: Name of datetime column
    
    Returns:
        DataFrame with 15+ new temporal columns
    """
    logger.info("=" * 60)
    logger.info("Adding Temporal Features")
    logger.info("=" * 60)
    
    # Validate datetime column exists
    if datetime_col not in df.columns:
        raise ValueError(f"DateTime column '{datetime_col}' not found in DataFrame")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Parse datetime
    dt_series = pd.to_datetime(df[datetime_col])
    
    # Detect data frequency
    data_freq = detect_data_frequency(df, datetime_col)
    logger.info(f"Detected data frequency: {data_freq}")
    
    # Store original columns
    original_columns = list(df.columns)
    
    # Add hour features only for intraday data
    if data_freq == 'intraday':
        df = add_hour_features(df, dt_series)
    else:
        logger.info("Skipping hour features for daily data")
    
    # Add day of week features
    df = add_day_features(df, dt_series)
    
    # Add week of month features
    df = add_week_features(df, dt_series)
    
    # Add month features
    df = add_month_features(df, dt_series)
    
    # Add expiry features
    df = add_expiry_features(df, dt_series)
    
    # Get list of new temporal columns
    temporal_columns = [col for col in df.columns if col not in original_columns]
    logger.info(f"Added {len(temporal_columns)} temporal columns: {temporal_columns}")
    
    # Check for NaN values
    nan_counts = df[temporal_columns].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN values found in temporal features:\n{nan_counts[nan_counts > 0]}")
    else:
        logger.info("No NaN values in temporal features")
    
    logger.info("-" * 60)
    logger.info(f"Final DataFrame shape: {df.shape}")
    logger.info("-" * 60)
    
    return df


def get_temporal_columns(is_intraday: bool = False) -> list:
    """
    Returns list of all temporal column names.
    
    Args:
        is_intraday: Whether data is intraday (includes hour features)
    
    Returns:
        List of temporal column names
    """
    columns = [
        # Day features
        'day_sin', 'day_cos',
        # Week features
        'week_of_month',
        # Month features
        'month_sin', 'month_cos',
        # Weekly expiry features
        'has_weekly_expiry', 'days_to_weekly_expiry', 'is_weekly_expiry',
        # Monthly expiry features
        'days_to_monthly_expiry', 'is_monthly_expiry',
        # Combined expiry features
        'days_to_expiry', 'is_expiry_day', 'is_expiry_week'
    ]
    
    if is_intraday:
        columns = ['hour_sin', 'hour_cos', 'is_first_hour', 'is_last_hour'] + columns
    
    return columns


def main():
    """Main function for CLI execution."""
    parser = argparse.ArgumentParser(
        description='Add temporal features to OHLCV data'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path with OHLCV data and datetime column'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path for data with temporal features'
    )
    parser.add_argument(
        '--datetime-col', '-d',
        default='datetime',
        help='Name of datetime column (default: datetime)'
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
    
    # Add temporal features
    df_with_temporal = add_temporal_features(df, datetime_col=args.datetime_col)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save output
    df_with_temporal.to_csv(args.output, index=False)
    logger.info(f"Saved data with temporal features to {args.output}")
    
    # Detect frequency for column count
    data_freq = detect_data_frequency(df, args.datetime_col)
    is_intraday = data_freq == 'intraday'
    temporal_cols = get_temporal_columns(is_intraday)
    existing_temporal = [col for col in temporal_cols if col in df_with_temporal.columns]
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input rows: {len(df)}")
    logger.info(f"Output rows: {len(df_with_temporal)}")
    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"Final columns: {len(df_with_temporal.columns)}")
    logger.info(f"Temporal columns added: {len(existing_temporal)}")
    logger.info(f"Data frequency: {data_freq}")
    logger.info(f"NaN values in output: {df_with_temporal.isna().sum().sum()}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
