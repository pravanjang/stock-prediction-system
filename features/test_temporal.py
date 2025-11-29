"""
Test module for temporal features with comprehensive BankNifty expiry regime testing.

Tests cover all expiry regimes:
1. Before May 27, 2016: Monthly only (last Thursday)
2. May 27, 2016 - Sept 3, 2023: Weekly Thursday, Monthly last Thursday  
3. Sept 4, 2023 - Feb 29, 2024: Weekly Wednesday, Monthly last Thursday
4. March 1, 2024 - Dec 31, 2024: All expiries on Wednesday
5. Jan 1, 2025 - Nov 13, 2024: Monthly Thursday, Weekly Wednesday
6. Nov 13, 2024+: No weekly expiry (SEBI discontinued)
7. April 4, 2025+: All expiries on Tuesday
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date

from temporal import (
    get_weekly_expiry_day,
    get_monthly_expiry_date,
    get_last_weekday_of_month,
    get_next_weekly_expiry,
    get_next_monthly_expiry,
    get_days_to_weekly_expiry,
    get_days_to_monthly_expiry,
    is_weekly_expiry_day,
    is_monthly_expiry_day,
    is_weekly_expiry_active,
    add_temporal_features,
    add_expiry_features,
    get_temporal_columns,
    WEEKLY_EXPIRY_START,
    WEEKLY_TO_WEDNESDAY,
    MONTHLY_TO_WEDNESDAY,
    MONTHLY_BACK_TO_THURSDAY,
    WEEKLY_EXPIRY_END,
    ALL_TO_TUESDAY,
)


class TestExpiryDateConstants(unittest.TestCase):
    """Test that key expiry date constants are correct."""
    
    def test_weekly_expiry_start(self):
        """Weekly expiry started May 27, 2016."""
        self.assertEqual(WEEKLY_EXPIRY_START, date(2016, 5, 27))
        
    def test_weekly_to_wednesday(self):
        """Weekly expiry moved to Wednesday on Sept 4, 2023."""
        self.assertEqual(WEEKLY_TO_WEDNESDAY, date(2023, 9, 4))
        
    def test_monthly_to_wednesday(self):
        """Monthly expiry moved to Wednesday on March 1, 2024."""
        self.assertEqual(MONTHLY_TO_WEDNESDAY, date(2024, 3, 1))
        
    def test_monthly_back_to_thursday(self):
        """Monthly expiry back to Thursday on Jan 1, 2025."""
        self.assertEqual(MONTHLY_BACK_TO_THURSDAY, date(2025, 1, 1))
        
    def test_weekly_expiry_end(self):
        """Weekly expiry ended Nov 13, 2024."""
        self.assertEqual(WEEKLY_EXPIRY_END, date(2024, 11, 13))
        
    def test_all_to_tuesday(self):
        """All expiries moved to Tuesday on April 4, 2025."""
        self.assertEqual(ALL_TO_TUESDAY, date(2025, 4, 4))


class TestWeeklyExpiryDay(unittest.TestCase):
    """Test get_weekly_expiry_day function."""
    
    def test_no_weekly_before_may_2016(self):
        """No weekly expiry before May 27, 2016."""
        dt = date(2015, 6, 15)
        self.assertIsNone(get_weekly_expiry_day(dt))
        
        dt = date(2016, 5, 26)  # Day before weekly started
        self.assertIsNone(get_weekly_expiry_day(dt))
        
    def test_weekly_thursday_may_2016_to_sept_2023(self):
        """Weekly on Thursday from May 27, 2016 to Sept 3, 2023."""
        # First day of weekly expiry
        dt = date(2016, 5, 27)
        self.assertEqual(get_weekly_expiry_day(dt), 3)  # Thursday
        
        # Random date in this period
        dt = date(2020, 8, 15)
        self.assertEqual(get_weekly_expiry_day(dt), 3)  # Thursday
        
        # Last day before Wednesday shift
        dt = date(2023, 9, 3)
        self.assertEqual(get_weekly_expiry_day(dt), 3)  # Thursday
        
    def test_weekly_wednesday_sept_2023_to_nov_2024(self):
        """Weekly on Wednesday from Sept 4, 2023 to Nov 13, 2024."""
        # First day of Wednesday weekly
        dt = date(2023, 9, 4)
        self.assertEqual(get_weekly_expiry_day(dt), 2)  # Wednesday
        
        # Random date in this period
        dt = date(2024, 6, 15)
        self.assertEqual(get_weekly_expiry_day(dt), 2)  # Wednesday
        
        # Last weekly expiry date
        dt = date(2024, 11, 13)
        self.assertEqual(get_weekly_expiry_day(dt), 2)  # Wednesday
        
    def test_no_weekly_after_nov_2024(self):
        """No weekly expiry after Nov 13, 2024."""
        dt = date(2024, 11, 14)
        self.assertIsNone(get_weekly_expiry_day(dt))
        
        dt = date(2025, 3, 15)
        self.assertIsNone(get_weekly_expiry_day(dt))


class TestMonthlyExpiryDate(unittest.TestCase):
    """Test get_monthly_expiry_date function."""
    
    def test_last_thursday_before_march_2024(self):
        """Monthly expiry on last Thursday before March 1, 2024."""
        # January 2020 - last Thursday is Jan 30
        expiry = get_monthly_expiry_date(date(2020, 1, 15))
        self.assertEqual(expiry.day, 30)
        self.assertEqual(expiry.weekday(), 3)  # Thursday
        
        # August 2023 - last Thursday is Aug 31
        expiry = get_monthly_expiry_date(date(2023, 8, 1))
        self.assertEqual(expiry.day, 31)
        self.assertEqual(expiry.weekday(), 3)  # Thursday
        
    def test_last_wednesday_march_to_dec_2024(self):
        """Monthly expiry on last Wednesday from March 1 to Dec 31, 2024."""
        # June 2024 - last Wednesday is June 26
        expiry = get_monthly_expiry_date(date(2024, 6, 15))
        self.assertEqual(expiry.day, 26)
        self.assertEqual(expiry.weekday(), 2)  # Wednesday
        
        # October 2024 - last Wednesday is Oct 30
        expiry = get_monthly_expiry_date(date(2024, 10, 1))
        self.assertEqual(expiry.day, 30)
        self.assertEqual(expiry.weekday(), 2)  # Wednesday
        
    def test_last_thursday_jan_to_april_2025(self):
        """Monthly expiry back to last Thursday from Jan 1 to April 3, 2025."""
        # January 2025 - last Thursday is Jan 30
        expiry = get_monthly_expiry_date(date(2025, 1, 15))
        self.assertEqual(expiry.day, 30)
        self.assertEqual(expiry.weekday(), 3)  # Thursday
        
        # March 2025 - last Thursday is March 27
        expiry = get_monthly_expiry_date(date(2025, 3, 1))
        self.assertEqual(expiry.day, 27)
        self.assertEqual(expiry.weekday(), 3)  # Thursday
        
    def test_last_tuesday_after_april_2025(self):
        """Monthly expiry on last Tuesday from April 4, 2025 onwards."""
        # April 2025 - last Tuesday is April 29
        expiry = get_monthly_expiry_date(date(2025, 4, 15))
        self.assertEqual(expiry.day, 29)
        self.assertEqual(expiry.weekday(), 1)  # Tuesday
        
        # December 2025 - last Tuesday is Dec 30
        expiry = get_monthly_expiry_date(date(2025, 12, 1))
        self.assertEqual(expiry.day, 30)
        self.assertEqual(expiry.weekday(), 1)  # Tuesday


class TestLastWeekdayOfMonth(unittest.TestCase):
    """Test get_last_weekday_of_month helper function."""
    
    def test_last_thursday_january_2020(self):
        """Last Thursday of January 2020 is Jan 30."""
        result = get_last_weekday_of_month(2020, 1, 3)  # Thursday = 3
        self.assertEqual(result, date(2020, 1, 30))
        
    def test_last_wednesday_june_2024(self):
        """Last Wednesday of June 2024 is June 26."""
        result = get_last_weekday_of_month(2024, 6, 2)  # Wednesday = 2
        self.assertEqual(result, date(2024, 6, 26))
        
    def test_last_tuesday_april_2025(self):
        """Last Tuesday of April 2025 is April 29."""
        result = get_last_weekday_of_month(2025, 4, 1)  # Tuesday = 1
        self.assertEqual(result, date(2025, 4, 29))


class TestWeeklyExpiryCalculations(unittest.TestCase):
    """Test weekly expiry date calculations."""
    
    def test_is_weekly_expiry_active_before_start(self):
        """Weekly not active before May 27, 2016."""
        self.assertFalse(is_weekly_expiry_active(date(2016, 5, 26)))
        
    def test_is_weekly_expiry_active_during_period(self):
        """Weekly active from May 27, 2016 to Nov 13, 2024."""
        self.assertTrue(is_weekly_expiry_active(date(2016, 5, 27)))
        self.assertTrue(is_weekly_expiry_active(date(2020, 8, 15)))
        self.assertTrue(is_weekly_expiry_active(date(2024, 11, 13)))
        
    def test_is_weekly_expiry_active_after_end(self):
        """Weekly not active after Nov 13, 2024."""
        self.assertFalse(is_weekly_expiry_active(date(2024, 11, 14)))
        self.assertFalse(is_weekly_expiry_active(date(2025, 1, 1)))
        
    def test_next_weekly_expiry_thursday_regime(self):
        """Next weekly expiry in Thursday regime."""
        # Monday Aug 17, 2020 - next Thursday is Aug 20
        next_expiry = get_next_weekly_expiry(date(2020, 8, 17))
        self.assertEqual(next_expiry, date(2020, 8, 20))
        self.assertEqual(next_expiry.weekday(), 3)  # Thursday
        
    def test_next_weekly_expiry_wednesday_regime(self):
        """Next weekly expiry in Wednesday regime."""
        # Monday Jan 8, 2024 - next Wednesday is Jan 10
        next_expiry = get_next_weekly_expiry(date(2024, 1, 8))
        self.assertEqual(next_expiry, date(2024, 1, 10))
        self.assertEqual(next_expiry.weekday(), 2)  # Wednesday
        
    def test_next_weekly_expiry_none_after_end(self):
        """No next weekly expiry after Nov 13, 2024."""
        self.assertIsNone(get_next_weekly_expiry(date(2024, 11, 14)))
        self.assertIsNone(get_next_weekly_expiry(date(2025, 3, 1)))
        
    def test_days_to_weekly_expiry(self):
        """Test days to weekly expiry calculation."""
        # Monday Aug 17, 2020 - next Thursday is Aug 20 = 3 days
        days = get_days_to_weekly_expiry(date(2020, 8, 17))
        self.assertEqual(days, 3)
        
        # On expiry day = 0 days
        days = get_days_to_weekly_expiry(date(2020, 8, 20))
        self.assertEqual(days, 0)
        
    def test_days_to_weekly_expiry_no_weekly(self):
        """Days to weekly is -1 when no weekly expiry."""
        self.assertEqual(get_days_to_weekly_expiry(date(2015, 1, 1)), -1)
        self.assertEqual(get_days_to_weekly_expiry(date(2025, 1, 1)), -1)
        
    def test_is_weekly_expiry_day_thursday(self):
        """Test is_weekly_expiry_day for Thursday regime."""
        # Aug 20, 2020 is Thursday
        self.assertTrue(is_weekly_expiry_day(date(2020, 8, 20)))
        self.assertFalse(is_weekly_expiry_day(date(2020, 8, 19)))  # Wednesday
        
    def test_is_weekly_expiry_day_wednesday(self):
        """Test is_weekly_expiry_day for Wednesday regime."""
        # Jan 10, 2024 is Wednesday
        self.assertTrue(is_weekly_expiry_day(date(2024, 1, 10)))
        self.assertFalse(is_weekly_expiry_day(date(2024, 1, 11)))  # Thursday


class TestMonthlyExpiryCalculations(unittest.TestCase):
    """Test monthly expiry calculations."""
    
    def test_next_monthly_expiry_same_month(self):
        """Next monthly expiry when date is before expiry in same month."""
        # Aug 1, 2020 - expiry is Aug 27 (last Thursday)
        next_expiry = get_next_monthly_expiry(date(2020, 8, 1))
        self.assertEqual(next_expiry.month, 8)
        self.assertEqual(next_expiry.weekday(), 3)  # Thursday
        
    def test_next_monthly_expiry_next_month(self):
        """Next monthly expiry when date is after expiry in current month."""
        # Aug 28, 2020 - expiry was Aug 27, next is Sept 24
        next_expiry = get_next_monthly_expiry(date(2020, 8, 28))
        self.assertEqual(next_expiry.month, 9)
        self.assertEqual(next_expiry.weekday(), 3)  # Thursday
        
    def test_days_to_monthly_expiry(self):
        """Test days to monthly expiry calculation."""
        # Aug 1, 2020 - expiry is Aug 27 = 26 days
        days = get_days_to_monthly_expiry(date(2020, 8, 1))
        self.assertEqual(days, 26)
        
        # On expiry day = 0 days
        days = get_days_to_monthly_expiry(date(2020, 8, 27))
        self.assertEqual(days, 0)
        
    def test_is_monthly_expiry_day(self):
        """Test is_monthly_expiry_day function."""
        # Aug 27, 2020 is last Thursday
        self.assertTrue(is_monthly_expiry_day(date(2020, 8, 27)))
        self.assertFalse(is_monthly_expiry_day(date(2020, 8, 26)))


class TestTemporalFeatures(unittest.TestCase):
    """Test temporal feature generation."""
    
    def test_add_temporal_features_columns_exist(self):
        """Test that add_temporal_features adds expected columns."""
        dates = pd.date_range('2020-01-01', periods=10, freq='1h')
        df = pd.DataFrame({'datetime': dates, 'Close': np.random.randn(10)})
        
        result = add_temporal_features(df.copy())
        
        # Check cyclical features exist
        self.assertIn('day_sin', result.columns)
        self.assertIn('day_cos', result.columns)
        self.assertIn('month_sin', result.columns)
        self.assertIn('month_cos', result.columns)
        self.assertIn('week_of_month', result.columns)
        
        # Check expiry features exist
        self.assertIn('days_to_weekly_expiry', result.columns)
        self.assertIn('days_to_monthly_expiry', result.columns)
        self.assertIn('is_weekly_expiry', result.columns)
        self.assertIn('is_monthly_expiry', result.columns)
        self.assertIn('has_weekly_expiry', result.columns)
        
    def test_cyclical_values_in_range(self):
        """Test that cyclical features are in [-1, 1] range."""
        dates = pd.date_range('2020-01-01', periods=100, freq='1h')
        df = pd.DataFrame({'datetime': dates, 'Close': np.random.randn(100)})
        
        result = add_temporal_features(df.copy())
        
        for col in ['day_sin', 'day_cos', 'month_sin', 'month_cos']:
            self.assertTrue(result[col].min() >= -1.0)
            self.assertTrue(result[col].max() <= 1.0)
            
    def test_intraday_has_hour_features(self):
        """Test that intraday data gets hour features."""
        # Create intraday data with multiple timestamps per day
        dates = pd.date_range('2020-01-02 09:15', periods=50, freq='15min')
        df = pd.DataFrame({'datetime': dates, 'Close': np.random.randn(50)})
        
        result = add_temporal_features(df.copy())
        
        self.assertIn('hour_sin', result.columns)
        self.assertIn('hour_cos', result.columns)


class TestExpiryFeatures(unittest.TestCase):
    """Test expiry feature generation."""
    
    def test_add_expiry_features_columns_exist(self):
        """Test expiry feature addition."""
        dates = pd.date_range('2020-08-01', periods=30, freq='1D')
        df = pd.DataFrame({'datetime': dates, 'Close': np.random.randn(30)})
        
        dt_series = pd.to_datetime(df['datetime'])
        result = add_expiry_features(df.copy(), dt_series)
        
        # Check all expiry columns exist
        expected_cols = [
            'days_to_weekly_expiry',
            'days_to_monthly_expiry',
            'is_weekly_expiry',
            'is_monthly_expiry',
            'has_weekly_expiry',
            'is_expiry_day',
            'days_to_expiry',
            'is_expiry_week',
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")
            
    def test_has_weekly_expiry_flag(self):
        """Test has_weekly_expiry flag for different periods."""
        # During weekly era (2020)
        dates = pd.date_range('2020-08-01', periods=5, freq='1D')
        df = pd.DataFrame({'datetime': dates, 'Close': np.random.randn(5)})
        result = add_expiry_features(df.copy(), pd.to_datetime(df['datetime']))
        self.assertTrue((result['has_weekly_expiry'] == 1).all())
        
        # After weekly ended (2025)
        dates = pd.date_range('2025-01-01', periods=5, freq='1D')
        df = pd.DataFrame({'datetime': dates, 'Close': np.random.randn(5)})
        result = add_expiry_features(df.copy(), pd.to_datetime(df['datetime']))
        self.assertTrue((result['has_weekly_expiry'] == 0).all())


class TestGetTemporalColumns(unittest.TestCase):
    """Test column name retrieval."""
    
    def test_get_temporal_columns_daily(self):
        """Test that get_temporal_columns returns expected columns for daily data."""
        cols = get_temporal_columns(is_intraday=False)
        
        expected = [
            'day_sin', 'day_cos',
            'week_of_month',
            'month_sin', 'month_cos',
            'has_weekly_expiry', 'days_to_weekly_expiry', 'is_weekly_expiry',
            'days_to_monthly_expiry', 'is_monthly_expiry',
            'days_to_expiry', 'is_expiry_day', 'is_expiry_week',
        ]
        
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")
            
    def test_get_temporal_columns_intraday(self):
        """Test that get_temporal_columns includes hour features for intraday."""
        cols = get_temporal_columns(is_intraday=True)
        
        self.assertIn('hour_sin', cols)
        self.assertIn('hour_cos', cols)
        self.assertIn('is_first_hour', cols)
        self.assertIn('is_last_hour', cols)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and regime boundaries."""
    
    def test_boundary_weekly_start(self):
        """Test boundary at May 27, 2016 (weekly start)."""
        # Day before - no weekly
        self.assertFalse(is_weekly_expiry_active(date(2016, 5, 26)))
        self.assertIsNone(get_weekly_expiry_day(date(2016, 5, 26)))
        
        # First day of weekly
        self.assertTrue(is_weekly_expiry_active(date(2016, 5, 27)))
        self.assertEqual(get_weekly_expiry_day(date(2016, 5, 27)), 3)  # Thursday
        
    def test_boundary_weekly_to_wednesday(self):
        """Test boundary at Sept 4, 2023 (weekly to Wednesday)."""
        # Day before - Thursday
        self.assertEqual(get_weekly_expiry_day(date(2023, 9, 3)), 3)
        
        # First day of Wednesday
        self.assertEqual(get_weekly_expiry_day(date(2023, 9, 4)), 2)
        
    def test_boundary_monthly_to_wednesday(self):
        """Test boundary at March 1, 2024 (monthly to Wednesday)."""
        # Feb 29, 2024 (leap year!) - Thursday
        expiry = get_monthly_expiry_date(date(2024, 2, 15))
        self.assertEqual(expiry.weekday(), 3)  # Thursday
        
        # March 2024 - Wednesday
        expiry = get_monthly_expiry_date(date(2024, 3, 15))
        self.assertEqual(expiry.weekday(), 2)  # Wednesday
        
    def test_boundary_monthly_back_to_thursday(self):
        """Test boundary at Jan 1, 2025 (monthly back to Thursday)."""
        # Dec 2024 - Wednesday
        expiry = get_monthly_expiry_date(date(2024, 12, 15))
        self.assertEqual(expiry.weekday(), 2)  # Wednesday
        
        # Jan 2025 - Thursday
        expiry = get_monthly_expiry_date(date(2025, 1, 15))
        self.assertEqual(expiry.weekday(), 3)  # Thursday
        
    def test_boundary_weekly_end(self):
        """Test boundary at Nov 13, 2024 (last weekly expiry)."""
        # Nov 13, 2024 - still active
        self.assertTrue(is_weekly_expiry_active(date(2024, 11, 13)))
        
        # Nov 14, 2024 - no longer active
        self.assertFalse(is_weekly_expiry_active(date(2024, 11, 14)))
        
    def test_boundary_all_to_tuesday(self):
        """Test boundary at April 4, 2025 (all to Tuesday)."""
        # April 3, 2025 - Thursday
        expiry = get_monthly_expiry_date(date(2025, 4, 3))
        self.assertEqual(expiry.weekday(), 3)  # Thursday (for March)
        
        # April 4, 2025 - Tuesday
        expiry = get_monthly_expiry_date(date(2025, 4, 4))
        self.assertEqual(expiry.weekday(), 1)  # Tuesday


if __name__ == '__main__':
    unittest.main()
