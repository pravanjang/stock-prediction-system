from evaluation.evaluate_final import backtest_trading_strategy
import pandas as pd


def create_synthetic_ohlcv_data(n_samples: int = 200) -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)
    start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_samples // 78)
    dates = pd.date_range(start_date.replace(hour=9, minute=15), periods=n_samples, freq='5min')
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
import numpy as np

np.random.seed(42)
# Create synthetic data
df = create_synthetic_ohlcv_data(200)
# Create predictions aligned with sequence_length 60
n_predictions = len(df) - 60
predictions = np.random.randint(0, 2, n_predictions)
proba = np.random.rand(n_predictions)

res = backtest_trading_strategy(
    df=df,
    predictions=predictions,
    proba=proba,
    sequence_length=60,
    lot_size=25,
    transaction_cost=0.0003,
    trades_csv_path='evaluation/reports/test_backtest_trades.csv'
)
print('Total trades:', res['total_trades'])
print('Trades csv saved to:', res.get('trades_csv_path'))
