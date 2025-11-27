from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class BrokerClient(ABC):
    """Abstract base class for broker API wrappers."""

    @abstractmethod
    def authenticate(self):
        """Authenticate with the broker API."""
        pass

    @abstractmethod
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Broker-specific symbol token or name.
            start_date: Start datetime.
            end_date: End datetime.
            interval: Data timeframe (e.g., '15m', '1d').
            
        Returns:
            pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and datetime index.
        """
        pass
