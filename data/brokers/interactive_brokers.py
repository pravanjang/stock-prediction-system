import logging
import pandas as pd
from datetime import datetime
import time
from .base import BrokerClient

try:
    from ib_insync import IB, Stock, Index, util
except ImportError:
    IB = None

logger = logging.getLogger(__name__)

class InteractiveBrokersClient(BrokerClient):
    def __init__(self, config: dict):
        if IB is None:
            raise ImportError("ib_insync package is not installed. Please install it using 'pip install ib_insync'.")
        
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497) # 7497 for TWS paper trading, 7496 for live
        self.client_id = config.get('client_id', 1)
        self.ib = IB()

    def authenticate(self):
        """
        Connect to TWS or IB Gateway.
        """
        logger.info(f"Connecting to IBKR at {self.host}:{self.port} with clientId {self.client_id}...")
        try:
            if not self.ib.isConnected():
                self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info("Connected to Interactive Brokers.")
        except Exception as e:
            logger.error(f"Connection to IBKR failed: {e}")
            logger.error("Ensure TWS or IB Gateway is running and API connections are enabled.")
            raise

    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """
        Fetch historical data from IBKR.
        
        Args:
            symbol: For IBKR, this should be the symbol name (e.g., 'BANKNIFTY'). 
                    Note: IBKR symbols for NSE might differ.
            interval: '1m', '1h', '1d', etc.
        """
        if not self.ib.isConnected():
            self.authenticate()

        # Map intervals to IBKR bar sizes
        # IBKR format: '1 min', '5 mins', '1 hour', '1 day'
        interval_map = {
            '1m': '1 min',
            '3m': '3 mins',
            '5m': '5 mins',
            '15m': '15 mins',
            '30m': '30 mins',
            '1h': '1 hour',
            '1d': '1 day'
        }
        bar_size = interval_map.get(interval, '1 day')
        
        # Determine duration string based on date range
        # IBKR requires a duration string like '1 D', '1 W', '1 M', '1 Y'
        # We will calculate the delta and request slightly more to cover the range
        delta = end_date - start_date
        if delta.days < 1:
            duration = '1 D'
        elif delta.days < 7:
            duration = f"{delta.days + 1} D"
        elif delta.days < 30:
            duration = '1 M'
        else:
            duration = '1 Y' # Simplified

        logger.info(f"Requesting {symbol} data. Duration: {duration}, Bar Size: {bar_size}")

        try:
            # Define Contract
            # Assuming NSE Index. You might need to adjust exchange/currency.
            # For Bank Nifty Index on NSE:
            contract = Index(symbol, 'NSE', 'INR')
            
            # Qualify contract to get local symbol and other details
            self.ib.qualifyContracts(contract)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning("No data returned from IBKR.")
                return pd.DataFrame()
            
            df = util.df(bars)
            
            # IBKR returns: date, open, high, low, close, volume, barCount, average
            df = df.rename(columns={
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert datetime to timezone aware if needed
            # IBKR 'date' is usually pd.Timestamp or datetime.date
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter by start_date as duration might have fetched more
            # Ensure start_date is timezone aware if df['datetime'] is
            if df['datetime'].dt.tz is not None and start_date.tzinfo is None:
                 start_date = start_date.replace(tzinfo=df['datetime'].dt.tz)
            
            df = df[df['datetime'] >= start_date]
            
            df = df.set_index('datetime')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df

        except Exception as e:
            logger.error(f"Error fetching data from IBKR: {e}")
            return pd.DataFrame()
