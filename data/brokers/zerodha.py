import logging
import pandas as pd
from datetime import datetime
from .base import BrokerClient
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

logger = logging.getLogger(__name__)

class ZerodhaClient(BrokerClient):
    def __init__(self, config: dict):
        if KiteConnect is None:
            raise ImportError("kiteconnect package is not installed. Please install it using 'pip install kiteconnect'.")
        
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.access_token = config.get('access_token')
        self.kite = KiteConnect(api_key=self.api_key)
        
        if self.access_token:
            self.kite.set_access_token(self.access_token)

    def authenticate(self):
        """
        Zerodha requires a manual login flow to get a request_token, 
        which is then exchanged for an access_token.
        If access_token is already provided in config, this is a no-op.
        """
        if self.access_token:
            logger.info("Using provided access token for Zerodha.")
            return

        login_url = self.kite.login_url()
        print(f"1. Login to this URL: {login_url}")
        request_token = input("2. Enter the request_token from the redirect URL: ")
        
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            logger.info(f"Authentication successful. Access Token: {self.access_token}")
            print(f"IMPORTANT: Update your config with this access_token to avoid logging in again: {self.access_token}")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """
        Fetch historical data from Zerodha.
        
        Args:
            symbol: For Zerodha, this should be the Instrument Token (int) or trading symbol (str).
                    If string, we need to lookup the instrument token.
            interval: 'minute', 'day', '3minute', '5minute', '10minute', '15minute', '30minute', '60minute'
        """
        # Map common intervals to Zerodha intervals
        interval_map = {
            '1m': 'minute',
            '3m': '3minute',
            '5m': '5minute',
            '10m': '10minute',
            '15m': '15minute',
            '30m': '30minute',
            '60m': '60minute',
            '1h': '60minute',
            '1d': 'day',
        }
        
        z_interval = interval_map.get(interval, interval)
        
        # If symbol is a string (e.g., "BANKNIFTY"), we might need to find its instrument token.
        # For simplicity, let's assume the user provides the correct Instrument Token or we do a basic lookup.
        # Here we will try to resolve "NSE:BANKNIFTY" or similar if passed, otherwise assume it's a token if int.
        
        instrument_token = self._resolve_symbol(symbol)
        
        logger.info(f"Fetching data for token {instrument_token} from {start_date} to {end_date} interval {z_interval}")
        
        try:
            records = self.kite.historical_data(
                instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=z_interval
            )
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Zerodha returns: date, open, high, low, close, volume
            # Rename columns to standard format
            df = df.rename(columns={
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # Ensure columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Zerodha: {e}")
            return pd.DataFrame()

    def _resolve_symbol(self, symbol: str) -> int:
        """
        Helper to resolve symbol string to instrument token.
        This is a simplified lookup. In a real app, you'd cache the instrument dump.
        """
        if str(symbol).isdigit():
            return int(symbol)
            
        # If it's a common index name, return hardcoded token (Example for Bank Nifty Index)
        # Note: These tokens change or are specific. 
        # Ideally, we should fetch instruments and search.
        # For now, we will fetch all instruments and search (expensive but accurate).
        
        logger.info(f"Resolving instrument token for {symbol}...")
        try:
            instruments = self.kite.instruments()
            # symbol format expected: "EXCHANGE:SYMBOL" e.g., "NSE:NIFTY BANK" or just "NIFTY BANK"
            
            search_exchange = None
            search_symbol = symbol
            
            if ":" in symbol:
                search_exchange, search_symbol = symbol.split(":")
            
            for inst in instruments:
                if inst['name'] == search_symbol or inst['tradingsymbol'] == search_symbol:
                    if search_exchange:
                        if inst['exchange'] == search_exchange:
                            logger.info(f"Found token {inst['instrument_token']} for {symbol}")
                            return inst['instrument_token']
                    else:
                        # Default to NSE if not specified or first match
                        if inst['exchange'] == 'NSE':
                             logger.info(f"Found token {inst['instrument_token']} for {symbol}")
                             return inst['instrument_token']
            
            # If not found exact match, try contains for indices
            if "BANK" in symbol and "NIFTY" in symbol:
                 # Fallback for Bank Nifty Index
                 for inst in instruments:
                     if inst['name'] == 'NIFTY BANK' and inst['segment'] == 'INDICES':
                         return inst['instrument_token']

            logger.warning(f"Could not resolve symbol {symbol}. Returning as is (might fail).")
            return symbol
        except Exception as e:
            logger.error(f"Error resolving symbol: {e}")
            return symbol
