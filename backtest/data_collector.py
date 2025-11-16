"""
HISTORICAL DATA COLLECTOR
=========================
Collects and stores historical price data for pattern validation.

Supports both Binance and MEXC exchanges via auto-detection.

Author: Grim (Institutional Standards)
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger

# Import both exchange clients
from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv('USE_MEXC', 'False').lower() == 'true'

# Project root
project_root = Path(__file__).parent.parent


class HistoricalDataCollector:
    """
    Collects historical OHLCV data from exchange API.
    
    Automatically uses MEXC or Binance based on USE_MEXC environment variable.
    """
    
    def __init__(self, client=None, data_dir: str = None):
        """
        Initialize historical data collector.
        
        Args:
            client: Exchange client (auto-creates if None)
            data_dir: Directory to store historical data
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                self.exchange_name = "MEXC"
                logger.info("HistoricalDataCollector using MEXC")
            else:
                self.client = BinanceClient()
                self.exchange_name = "Binance"
                logger.info("HistoricalDataCollector using Binance")
        else:
            self.client = client
            self.exchange_name = "MEXC" if USE_MEXC else "Binance"
        
        # Set data directory
        if data_dir is None:
            self.data_dir = Path(project_root) / "data" / "historical"
        else:
            self.data_dir = Path(data_dir)
        
        # Create directory if doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported timeframes
        self.timeframes = ['15m', '1h', '4h', '1d']
        
        logger.info(f"HistoricalDataCollector initialized (exchange={self.exchange_name}, data_dir={self.data_dir})")
    
    
    def collect_pump_data(
        self,
        symbol: str,
        pump_start: datetime,
        pump_end: datetime,
        timeframes: List[str] = None
    ) -> Dict:
        """
        Collect data for a specific pump event.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            pump_start: When pump started
            pump_end: When pump ended
            timeframes: List of timeframes to collect (default: ['15m', '1h', '4h'])
            
        Returns:
            Dict with collected data for each timeframe
        """
        if timeframes is None:
            timeframes = ['15m', '1h', '4h']
        
        logger.info(f"Collecting pump data for {symbol}")
        logger.info(f"  Period: {pump_start} to {pump_end}")
        logger.info(f"  Timeframes: {timeframes}")
        
        collected_data = {}
        
        for tf in timeframes:
            logger.info(f"  Fetching {tf} data...")
            
            # Get klines for this timeframe
            klines = self.client.get_klines(
                symbol=symbol,
                interval=tf,
                start_time=int(pump_start.timestamp() * 1000),
                end_time=int(pump_end.timestamp() * 1000),
                limit=1000
            )
            
            if klines:
                # Convert to DataFrame
                df = self._klines_to_dataframe(klines)
                collected_data[tf] = df
                
                logger.info(f"    ✓ Collected {len(df)} candles")
            else:
                logger.warning(f"    ✗ No data returned for {tf}")
                collected_data[tf] = pd.DataFrame()
        
        return collected_data
    
    
    def save_pump_data(
        self,
        symbol: str,
        pump_name: str,
        data: Dict[str, pd.DataFrame]
    ):
        """
        Save collected pump data to disk.
        
        Args:
            symbol: Trading pair
            pump_name: Identifier for this pump (e.g., 'pengu_dec2024')
            data: Dict of DataFrames (timeframe -> df)
        """
        # Create directory for this symbol
        symbol_dir = self.data_dir / symbol.lower()
        symbol_dir.mkdir(exist_ok=True)
        
        # Create directory for this pump
        pump_dir = symbol_dir / pump_name
        pump_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving pump data to {pump_dir}")
        
        for timeframe, df in data.items():
            if not df.empty:
                # Save as CSV
                csv_path = pump_dir / f"{timeframe}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"  ✓ Saved {timeframe}: {len(df)} candles")
        
        logger.info(f"Pump data saved successfully")
    
    
    def load_pump_data(
        self,
        symbol: str,
        pump_name: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load saved pump data from disk.
        
        Args:
            symbol: Trading pair
            pump_name: Identifier for this pump
            
        Returns:
            Dict of DataFrames (timeframe -> df)
        """
        pump_dir = self.data_dir / symbol.lower() / pump_name
        
        if not pump_dir.exists():
            logger.error(f"Pump data not found: {pump_dir}")
            return {}
        
        logger.info(f"Loading pump data from {pump_dir}")
        
        data = {}
        for csv_file in pump_dir.glob("*.csv"):
            timeframe = csv_file.stem  # Filename without extension
            df = pd.read_csv(csv_file)
            data[timeframe] = df
            logger.info(f"  ✓ Loaded {timeframe}: {len(df)} candles")
        
        return data
    
    
    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Convert klines to pandas DataFrame.
        
        Args:
            klines: List of klines from API
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Keep only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the historical data collector"""
    
    print("\n" + "="*80)
    print("HISTORICAL DATA COLLECTOR - TESTING")
    print("="*80 + "\n")
    
    # Initialize with auto-detected exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")
    
    collector = HistoricalDataCollector()
    
    # Test: Collect data for a recent pump
    print("TEST: Collect Recent BTC Data")
    print("-" * 40)
    
    # Get data from last 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    data = collector.collect_pump_data(
        symbol='BTCUSDT',
        pump_start=start_time,
        pump_end=end_time,
        timeframes=['1h', '4h']
    )
    
    print("\nCollected data:")
    for tf, df in data.items():
        if not df.empty:
            print(f"  {tf}: {len(df)} candles")
            print(f"    First: {df.iloc[0]['timestamp']}")
            print(f"    Last:  {df.iloc[-1]['timestamp']}")
            print(f"    Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)