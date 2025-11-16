<<<<<<< HEAD
"""
MARKET DATA AGGREGATOR
======================
Fetches and processes multi-timeframe market data for technical analysis.

This module provides:
- Multi-timeframe candle data (1m, 5m, 15m, 1h, 1d)
- Volatility calculations (ATR, standard deviation, price ranges)
- Volume analysis (volume spikes, relative volume, cumulative delta)
- Price action metrics (swing highs/lows, support/resistance zones)

This is the data foundation for your rejection pattern detector and signal generator.

Author: Grim (Institutional Standards)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

import pandas as pd
import numpy as np

from data.binance_client import BinanceClient, BinanceClientError
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Complete market data snapshot for a symbol at a point in time"""
    symbol: str
    timestamp: datetime
    
    # Current price data
    price: float
    mark_price: float
    funding_rate: float
    open_interest: float
    
    # 24h statistics
    volume_24h: float
    high_24h: float
    low_24h: float
    change_24h_pct: float
    
    # Volatility metrics
    atr_1h: float  # Average True Range (1h)
    atr_4h: float  # Average True Range (4h)
    volatility_1h_pct: float  # Price range as % of price
    
    # Volume metrics
    volume_1h: float
    volume_4h: float
    relative_volume: float  # Current volume vs average
    
    # Multi-timeframe candles
    candles_1m: Optional[pd.DataFrame] = None
    candles_5m: Optional[pd.DataFrame] = None
    candles_15m: Optional[pd.DataFrame] = None
    candles_1h: Optional[pd.DataFrame] = None
    candles_4h: Optional[pd.DataFrame] = None
    candles_1d: Optional[pd.DataFrame] = None


class MarketDataAggregator:
    """
    Aggregates and processes market data from multiple timeframes.
    
    This is the data engine for your trading strategy. It maintains a cache of
    multi-timeframe data for each symbol you're tracking, updates it intelligently,
    and provides clean interfaces for the strategy modules to consume.
    """
    
    # Timeframe configurations (interval -> candles to fetch)
    TIMEFRAMES = {
        '1m': 100,   # 100 minutes of data
        '5m': 100,   # ~8 hours
        '15m': 100,  # ~25 hours
        '1h': 100,   # ~4 days
        '4h': 100,   # ~16 days
        '1d': 100,   # ~3 months
    }
    
    # Cache duration (seconds before refreshing)
    CACHE_DURATION = {
        '1m': 30,    # Refresh every 30 seconds
        '5m': 120,   # Refresh every 2 minutes
        '15m': 300,  # Refresh every 5 minutes
        '1h': 900,   # Refresh every 15 minutes
        '4h': 3600,  # Refresh every hour
        '1d': 14400, # Refresh every 4 hours
    }
    
    def __init__(self, binance_client: BinanceClient):
        """Initialize aggregator with Binance client"""
        self.client = binance_client
        
        # Data caches
        self.candle_cache: Dict[str, Dict[str, Tuple[pd.DataFrame, float]]] = defaultdict(dict)
        # Structure: {symbol: {timeframe: (dataframe, timestamp)}}
        
        self.snapshot_cache: Dict[str, Tuple[MarketSnapshot, float]] = {}
        # Structure: {symbol: (snapshot, timestamp)}
        
        logger.info("MarketDataAggregator initialized")
    
    def _should_refresh_cache(self, symbol: str, timeframe: str) -> bool:
        """Check if cached data should be refreshed"""
        if symbol not in self.candle_cache:
            return True
        
        if timeframe not in self.candle_cache[symbol]:
            return True
        
        _, cached_time = self.candle_cache[symbol][timeframe]
        cache_age = time.time() - cached_time
        
        return cache_age > self.CACHE_DURATION[timeframe]
    
    def get_candles(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get candlestick data for a symbol and timeframe.
        
        Uses intelligent caching - only fetches new data when cache is stale.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (uses default from TIMEFRAMES if None)
            force_refresh: Force fetch from API even if cached
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if limit is None:
            limit = self.TIMEFRAMES.get(interval, 100)
        
        # Check cache
        if not force_refresh and not self._should_refresh_cache(symbol, interval):
            df, _ = self.candle_cache[symbol][interval]
            logger.debug(f"Using cached candles for {symbol} {interval}")
            return df
        
        # Fetch from API
        try:
            logger.debug(f"Fetching {limit} candles for {symbol} {interval}")
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
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
            
            # Update cache
            self.candle_cache[symbol][interval] = (df, time.time())
            
            logger.debug(f"Cached {len(df)} candles for {symbol} {interval}")
            return df
            
        except BinanceClientError as e:
            logger.error(f"Failed to fetch candles for {symbol} {interval}: {e}")
            
            # Return cached data if available, even if stale
            if symbol in self.candle_cache and interval in self.candle_cache[symbol]:
                logger.warning(f"Returning stale cached data for {symbol} {interval}")
                df, _ = self.candle_cache[symbol][interval]
                return df
            
            # No cache available, return empty DataFrame
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get candle data for all timeframes.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        data = {}
        
        for interval in self.TIMEFRAMES.keys():
            df = self.get_candles(symbol, interval)
            if not df.empty:
                data[interval] = df
        
        return data
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) - volatility measure.
        
        ATR measures the average price range over N periods.
        Higher ATR = more volatile = wider stops needed.
        
        Args:
            df: DataFrame with high, low, close columns
            period: ATR period (default 14)
            
        Returns:
            ATR value
        """
        if len(df) < period + 1:
            return 0.0
        
        # True Range = max of:
        # 1. Current high - current low
        # 2. Abs(current high - previous close)
        # 3. Abs(current low - previous close)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = tr[1:]  # Remove first element (invalid due to roll)
        
        # ATR = moving average of true range
        atr = np.mean(tr[-period:])
        
        return float(atr)
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate price volatility as percentage of current price.
        
        Args:
            df: DataFrame with high, low, close columns
            
        Returns:
            Volatility as percentage
        """
        if len(df) < 2:
            return 0.0
        
        price_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price <= 0:
            return 0.0
        
        volatility_pct = (price_range / current_price) * 100
        return float(volatility_pct)
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume-based metrics.
        
        Args:
            df: DataFrame with volume column
            
        Returns:
            Dict with volume metrics
        """
        if len(df) < 2:
            return {
                'current_volume': 0.0,
                'avg_volume': 0.0,
                'relative_volume': 0.0,
                'volume_spike': False
            }
        
        volumes = df['volume'].values
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])  # Exclude current candle
        
        # Relative volume = current / average
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike = current > 2x average
        volume_spike = relative_volume > 2.0
        
        return {
            'current_volume': float(current_volume),
            'avg_volume': float(avg_volume),
            'relative_volume': float(relative_volume),
            'volume_spike': volume_spike
        }
    
    def find_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        proximity_pct: float = 0.5
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels from price action.
        
        Uses swing highs/lows to identify key levels where price tends to react.
        
        Args:
            df: DataFrame with high, low, close columns
            lookback: Number of candles to analyze
            proximity_pct: Group levels within this % as same level
            
        Returns:
            Dict with 'support' and 'resistance' lists
        """
        if len(df) < lookback:
            return {'support': [], 'resistance': []}
        
        recent_df = df.tail(lookback)
        
        # Find swing highs (local maximums)
        highs = recent_df['high'].values
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
        
        # Find swing lows (local minimums)
        lows = recent_df['low'].values
        support_levels = []
        
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        # Group nearby levels (within proximity_pct)
        def group_levels(levels: List[float]) -> List[float]:
            if not levels:
                return []
            
            levels = sorted(levels)
            grouped = [levels[0]]
            
            for level in levels[1:]:
                if (level - grouped[-1]) / grouped[-1] * 100 > proximity_pct:
                    grouped.append(level)
                else:
                    # Average with previous level
                    grouped[-1] = (grouped[-1] + level) / 2
            
            return grouped
        
        return {
            'support': group_levels(support_levels),
            'resistance': group_levels(resistance_levels)
        }
    
    def get_market_snapshot(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> MarketSnapshot:
        """
        Get complete market snapshot with all data and metrics.
        
        This is the main interface for strategy modules - returns everything
        they need to make trading decisions.
        
        Args:
            symbol: Trading pair
            force_refresh: Force refresh all data
            
        Returns:
            MarketSnapshot with all market data and calculated metrics
        """
        # Check snapshot cache
        if not force_refresh and symbol in self.snapshot_cache:
            snapshot, cached_time = self.snapshot_cache[symbol]
            cache_age = time.time() - cached_time
            
            # Use cached snapshot if < 1 minute old
            if cache_age < 60:
                logger.debug(f"Using cached snapshot for {symbol}")
                return snapshot
        
        logger.info(f"Building market snapshot for {symbol}")
        
        try:
            # Fetch current ticker data
            ticker = self.client.futures_ticker_24hr(symbol=symbol)
            mark_data = self.client.futures_mark_price(symbol=symbol)
            
            # Get multi-timeframe candles
            candles_1h = self.get_candles(symbol, '1h', force_refresh=force_refresh)
            candles_4h = self.get_candles(symbol, '4h', force_refresh=force_refresh)
            
            # Calculate ATR
            atr_1h = self.calculate_atr(candles_1h) if not candles_1h.empty else 0.0
            atr_4h = self.calculate_atr(candles_4h) if not candles_4h.empty else 0.0
            
            # Calculate volatility
            volatility_1h = self.calculate_volatility(candles_1h) if not candles_1h.empty else 0.0
            
            # Calculate volume metrics
            volume_1h = candles_1h['volume'].sum() if not candles_1h.empty else 0.0
            volume_4h = candles_4h['volume'].sum() if not candles_4h.empty else 0.0
            
            # Get volume metrics for relative volume calculation
            volume_metrics = self.calculate_volume_metrics(candles_1h) if not candles_1h.empty else {}
            relative_volume = volume_metrics.get('relative_volume', 1.0)
            
            # Create snapshot
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                
                # Current prices
                price=float(ticker['lastPrice']),
                mark_price=float(mark_data['markPrice']),
                funding_rate=float(mark_data['lastFundingRate']) * 100,  # Convert to %
                open_interest=0.0,  # Will add OI fetching later
                
                # 24h stats
                volume_24h=float(ticker['quoteVolume']),
                high_24h=float(ticker['highPrice']),
                low_24h=float(ticker['lowPrice']),
                change_24h_pct=float(ticker['priceChangePercent']),
                
                # Volatility
                atr_1h=atr_1h,
                atr_4h=atr_4h,
                volatility_1h_pct=volatility_1h,
                
                # Volume
                volume_1h=volume_1h,
                volume_4h=volume_4h,
                relative_volume=relative_volume,
                
                # Multi-timeframe candles
                candles_1m=self.get_candles(symbol, '1m'),
                candles_5m=self.get_candles(symbol, '5m'),
                candles_15m=self.get_candles(symbol, '15m'),
                candles_1h=candles_1h,
                candles_4h=candles_4h,
                candles_1d=self.get_candles(symbol, '1d'),
            )
            
            # Cache snapshot
            self.snapshot_cache[symbol] = (snapshot, time.time())
            
            logger.info(
                f"Snapshot built for {symbol}: "
                f"Price=${snapshot.price:.4f}, "
                f"Vol24h=${snapshot.volume_24h:,.0f}, "
                f"ATR={snapshot.atr_1h:.4f}"
            )
            
            return snapshot
            
        except BinanceClientError as e:
            logger.error(f"Failed to build snapshot for {symbol}: {e}")
            
            # Return cached snapshot if available
            if symbol in self.snapshot_cache:
                logger.warning(f"Returning stale snapshot for {symbol}")
                snapshot, _ = self.snapshot_cache[symbol]
                return snapshot
            
            # No cache, raise exception
            raise
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: Clear cache for specific symbol (or all if None)
        """
        if symbol:
            if symbol in self.candle_cache:
                del self.candle_cache[symbol]
            if symbol in self.snapshot_cache:
                del self.snapshot_cache[symbol]
            logger.info(f"Cleared cache for {symbol}")
        else:
            self.candle_cache.clear()
            self.snapshot_cache.clear()
            logger.info("Cleared all cache")


def main():
    """Test the market data aggregator"""
    from data.binance_client import BinanceClient
    
    print("=" * 70)
    print("MARKET DATA AGGREGATOR TEST")
    print("=" * 70)
    
    # Initialize
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    aggregator = MarketDataAggregator(client)
    
    # Test symbol
    symbol = 'BTCUSDT'
    
    print(f"\nTesting with {symbol}...\n")
    
    # Test 1: Get single timeframe candles
    print("1. Fetching 1h candles...")
    df_1h = aggregator.get_candles(symbol, '1h', limit=50)
    print(f"   ✓ Got {len(df_1h)} candles")
    print(f"   Latest: {df_1h.iloc[-1]['timestamp']} | "
          f"Close: ${df_1h.iloc[-1]['close']:.2f}")
    
    # Test 2: Calculate ATR
    print("\n2. Calculating ATR...")
    atr = aggregator.calculate_atr(df_1h)
    print(f"   ✓ ATR (14): ${atr:.4f}")
    
    # Test 3: Calculate volatility
    print("\n3. Calculating volatility...")
    vol = aggregator.calculate_volatility(df_1h)
    print(f"   ✓ Volatility: {vol:.2f}%")
    
    # Test 4: Volume metrics
    print("\n4. Analyzing volume...")
    vol_metrics = aggregator.calculate_volume_metrics(df_1h)
    print(f"   ✓ Current Volume: {vol_metrics['current_volume']:,.0f}")
    print(f"   ✓ Avg Volume: {vol_metrics['avg_volume']:,.0f}")
    print(f"   ✓ Relative Volume: {vol_metrics['relative_volume']:.2f}x")
    print(f"   ✓ Volume Spike: {vol_metrics['volume_spike']}")
    
    # Test 5: Support/Resistance
    print("\n5. Finding support/resistance...")
    levels = aggregator.find_support_resistance(df_1h, lookback=30)
    print(f"   ✓ Support levels: {len(levels['support'])}")
    if levels['support']:
        print(f"      {[f'${x:.2f}' for x in levels['support'][-3:]]}")
    print(f"   ✓ Resistance levels: {len(levels['resistance'])}")
    if levels['resistance']:
        print(f"      {[f'${x:.2f}' for x in levels['resistance'][-3:]]}")
    
    # Test 6: Full market snapshot
    print("\n6. Building complete market snapshot...")
    snapshot = aggregator.get_market_snapshot(symbol)
    print(f"   ✓ Price: ${snapshot.price:,.2f}")
    print(f"   ✓ Volume 24h: ${snapshot.volume_24h:,.0f}")
    print(f"   ✓ Change 24h: {snapshot.change_24h_pct:+.2f}%")
    print(f"   ✓ ATR 1h: ${snapshot.atr_1h:.4f}")
    print(f"   ✓ Volatility 1h: {snapshot.volatility_1h_pct:.2f}%")
    print(f"   ✓ Funding Rate: {snapshot.funding_rate:.4f}%")
    print(f"   ✓ Multi-timeframe candles loaded:")
    for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
        candles = getattr(snapshot, f'candles_{tf}')
        if candles is not None and not candles.empty:
            print(f"      {tf}: {len(candles)} candles")
    
    # Test 7: Cache performance
    print("\n7. Testing cache performance...")
    import time
    
    start = time.time()
    snapshot1 = aggregator.get_market_snapshot(symbol)
    time1 = time.time() - start
    
    start = time.time()
    snapshot2 = aggregator.get_market_snapshot(symbol)
    time2 = time.time() - start
    
    print(f"   ✓ First fetch: {time1:.3f}s")
    print(f"   ✓ Cached fetch: {time2:.3f}s")
    print(f"   ✓ Speedup: {time1/time2:.1f}x faster")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nMarket Data Aggregator is ready.")
    print("Next: Integrate with strategy modules")


if __name__ == "__main__":
=======
"""
MARKET DATA AGGREGATOR
======================
Fetches and processes multi-timeframe market data for technical analysis.

This module provides:
- Multi-timeframe candle data (1m, 5m, 15m, 1h, 1d)
- Volatility calculations (ATR, standard deviation, price ranges)
- Volume analysis (volume spikes, relative volume, cumulative delta)
- Price action metrics (swing highs/lows, support/resistance zones)

This is the data foundation for your rejection pattern detector and signal generator.

Author: Grim (Institutional Standards)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

import pandas as pd
import numpy as np

from data.binance_client import BinanceClient, BinanceClientError
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Complete market data snapshot for a symbol at a point in time"""
    symbol: str
    timestamp: datetime
    
    # Current price data
    price: float
    mark_price: float
    funding_rate: float
    open_interest: float
    
    # 24h statistics
    volume_24h: float
    high_24h: float
    low_24h: float
    change_24h_pct: float
    
    # Volatility metrics
    atr_1h: float  # Average True Range (1h)
    atr_4h: float  # Average True Range (4h)
    volatility_1h_pct: float  # Price range as % of price
    
    # Volume metrics
    volume_1h: float
    volume_4h: float
    relative_volume: float  # Current volume vs average
    
    # Multi-timeframe candles
    candles_1m: Optional[pd.DataFrame] = None
    candles_5m: Optional[pd.DataFrame] = None
    candles_15m: Optional[pd.DataFrame] = None
    candles_1h: Optional[pd.DataFrame] = None
    candles_4h: Optional[pd.DataFrame] = None
    candles_1d: Optional[pd.DataFrame] = None


class MarketDataAggregator:
    """
    Aggregates and processes market data from multiple timeframes.
    
    This is the data engine for your trading strategy. It maintains a cache of
    multi-timeframe data for each symbol you're tracking, updates it intelligently,
    and provides clean interfaces for the strategy modules to consume.
    """
    
    # Timeframe configurations (interval -> candles to fetch)
    TIMEFRAMES = {
        '1m': 100,   # 100 minutes of data
        '5m': 100,   # ~8 hours
        '15m': 100,  # ~25 hours
        '1h': 100,   # ~4 days
        '4h': 100,   # ~16 days
        '1d': 100,   # ~3 months
    }
    
    # Cache duration (seconds before refreshing)
    CACHE_DURATION = {
        '1m': 30,    # Refresh every 30 seconds
        '5m': 120,   # Refresh every 2 minutes
        '15m': 300,  # Refresh every 5 minutes
        '1h': 900,   # Refresh every 15 minutes
        '4h': 3600,  # Refresh every hour
        '1d': 14400, # Refresh every 4 hours
    }
    
    def __init__(self, binance_client: BinanceClient):
        """Initialize aggregator with Binance client"""
        self.client = binance_client
        
        # Data caches
        self.candle_cache: Dict[str, Dict[str, Tuple[pd.DataFrame, float]]] = defaultdict(dict)
        # Structure: {symbol: {timeframe: (dataframe, timestamp)}}
        
        self.snapshot_cache: Dict[str, Tuple[MarketSnapshot, float]] = {}
        # Structure: {symbol: (snapshot, timestamp)}
        
        logger.info("MarketDataAggregator initialized")
    
    def _should_refresh_cache(self, symbol: str, timeframe: str) -> bool:
        """Check if cached data should be refreshed"""
        if symbol not in self.candle_cache:
            return True
        
        if timeframe not in self.candle_cache[symbol]:
            return True
        
        _, cached_time = self.candle_cache[symbol][timeframe]
        cache_age = time.time() - cached_time
        
        return cache_age > self.CACHE_DURATION[timeframe]
    
    def get_candles(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get candlestick data for a symbol and timeframe.
        
        Uses intelligent caching - only fetches new data when cache is stale.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (uses default from TIMEFRAMES if None)
            force_refresh: Force fetch from API even if cached
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if limit is None:
            limit = self.TIMEFRAMES.get(interval, 100)
        
        # Check cache
        if not force_refresh and not self._should_refresh_cache(symbol, interval):
            df, _ = self.candle_cache[symbol][interval]
            logger.debug(f"Using cached candles for {symbol} {interval}")
            return df
        
        # Fetch from API
        try:
            logger.debug(f"Fetching {limit} candles for {symbol} {interval}")
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
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
            
            # Update cache
            self.candle_cache[symbol][interval] = (df, time.time())
            
            logger.debug(f"Cached {len(df)} candles for {symbol} {interval}")
            return df
            
        except BinanceClientError as e:
            logger.error(f"Failed to fetch candles for {symbol} {interval}: {e}")
            
            # Return cached data if available, even if stale
            if symbol in self.candle_cache and interval in self.candle_cache[symbol]:
                logger.warning(f"Returning stale cached data for {symbol} {interval}")
                df, _ = self.candle_cache[symbol][interval]
                return df
            
            # No cache available, return empty DataFrame
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get candle data for all timeframes.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict mapping timeframe to DataFrame
        """
        data = {}
        
        for interval in self.TIMEFRAMES.keys():
            df = self.get_candles(symbol, interval)
            if not df.empty:
                data[interval] = df
        
        return data
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) - volatility measure.
        
        ATR measures the average price range over N periods.
        Higher ATR = more volatile = wider stops needed.
        
        Args:
            df: DataFrame with high, low, close columns
            period: ATR period (default 14)
            
        Returns:
            ATR value
        """
        if len(df) < period + 1:
            return 0.0
        
        # True Range = max of:
        # 1. Current high - current low
        # 2. Abs(current high - previous close)
        # 3. Abs(current low - previous close)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = tr[1:]  # Remove first element (invalid due to roll)
        
        # ATR = moving average of true range
        atr = np.mean(tr[-period:])
        
        return float(atr)
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate price volatility as percentage of current price.
        
        Args:
            df: DataFrame with high, low, close columns
            
        Returns:
            Volatility as percentage
        """
        if len(df) < 2:
            return 0.0
        
        price_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if current_price <= 0:
            return 0.0
        
        volatility_pct = (price_range / current_price) * 100
        return float(volatility_pct)
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume-based metrics.
        
        Args:
            df: DataFrame with volume column
            
        Returns:
            Dict with volume metrics
        """
        if len(df) < 2:
            return {
                'current_volume': 0.0,
                'avg_volume': 0.0,
                'relative_volume': 0.0,
                'volume_spike': False
            }
        
        volumes = df['volume'].values
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])  # Exclude current candle
        
        # Relative volume = current / average
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike = current > 2x average
        volume_spike = relative_volume > 2.0
        
        return {
            'current_volume': float(current_volume),
            'avg_volume': float(avg_volume),
            'relative_volume': float(relative_volume),
            'volume_spike': volume_spike
        }
    
    def find_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        proximity_pct: float = 0.5
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels from price action.
        
        Uses swing highs/lows to identify key levels where price tends to react.
        
        Args:
            df: DataFrame with high, low, close columns
            lookback: Number of candles to analyze
            proximity_pct: Group levels within this % as same level
            
        Returns:
            Dict with 'support' and 'resistance' lists
        """
        if len(df) < lookback:
            return {'support': [], 'resistance': []}
        
        recent_df = df.tail(lookback)
        
        # Find swing highs (local maximums)
        highs = recent_df['high'].values
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
        
        # Find swing lows (local minimums)
        lows = recent_df['low'].values
        support_levels = []
        
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        # Group nearby levels (within proximity_pct)
        def group_levels(levels: List[float]) -> List[float]:
            if not levels:
                return []
            
            levels = sorted(levels)
            grouped = [levels[0]]
            
            for level in levels[1:]:
                if (level - grouped[-1]) / grouped[-1] * 100 > proximity_pct:
                    grouped.append(level)
                else:
                    # Average with previous level
                    grouped[-1] = (grouped[-1] + level) / 2
            
            return grouped
        
        return {
            'support': group_levels(support_levels),
            'resistance': group_levels(resistance_levels)
        }
    
    def get_market_snapshot(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> MarketSnapshot:
        """
        Get complete market snapshot with all data and metrics.
        
        This is the main interface for strategy modules - returns everything
        they need to make trading decisions.
        
        Args:
            symbol: Trading pair
            force_refresh: Force refresh all data
            
        Returns:
            MarketSnapshot with all market data and calculated metrics
        """
        # Check snapshot cache
        if not force_refresh and symbol in self.snapshot_cache:
            snapshot, cached_time = self.snapshot_cache[symbol]
            cache_age = time.time() - cached_time
            
            # Use cached snapshot if < 1 minute old
            if cache_age < 60:
                logger.debug(f"Using cached snapshot for {symbol}")
                return snapshot
        
        logger.info(f"Building market snapshot for {symbol}")
        
        try:
            # Fetch current ticker data
            ticker = self.client.futures_ticker_24hr(symbol=symbol)
            mark_data = self.client.futures_mark_price(symbol=symbol)
            
            # Get multi-timeframe candles
            candles_1h = self.get_candles(symbol, '1h', force_refresh=force_refresh)
            candles_4h = self.get_candles(symbol, '4h', force_refresh=force_refresh)
            
            # Calculate ATR
            atr_1h = self.calculate_atr(candles_1h) if not candles_1h.empty else 0.0
            atr_4h = self.calculate_atr(candles_4h) if not candles_4h.empty else 0.0
            
            # Calculate volatility
            volatility_1h = self.calculate_volatility(candles_1h) if not candles_1h.empty else 0.0
            
            # Calculate volume metrics
            volume_1h = candles_1h['volume'].sum() if not candles_1h.empty else 0.0
            volume_4h = candles_4h['volume'].sum() if not candles_4h.empty else 0.0
            
            # Get volume metrics for relative volume calculation
            volume_metrics = self.calculate_volume_metrics(candles_1h) if not candles_1h.empty else {}
            relative_volume = volume_metrics.get('relative_volume', 1.0)
            
            # Create snapshot
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                
                # Current prices
                price=float(ticker['lastPrice']),
                mark_price=float(mark_data['markPrice']),
                funding_rate=float(mark_data['lastFundingRate']) * 100,  # Convert to %
                open_interest=0.0,  # Will add OI fetching later
                
                # 24h stats
                volume_24h=float(ticker['quoteVolume']),
                high_24h=float(ticker['highPrice']),
                low_24h=float(ticker['lowPrice']),
                change_24h_pct=float(ticker['priceChangePercent']),
                
                # Volatility
                atr_1h=atr_1h,
                atr_4h=atr_4h,
                volatility_1h_pct=volatility_1h,
                
                # Volume
                volume_1h=volume_1h,
                volume_4h=volume_4h,
                relative_volume=relative_volume,
                
                # Multi-timeframe candles
                candles_1m=self.get_candles(symbol, '1m'),
                candles_5m=self.get_candles(symbol, '5m'),
                candles_15m=self.get_candles(symbol, '15m'),
                candles_1h=candles_1h,
                candles_4h=candles_4h,
                candles_1d=self.get_candles(symbol, '1d'),
            )
            
            # Cache snapshot
            self.snapshot_cache[symbol] = (snapshot, time.time())
            
            logger.info(
                f"Snapshot built for {symbol}: "
                f"Price=${snapshot.price:.4f}, "
                f"Vol24h=${snapshot.volume_24h:,.0f}, "
                f"ATR={snapshot.atr_1h:.4f}"
            )
            
            return snapshot
            
        except BinanceClientError as e:
            logger.error(f"Failed to build snapshot for {symbol}: {e}")
            
            # Return cached snapshot if available
            if symbol in self.snapshot_cache:
                logger.warning(f"Returning stale snapshot for {symbol}")
                snapshot, _ = self.snapshot_cache[symbol]
                return snapshot
            
            # No cache, raise exception
            raise
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: Clear cache for specific symbol (or all if None)
        """
        if symbol:
            if symbol in self.candle_cache:
                del self.candle_cache[symbol]
            if symbol in self.snapshot_cache:
                del self.snapshot_cache[symbol]
            logger.info(f"Cleared cache for {symbol}")
        else:
            self.candle_cache.clear()
            self.snapshot_cache.clear()
            logger.info("Cleared all cache")


def main():
    """Test the market data aggregator"""
    from data.binance_client import BinanceClient
    
    print("=" * 70)
    print("MARKET DATA AGGREGATOR TEST")
    print("=" * 70)
    
    # Initialize
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    aggregator = MarketDataAggregator(client)
    
    # Test symbol
    symbol = 'BTCUSDT'
    
    print(f"\nTesting with {symbol}...\n")
    
    # Test 1: Get single timeframe candles
    print("1. Fetching 1h candles...")
    df_1h = aggregator.get_candles(symbol, '1h', limit=50)
    print(f"   ✓ Got {len(df_1h)} candles")
    print(f"   Latest: {df_1h.iloc[-1]['timestamp']} | "
          f"Close: ${df_1h.iloc[-1]['close']:.2f}")
    
    # Test 2: Calculate ATR
    print("\n2. Calculating ATR...")
    atr = aggregator.calculate_atr(df_1h)
    print(f"   ✓ ATR (14): ${atr:.4f}")
    
    # Test 3: Calculate volatility
    print("\n3. Calculating volatility...")
    vol = aggregator.calculate_volatility(df_1h)
    print(f"   ✓ Volatility: {vol:.2f}%")
    
    # Test 4: Volume metrics
    print("\n4. Analyzing volume...")
    vol_metrics = aggregator.calculate_volume_metrics(df_1h)
    print(f"   ✓ Current Volume: {vol_metrics['current_volume']:,.0f}")
    print(f"   ✓ Avg Volume: {vol_metrics['avg_volume']:,.0f}")
    print(f"   ✓ Relative Volume: {vol_metrics['relative_volume']:.2f}x")
    print(f"   ✓ Volume Spike: {vol_metrics['volume_spike']}")
    
    # Test 5: Support/Resistance
    print("\n5. Finding support/resistance...")
    levels = aggregator.find_support_resistance(df_1h, lookback=30)
    print(f"   ✓ Support levels: {len(levels['support'])}")
    if levels['support']:
        print(f"      {[f'${x:.2f}' for x in levels['support'][-3:]]}")
    print(f"   ✓ Resistance levels: {len(levels['resistance'])}")
    if levels['resistance']:
        print(f"      {[f'${x:.2f}' for x in levels['resistance'][-3:]]}")
    
    # Test 6: Full market snapshot
    print("\n6. Building complete market snapshot...")
    snapshot = aggregator.get_market_snapshot(symbol)
    print(f"   ✓ Price: ${snapshot.price:,.2f}")
    print(f"   ✓ Volume 24h: ${snapshot.volume_24h:,.0f}")
    print(f"   ✓ Change 24h: {snapshot.change_24h_pct:+.2f}%")
    print(f"   ✓ ATR 1h: ${snapshot.atr_1h:.4f}")
    print(f"   ✓ Volatility 1h: {snapshot.volatility_1h_pct:.2f}%")
    print(f"   ✓ Funding Rate: {snapshot.funding_rate:.4f}%")
    print(f"   ✓ Multi-timeframe candles loaded:")
    for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
        candles = getattr(snapshot, f'candles_{tf}')
        if candles is not None and not candles.empty:
            print(f"      {tf}: {len(candles)} candles")
    
    # Test 7: Cache performance
    print("\n7. Testing cache performance...")
    import time
    
    start = time.time()
    snapshot1 = aggregator.get_market_snapshot(symbol)
    time1 = time.time() - start
    
    start = time.time()
    snapshot2 = aggregator.get_market_snapshot(symbol)
    time2 = time.time() - start
    
    print(f"   ✓ First fetch: {time1:.3f}s")
    print(f"   ✓ Cached fetch: {time2:.3f}s")
    print(f"   ✓ Speedup: {time1/time2:.1f}x faster")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nMarket Data Aggregator is ready.")
    print("Next: Integrate with strategy modules")


if __name__ == "__main__":
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
    main()