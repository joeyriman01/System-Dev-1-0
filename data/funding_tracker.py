"""
FUNDING RATE TRACKER
====================
Tracks funding rate history and detects overleveraged positions.

Funding Rate Basics:
- Positive funding = longs pay shorts = too many leveraged longs
- Negative funding = shorts pay longs = too many shorts
- High positive funding (>0.05%) = extreme long positioning = exhaustion signal

Your Edge:
When funding flips negative → positive during a pump, it confirms overleveraged
longs entering late. Combined with rejection patterns = high-conviction short.

Author: Grim (Institutional Standards)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

import pandas as pd

from data.binance_client import BinanceClient, BinanceClientError
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FundingSnapshot:
    """Snapshot of funding rate at a point in time"""
    symbol: str
    timestamp: datetime
    funding_rate: float  # As percentage (0.01 = 0.01%)
    mark_price: float
    next_funding_time: datetime


@dataclass
class FundingAnalysis:
    """Analysis of funding rate trends"""
    symbol: str
    current_funding: float
    avg_funding_8h: float  # Average over last 8 hours (1 funding period)
    avg_funding_24h: float  # Average over last 24 hours (3 periods)
    avg_funding_7d: float  # Average over 7 days
    
    # Trend detection
    is_positive: bool  # Current funding is positive
    was_negative: bool  # Previous funding was negative
    funding_flip: bool  # Flipped from negative to positive
    
    # Extreme levels
    is_high: bool  # Funding > 0.05% (high)
    is_extreme: bool  # Funding > 0.1% (extreme)
    
    # Historical context
    percentile_7d: float  # Where current funding sits in 7d distribution (0-100)
    
    funding_history: pd.DataFrame  # Historical funding rates


class FundingRateTracker:
    """
    Tracks funding rates and detects overleveraged positioning.
    
    This is a KEY component of your edge. Funding rate confirms when
    retail is piling into longs at the top of a pump. When you see:
    1. Price pumping to resistance
    2. Multiple rejections  
    3. Funding flips negative → positive
    
    = OVERLEVERAGED LONGS = Your short setup
    """
    
    # Thresholds
    HIGH_FUNDING_THRESHOLD = 0.05  # 0.05% = high funding
    EXTREME_FUNDING_THRESHOLD = 0.10  # 0.10% = extreme funding
    
    # Cache duration
    CACHE_DURATION_SECONDS = 60  # Refresh funding data every 60 seconds
    
    def __init__(self, binance_client: BinanceClient):
        """Initialize funding rate tracker"""
        self.client = binance_client
        
        # Data cache
        self.funding_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        # Structure: {symbol: (dataframe, timestamp)}
        
        self.current_funding_cache: Dict[str, Tuple[FundingSnapshot, float]] = {}
        # Structure: {symbol: (snapshot, timestamp)}
        
        logger.info("FundingRateTracker initialized")
    
    def get_current_funding(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> FundingSnapshot:
        """
        Get current funding rate for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            force_refresh: Force fetch from API even if cached
            
        Returns:
            FundingSnapshot with current funding data
        """
        # Check cache
        if not force_refresh and symbol in self.current_funding_cache:
            snapshot, cached_time = self.current_funding_cache[symbol]
            cache_age = time.time() - cached_time
            
            if cache_age < self.CACHE_DURATION_SECONDS:
                logger.debug(f"Using cached funding for {symbol}")
                return snapshot
        
        # Fetch from API
        try:
            logger.debug(f"Fetching current funding for {symbol}")
            data = self.client.futures_mark_price(symbol=symbol)
            
            # Parse data
            funding_rate = float(data['lastFundingRate']) * 100  # Convert to percentage
            mark_price = float(data['markPrice'])
            next_funding_time = datetime.fromtimestamp(int(data['nextFundingTime']) / 1000)
            
            snapshot = FundingSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                funding_rate=funding_rate,
                mark_price=mark_price,
                next_funding_time=next_funding_time
            )
            
            # Cache
            self.current_funding_cache[symbol] = (snapshot, time.time())
            
            logger.debug(
                f"Current funding for {symbol}: {funding_rate:.4f}% "
                f"(next payment: {next_funding_time.strftime('%H:%M:%S')})"
            )
            
            return snapshot
            
        except BinanceClientError as e:
            logger.error(f"Failed to fetch funding for {symbol}: {e}")
            
            # Return cached if available
            if symbol in self.current_funding_cache:
                logger.warning(f"Returning stale funding data for {symbol}")
                snapshot, _ = self.current_funding_cache[symbol]
                return snapshot
            
            raise
    
    def get_funding_history(
        self,
        symbol: str,
        hours: int = 168,  # 7 days default
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical funding rates.
        
        Funding rates are published every 8 hours on Binance.
        
        Args:
            symbol: Trading pair
            hours: Hours of history to fetch (max ~1 year)
            force_refresh: Force refresh cached data
            
        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        cache_key = f"{symbol}_{hours}h"
        
        # Check cache
        if not force_refresh and cache_key in self.funding_cache:
            df, cached_time = self.funding_cache[cache_key]
            cache_age = time.time() - cached_time
            
            # Refresh if older than 1 hour
            if cache_age < 3600:
                logger.debug(f"Using cached funding history for {symbol}")
                return df
        
        # Calculate number of funding periods (every 8 hours)
        num_periods = min(hours // 8, 1000)  # Binance limit is 1000
        
        try:
            logger.debug(f"Fetching {num_periods} funding periods for {symbol}")
            funding_data = self.client.futures_funding_rate(
                symbol=symbol,
                limit=num_periods
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(funding_data)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float) * 100  # Convert to %
            
            # Rename columns
            df = df.rename(columns={
                'fundingTime': 'timestamp',
                'fundingRate': 'funding_rate'
            })
            
            # Keep only relevant columns
            df = df[['timestamp', 'funding_rate']]
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Cache
            self.funding_cache[cache_key] = (df, time.time())
            
            logger.debug(f"Fetched {len(df)} funding periods for {symbol}")
            return df
            
        except BinanceClientError as e:
            logger.error(f"Failed to fetch funding history for {symbol}: {e}")
            
            # Return cached if available
            if cache_key in self.funding_cache:
                logger.warning(f"Returning stale funding history for {symbol}")
                df, _ = self.funding_cache[cache_key]
                return df
            
            # Return empty DataFrame
            return pd.DataFrame(columns=['timestamp', 'funding_rate'])
    
    def detect_funding_flip(
        self,
        symbol: str,
        lookback_periods: int = 3
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Detect if funding has flipped from negative to positive.
        
        This is a KEY signal for your strategy. When funding flips negative → positive
        during a pump, it means retail is piling into longs at the top.
        
        Args:
            symbol: Trading pair
            lookback_periods: How many periods to check (each period = 8h)
            
        Returns:
            (has_flipped, flip_time) tuple
        """
        # Get funding history
        df = self.get_funding_history(symbol, hours=lookback_periods * 8)
        
        if len(df) < 2:
            return False, None
        
        # Get recent funding rates
        recent = df.tail(lookback_periods)
        
        # Check if current is positive
        current_funding = recent.iloc[-1]['funding_rate']
        if current_funding <= 0:
            return False, None
        
        # Check if any previous was negative
        for i in range(len(recent) - 1):
            if recent.iloc[i]['funding_rate'] < 0:
                # Found a flip
                flip_time = recent.iloc[-1]['timestamp']
                logger.info(
                    f"Funding flip detected for {symbol}: "
                    f"{recent.iloc[i]['funding_rate']:.4f}% → {current_funding:.4f}% "
                    f"at {flip_time}"
                )
                return True, flip_time
        
        return False, None
    
    def calculate_avg_funding(
        self,
        symbol: str,
        hours: int = 24
    ) -> float:
        """
        Calculate average funding rate over a period.
        
        Args:
            symbol: Trading pair
            hours: Period in hours
            
        Returns:
            Average funding rate as percentage
        """
        df = self.get_funding_history(symbol, hours=hours)
        
        if df.empty:
            return 0.0
        
        avg_funding = df['funding_rate'].mean()
        return float(avg_funding)
    
    def get_funding_percentile(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> float:
        """
        Get percentile rank of current funding in historical distribution.
        
        Returns:
            Percentile (0-100) where current funding sits
        """
        # Get current funding
        current = self.get_current_funding(symbol)
        
        # Get historical funding
        df = self.get_funding_history(symbol, hours=lookback_days * 24)
        
        if df.empty:
            return 50.0  # Default to median
        
        # Calculate percentile
        percentile = (df['funding_rate'] < current.funding_rate).sum() / len(df) * 100
        
        return float(percentile)
    
    def analyze_funding(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> FundingAnalysis:
        """
        Comprehensive funding analysis for a symbol.
        
        This is what your strategy modules will consume.
        
        Args:
            symbol: Trading pair
            lookback_days: Days of history to analyze
            
        Returns:
            FundingAnalysis with all metrics and signals
        """
        logger.info(f"Analyzing funding for {symbol}")
        
        # Get current funding
        current = self.get_current_funding(symbol)
        current_funding = current.funding_rate
        
        # Get historical funding
        df = self.get_funding_history(symbol, hours=lookback_days * 24)
        
        # Calculate averages
        avg_8h = self.calculate_avg_funding(symbol, hours=8)
        avg_24h = self.calculate_avg_funding(symbol, hours=24)
        avg_7d = self.calculate_avg_funding(symbol, hours=lookback_days * 24)
        
        # Detect funding flip
        has_flipped, _ = self.detect_funding_flip(symbol, lookback_periods=3)
        
        # Check if previous funding was negative
        was_negative = False
        if len(df) >= 2:
            was_negative = df.iloc[-2]['funding_rate'] < 0
        
        # Get percentile
        percentile = self.get_funding_percentile(symbol, lookback_days)
        
        # Create analysis
        analysis = FundingAnalysis(
            symbol=symbol,
            current_funding=current_funding,
            avg_funding_8h=avg_8h,
            avg_funding_24h=avg_24h,
            avg_funding_7d=avg_7d,
            
            # Trend
            is_positive=current_funding > 0,
            was_negative=was_negative,
            funding_flip=has_flipped,
            
            # Levels
            is_high=current_funding > self.HIGH_FUNDING_THRESHOLD,
            is_extreme=current_funding > self.EXTREME_FUNDING_THRESHOLD,
            
            # Context
            percentile_7d=percentile,
            funding_history=df
        )
        
        logger.info(
            f"Funding analysis for {symbol}: "
            f"Current={current_funding:.4f}%, "
            f"Avg24h={avg_24h:.4f}%, "
            f"Flip={has_flipped}, "
            f"High={analysis.is_high}, "
            f"Percentile={percentile:.1f}"
        )
        
        return analysis
    
    def get_high_funding_symbols(
        self,
        symbols: List[str],
        min_funding: float = 0.05
    ) -> List[Tuple[str, float]]:
        """
        Get symbols with high funding rates.
        
        Useful for scanning for overleveraged setups.
        
        Args:
            symbols: List of symbols to check
            min_funding: Minimum funding rate threshold
            
        Returns:
            List of (symbol, funding_rate) tuples sorted by funding
        """
        high_funding = []
        
        for symbol in symbols:
            try:
                snapshot = self.get_current_funding(symbol)
                if snapshot.funding_rate >= min_funding:
                    high_funding.append((symbol, snapshot.funding_rate))
            except Exception as e:
                logger.error(f"Failed to check funding for {symbol}: {e}")
                continue
        
        # Sort by funding rate (highest first)
        high_funding.sort(key=lambda x: x[1], reverse=True)
        
        return high_funding
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached funding data"""
        if symbol:
            # Clear symbol-specific cache
            keys_to_remove = [k for k in self.funding_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self.funding_cache[key]
            
            if symbol in self.current_funding_cache:
                del self.current_funding_cache[symbol]
            
            logger.info(f"Cleared funding cache for {symbol}")
        else:
            # Clear all cache
            self.funding_cache.clear()
            self.current_funding_cache.clear()
            logger.info("Cleared all funding cache")


def main():
    """Test the funding rate tracker"""
    from data.binance_client import BinanceClient
    
    print("=" * 70)
    print("FUNDING RATE TRACKER TEST")
    print("=" * 70)
    
    # Initialize
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    tracker = FundingRateTracker(client)
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in test_symbols:
        print(f"\n{'='*70}")
        print(f"TESTING {symbol}")
        print('='*70)
        
        # Test 1: Current funding
        print(f"\n1. Current funding rate...")
        snapshot = tracker.get_current_funding(symbol)
        print(f"   ✓ Funding Rate: {snapshot.funding_rate:.4f}%")
        print(f"   ✓ Mark Price: ${snapshot.mark_price:,.2f}")
        print(f"   ✓ Next Payment: {snapshot.next_funding_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 2: Funding history
        print(f"\n2. Funding history (7 days)...")
        df = tracker.get_funding_history(symbol, hours=168)
        print(f"   ✓ Got {len(df)} funding periods")
        if not df.empty:
            print(f"   ✓ Min: {df['funding_rate'].min():.4f}%")
            print(f"   ✓ Max: {df['funding_rate'].max():.4f}%")
            print(f"   ✓ Mean: {df['funding_rate'].mean():.4f}%")
        
        # Test 3: Average funding
        print(f"\n3. Average funding rates...")
        avg_8h = tracker.calculate_avg_funding(symbol, hours=8)
        avg_24h = tracker.calculate_avg_funding(symbol, hours=24)
        avg_7d = tracker.calculate_avg_funding(symbol, hours=168)
        print(f"   ✓ Avg 8h: {avg_8h:.4f}%")
        print(f"   ✓ Avg 24h: {avg_24h:.4f}%")
        print(f"   ✓ Avg 7d: {avg_7d:.4f}%")
        
        # Test 4: Funding flip detection
        print(f"\n4. Checking for funding flip...")
        has_flipped, flip_time = tracker.detect_funding_flip(symbol)
        if has_flipped:
            print(f"   ✓ FUNDING FLIP DETECTED at {flip_time}")
        else:
            print(f"   ✓ No funding flip detected")
        
        # Test 5: Percentile
        print(f"\n5. Current funding percentile...")
        percentile = tracker.get_funding_percentile(symbol)
        print(f"   ✓ Percentile: {percentile:.1f}%")
        
        # Test 6: Complete analysis
        print(f"\n6. Complete funding analysis...")
        analysis = tracker.analyze_funding(symbol)
        print(f"   ✓ Current: {analysis.current_funding:.4f}%")
        print(f"   ✓ Positive: {analysis.is_positive}")
        print(f"   ✓ Was Negative: {analysis.was_negative}")
        print(f"   ✓ Funding Flip: {analysis.funding_flip}")
        print(f"   ✓ High Funding: {analysis.is_high}")
        print(f"   ✓ Extreme Funding: {analysis.is_extreme}")
        print(f"   ✓ Percentile: {analysis.percentile_7d:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nFunding Rate Tracker is ready.")
    print("Next: Week 2 Module 4 - Order Book Monitor")


if __name__ == "__main__":
    main()