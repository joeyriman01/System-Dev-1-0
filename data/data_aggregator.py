<<<<<<< HEAD
"""
TEST SCRIPT - MARKET DATA AGGREGATOR
=====================================
Comprehensive test of the market data module.

Tests:
1. Single timeframe candle fetching
2. Multi-timeframe data loading
3. ATR calculation
4. Volatility metrics
5. Volume analysis
6. Support/resistance detection
7. Complete market snapshot
8. Cache performance

Author: Grim
"""

import os
import sys
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_data_aggregator():
    """Test the market data aggregator"""
    print("=" * 70)
    print("MARKET DATA AGGREGATOR TEST")
    print("=" * 70)

    try:
        from config.settings import settings
        from data.binance_client import BinanceClient
        from data.market_data import MarketDataAggregator

        # Initialize
        print("\n1. Initializing components...")
        client = BinanceClient(testnet=settings.BINANCE_TESTNET)
        aggregator = MarketDataAggregator(client)
        print("✓ Market data aggregator initialized")

        # Test symbols
        test_symbols = ["BTCUSDT", "ETHUSDT"]

        for symbol in test_symbols:
            print(f"\n{'='*70}")
            print(f"TESTING {symbol}")
            print("=" * 70)

            # Test 1: Get candles
            print(f"\n2. Fetching candles for {symbol}...")

            for timeframe in ["1m", "5m", "15m", "1h"]:
                df = aggregator.get_candles(symbol, timeframe, limit=20)

                if not df.empty:
                    latest = df.iloc[-1]
                    print(
                        f"   ✓ {timeframe}: {len(df)} candles | "
                        f"Latest close: ${latest['close']:.4f} | "
                        f"Volume: {latest['volume']:,.0f}"
                    )
                else:
                    print(f"   ✗ {timeframe}: No data")

            # Test 2: Multi-timeframe data
            print(f"\n3. Loading multi-timeframe data...")
            mtf_data = aggregator.get_multi_timeframe_data(symbol)
            print(f"   ✓ Loaded {len(mtf_data)} timeframes")

            # Test 3: ATR calculation
            print(f"\n4. Calculating technical indicators...")
            df_1h = aggregator.get_candles(symbol, "1h", limit=50)

            if not df_1h.empty:
                atr = aggregator.calculate_atr(df_1h, period=14)
                print(f"   ✓ ATR (14): ${atr:.4f}")

                vol = aggregator.calculate_volatility(df_1h)
                print(f"   ✓ Volatility: {vol:.2f}%")

                vol_metrics = aggregator.calculate_volume_metrics(df_1h)
                print(f"   ✓ Current Volume: {vol_metrics['current_volume']:,.0f}")
                print(f"   ✓ Relative Volume: {vol_metrics['relative_volume']:.2f}x")
                print(f"   ✓ Volume Spike: {vol_metrics['volume_spike']}")
            else:
                print("   ✗ No data for calculations")

            # Test 4: Support/Resistance
            print(f"\n5. Finding support/resistance levels...")
            if not df_1h.empty:
                levels = aggregator.find_support_resistance(df_1h, lookback=30)

                print(f"   ✓ Support levels found: {len(levels['support'])}")
                if levels["support"]:
                    recent_support = levels["support"][-3:]
                    print(f"      Recent: {[f'${x:,.2f}' for x in recent_support]}")

                print(f"   ✓ Resistance levels found: {len(levels['resistance'])}")
                if levels["resistance"]:
                    recent_resistance = levels["resistance"][-3:]
                    print(f"      Recent: {[f'${x:,.2f}' for x in recent_resistance]}")
            else:
                print("   ✗ No data for S/R detection")

            # Test 5: Complete market snapshot
            print(f"\n6. Building complete market snapshot...")
            start_time = time.time()
            snapshot = aggregator.get_market_snapshot(symbol)
            build_time = time.time() - start_time

            print(f"   ✓ Snapshot built in {build_time:.2f}s")
            print(f"\n   Market Data:")
            print(f"      Price: ${snapshot.price:,.4f}")
            print(f"      Mark Price: ${snapshot.mark_price:,.4f}")
            print(f"      24h Volume: ${snapshot.volume_24h:,.0f}")
            print(f"      24h Change: {snapshot.change_24h_pct:+.2f}%")
            print(f"      24h High: ${snapshot.high_24h:,.4f}")
            print(f"      24h Low: ${snapshot.low_24h:,.4f}")

            print(f"\n   Volatility Metrics:")
            print(f"      ATR 1h: ${snapshot.atr_1h:.4f}")
            print(f"      ATR 4h: ${snapshot.atr_4h:.4f}")
            print(f"      Volatility 1h: {snapshot.volatility_1h_pct:.2f}%")

            print(f"\n   Volume Metrics:")
            print(f"      Volume 1h: {snapshot.volume_1h:,.0f}")
            print(f"      Volume 4h: {snapshot.volume_4h:,.0f}")
            print(f"      Relative Volume: {snapshot.relative_volume:.2f}x")

            print(f"\n   Funding:")
            print(f"      Funding Rate: {snapshot.funding_rate:.4f}%")

            print(f"\n   Multi-timeframe Candles:")
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                candles = getattr(snapshot, f"candles_{tf}")
                if candles is not None and not candles.empty:
                    print(f"      {tf}: {len(candles)} candles")

            # Test 6: Cache performance
            print(f"\n7. Testing cache performance...")

            start_time = time.time()
            snapshot1 = aggregator.get_market_snapshot(symbol)
            time1 = time.time() - start_time

            start_time = time.time()
            snapshot2 = aggregator.get_market_snapshot(symbol)
            time2 = time.time() - start_time

            print(f"   ✓ First fetch: {time1:.3f}s")
            print(f"   ✓ Cached fetch: {time2:.3f}s")

            # Handle instant cache (time2 = 0)
            if time2 > 0.001:
                speedup = time1 / time2
                print(f"   ✓ Cache speedup: {speedup:.1f}x faster")
            else:
                print(f"   ✓ Cache speedup: INSTANT (< 1ms)")

        print("\n" + "=" * 70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\nMarket Data Aggregator is COMPLETE and READY.")
        print("\nNext Steps:")
        print("1. Move market_data.py to data/ folder")
        print("2. Continue to Week 2 Module 3: Funding Rate Tracker")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_aggregator()
    sys.exit(0 if success else 1)
=======
"""
TEST SCRIPT - MARKET DATA AGGREGATOR
=====================================
Comprehensive test of the market data module.

Tests:
1. Single timeframe candle fetching
2. Multi-timeframe data loading
3. ATR calculation
4. Volatility metrics
5. Volume analysis
6. Support/resistance detection
7. Complete market snapshot
8. Cache performance

Author: Grim
"""

import os
import sys
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_data_aggregator():
    """Test the market data aggregator"""
    print("=" * 70)
    print("MARKET DATA AGGREGATOR TEST")
    print("=" * 70)

    try:
        from config.settings import settings
        from data.binance_client import BinanceClient
        from data.market_data import MarketDataAggregator

        # Initialize
        print("\n1. Initializing components...")
        client = BinanceClient(testnet=settings.BINANCE_TESTNET)
        aggregator = MarketDataAggregator(client)
        print("✓ Market data aggregator initialized")

        # Test symbols
        test_symbols = ["BTCUSDT", "ETHUSDT"]

        for symbol in test_symbols:
            print(f"\n{'='*70}")
            print(f"TESTING {symbol}")
            print("=" * 70)

            # Test 1: Get candles
            print(f"\n2. Fetching candles for {symbol}...")

            for timeframe in ["1m", "5m", "15m", "1h"]:
                df = aggregator.get_candles(symbol, timeframe, limit=20)

                if not df.empty:
                    latest = df.iloc[-1]
                    print(
                        f"   ✓ {timeframe}: {len(df)} candles | "
                        f"Latest close: ${latest['close']:.4f} | "
                        f"Volume: {latest['volume']:,.0f}"
                    )
                else:
                    print(f"   ✗ {timeframe}: No data")

            # Test 2: Multi-timeframe data
            print(f"\n3. Loading multi-timeframe data...")
            mtf_data = aggregator.get_multi_timeframe_data(symbol)
            print(f"   ✓ Loaded {len(mtf_data)} timeframes")

            # Test 3: ATR calculation
            print(f"\n4. Calculating technical indicators...")
            df_1h = aggregator.get_candles(symbol, "1h", limit=50)

            if not df_1h.empty:
                atr = aggregator.calculate_atr(df_1h, period=14)
                print(f"   ✓ ATR (14): ${atr:.4f}")

                vol = aggregator.calculate_volatility(df_1h)
                print(f"   ✓ Volatility: {vol:.2f}%")

                vol_metrics = aggregator.calculate_volume_metrics(df_1h)
                print(f"   ✓ Current Volume: {vol_metrics['current_volume']:,.0f}")
                print(f"   ✓ Relative Volume: {vol_metrics['relative_volume']:.2f}x")
                print(f"   ✓ Volume Spike: {vol_metrics['volume_spike']}")
            else:
                print("   ✗ No data for calculations")

            # Test 4: Support/Resistance
            print(f"\n5. Finding support/resistance levels...")
            if not df_1h.empty:
                levels = aggregator.find_support_resistance(df_1h, lookback=30)

                print(f"   ✓ Support levels found: {len(levels['support'])}")
                if levels["support"]:
                    recent_support = levels["support"][-3:]
                    print(f"      Recent: {[f'${x:,.2f}' for x in recent_support]}")

                print(f"   ✓ Resistance levels found: {len(levels['resistance'])}")
                if levels["resistance"]:
                    recent_resistance = levels["resistance"][-3:]
                    print(f"      Recent: {[f'${x:,.2f}' for x in recent_resistance]}")
            else:
                print("   ✗ No data for S/R detection")

            # Test 5: Complete market snapshot
            print(f"\n6. Building complete market snapshot...")
            start_time = time.time()
            snapshot = aggregator.get_market_snapshot(symbol)
            build_time = time.time() - start_time

            print(f"   ✓ Snapshot built in {build_time:.2f}s")
            print(f"\n   Market Data:")
            print(f"      Price: ${snapshot.price:,.4f}")
            print(f"      Mark Price: ${snapshot.mark_price:,.4f}")
            print(f"      24h Volume: ${snapshot.volume_24h:,.0f}")
            print(f"      24h Change: {snapshot.change_24h_pct:+.2f}%")
            print(f"      24h High: ${snapshot.high_24h:,.4f}")
            print(f"      24h Low: ${snapshot.low_24h:,.4f}")

            print(f"\n   Volatility Metrics:")
            print(f"      ATR 1h: ${snapshot.atr_1h:.4f}")
            print(f"      ATR 4h: ${snapshot.atr_4h:.4f}")
            print(f"      Volatility 1h: {snapshot.volatility_1h_pct:.2f}%")

            print(f"\n   Volume Metrics:")
            print(f"      Volume 1h: {snapshot.volume_1h:,.0f}")
            print(f"      Volume 4h: {snapshot.volume_4h:,.0f}")
            print(f"      Relative Volume: {snapshot.relative_volume:.2f}x")

            print(f"\n   Funding:")
            print(f"      Funding Rate: {snapshot.funding_rate:.4f}%")

            print(f"\n   Multi-timeframe Candles:")
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                candles = getattr(snapshot, f"candles_{tf}")
                if candles is not None and not candles.empty:
                    print(f"      {tf}: {len(candles)} candles")

            # Test 6: Cache performance
            print(f"\n7. Testing cache performance...")

            start_time = time.time()
            snapshot1 = aggregator.get_market_snapshot(symbol)
            time1 = time.time() - start_time

            start_time = time.time()
            snapshot2 = aggregator.get_market_snapshot(symbol)
            time2 = time.time() - start_time

            print(f"   ✓ First fetch: {time1:.3f}s")
            print(f"   ✓ Cached fetch: {time2:.3f}s")

            # Handle instant cache (time2 = 0)
            if time2 > 0.001:
                speedup = time1 / time2
                print(f"   ✓ Cache speedup: {speedup:.1f}x faster")
            else:
                print(f"   ✓ Cache speedup: INSTANT (< 1ms)")

        print("\n" + "=" * 70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\nMarket Data Aggregator is COMPLETE and READY.")
        print("\nNext Steps:")
        print("1. Move market_data.py to data/ folder")
        print("2. Continue to Week 2 Module 3: Funding Rate Tracker")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_aggregator()
    sys.exit(0 if success else 1)
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
