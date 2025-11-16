"""
SIMPLE MARKET DATA TEST - NO IDE BULLSHIT
==========================================
This test adds the paths explicitly and runs the test.
If this doesn't work, nothing will.
"""

import sys
import os

# Force add paths
project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'data'))
sys.path.insert(0, os.path.join(project_root, 'config'))

print("Python paths:")
for p in sys.path[:5]:
    print(f"  {p}")

print("\nAttempting imports...")

try:
    from config.settings import settings
    print("✓ Settings imported")
except Exception as e:
    print(f"✗ Settings import failed: {e}")
    sys.exit(1)

try:
    from data.binance_client import BinanceClient
    print("✓ BinanceClient imported")
except Exception as e:
    print(f"✗ BinanceClient import failed: {e}")
    sys.exit(1)

try:
    from data.market_data import MarketDataAggregator
    print("✓ MarketDataAggregator imported")
except Exception as e:
    print(f"✗ MarketDataAggregator import failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("RUNNING MARKET DATA TEST")
print("=" * 70)

# Initialize
client = BinanceClient(testnet=settings.BINANCE_TESTNET)
aggregator = MarketDataAggregator(client)

# Test with BTC
symbol = 'BTCUSDT'
print(f"\nTesting {symbol}...")

# Get candles
print("\n1. Fetching 1h candles...")
df = aggregator.get_candles(symbol, '1h', limit=20)
print(f"   ✓ Got {len(df)} candles")

# Calculate ATR
print("\n2. Calculating ATR...")
atr = aggregator.calculate_atr(df)
print(f"   ✓ ATR: ${atr:.2f}")

# Get snapshot
print("\n3. Building market snapshot...")
snapshot = aggregator.get_market_snapshot(symbol)
print(f"   ✓ Price: ${snapshot.price:,.2f}")
print(f"   ✓ Volume 24h: ${snapshot.volume_24h:,.0f}")
print(f"   ✓ Change: {snapshot.change_24h_pct:+.2f}%")
print(f"   ✓ ATR 1h: ${snapshot.atr_1h:.2f}")
print(f"   ✓ Funding: {snapshot.funding_rate:.4f}%")

print("\n" + "=" * 70)
print("✓ TEST PASSED")
print("=" * 70)
print("\nMarket Data Aggregator works. Move on to Module 3.")