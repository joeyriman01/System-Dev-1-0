<<<<<<< HEAD
"""
ORDER BOOK MONITOR TEST
=======================
Simple test for the order book monitor module.
"""

import sys
import os

# Add project root to path
project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
sys.path.insert(0, project_root)

from data.binance_client import BinanceClient
from data.orderbook_monitor import OrderBookMonitor
from config.settings import settings

print("=" * 70)
print("ORDER BOOK MONITOR TEST")
print("=" * 70)

# Initialize
client = BinanceClient(testnet=settings.BINANCE_TESTNET)
monitor = OrderBookMonitor(client)

# Test symbol
symbol = 'BTCUSDT'
print(f"\nTesting {symbol}...\n")

# Test 1: Get order book
print("1. Fetching order book...")
snapshot = monitor.get_orderbook(symbol, depth=50)
print(f"   ✓ Mid Price: ${snapshot.mid_price:,.2f}")
print(f"   ✓ Spread: {snapshot.spread_bps:.1f} bps")

# Test 2: Depth analysis
print(f"\n2. Analyzing depth...")
print(f"   ✓ Bid Depth: ${snapshot.bid_depth_usdt:,.0f}")
print(f"   ✓ Ask Depth: ${snapshot.ask_depth_usdt:,.0f}")
print(f"   ✓ Imbalance: {snapshot.imbalance:+.2f}")
print(f"   ✓ Pressure: {snapshot.pressure}")

# Test 3: Walls
print(f"\n3. Finding walls...")
print(f"   ✓ Bid Walls: {len(snapshot.bid_walls)}")
print(f"   ✓ Ask Walls: {len(snapshot.ask_walls)}")

# Test 4: Pressure score
print(f"\n4. Pressure score...")
pressure_score = monitor.get_pressure_score(symbol)
print(f"   ✓ Score: {pressure_score:+.1f}/100")

# Test 5: Resistance analysis
print(f"\n5. Analyzing resistance zone...")
resistance = snapshot.mid_price * 1.02  # 2% above
analysis = monitor.analyze_resistance_zone(symbol, resistance, zone_width_pct=0.5)
print(f"   ✓ Distance: {analysis['distance_to_zone_pct']:+.2f}%")
print(f"   ✓ Sell/Buy Ratio: {analysis['sell_buy_ratio']:.2f}x")
print(f"   ✓ Heavy Resistance: {analysis['heavy_resistance']}")

print("\n" + "=" * 70)
print("✓ TEST PASSED")
print("=" * 70)
print("\n🎉 WEEK 2 DATA COLLECTION: COMPLETE")
print("\nYou now have:")
print("  ✓ New Listing Scanner")
print("  ✓ Market Data Aggregator") 
print("  ✓ Funding Rate Tracker")
print("  ✓ Order Book Monitor")
=======
"""
ORDER BOOK MONITOR TEST
=======================
Simple test for the order book monitor module.
"""

import sys
import os

# Add project root to path
project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
sys.path.insert(0, project_root)

from data.binance_client import BinanceClient
from data.orderbook_monitor import OrderBookMonitor
from config.settings import settings

print("=" * 70)
print("ORDER BOOK MONITOR TEST")
print("=" * 70)

# Initialize
client = BinanceClient(testnet=settings.BINANCE_TESTNET)
monitor = OrderBookMonitor(client)

# Test symbol
symbol = 'BTCUSDT'
print(f"\nTesting {symbol}...\n")

# Test 1: Get order book
print("1. Fetching order book...")
snapshot = monitor.get_orderbook(symbol, depth=50)
print(f"   ✓ Mid Price: ${snapshot.mid_price:,.2f}")
print(f"   ✓ Spread: {snapshot.spread_bps:.1f} bps")

# Test 2: Depth analysis
print(f"\n2. Analyzing depth...")
print(f"   ✓ Bid Depth: ${snapshot.bid_depth_usdt:,.0f}")
print(f"   ✓ Ask Depth: ${snapshot.ask_depth_usdt:,.0f}")
print(f"   ✓ Imbalance: {snapshot.imbalance:+.2f}")
print(f"   ✓ Pressure: {snapshot.pressure}")

# Test 3: Walls
print(f"\n3. Finding walls...")
print(f"   ✓ Bid Walls: {len(snapshot.bid_walls)}")
print(f"   ✓ Ask Walls: {len(snapshot.ask_walls)}")

# Test 4: Pressure score
print(f"\n4. Pressure score...")
pressure_score = monitor.get_pressure_score(symbol)
print(f"   ✓ Score: {pressure_score:+.1f}/100")

# Test 5: Resistance analysis
print(f"\n5. Analyzing resistance zone...")
resistance = snapshot.mid_price * 1.02  # 2% above
analysis = monitor.analyze_resistance_zone(symbol, resistance, zone_width_pct=0.5)
print(f"   ✓ Distance: {analysis['distance_to_zone_pct']:+.2f}%")
print(f"   ✓ Sell/Buy Ratio: {analysis['sell_buy_ratio']:.2f}x")
print(f"   ✓ Heavy Resistance: {analysis['heavy_resistance']}")

print("\n" + "=" * 70)
print("✓ TEST PASSED")
print("=" * 70)
print("\n🎉 WEEK 2 DATA COLLECTION: COMPLETE")
print("\nYou now have:")
print("  ✓ New Listing Scanner")
print("  ✓ Market Data Aggregator") 
print("  ✓ Funding Rate Tracker")
print("  ✓ Order Book Monitor")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
print("\nNext: Week 3 - Strategy Logic (rejection detector, signal generator)")