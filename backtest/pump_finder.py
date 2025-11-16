<<<<<<< HEAD
"""
PUMP FINDER
===========
Scans recent meme coin data to find specific pump events (not long-term trends).

This identifies SHORT-TERM pumps that fit your "3 pumps rule" profile:
- 50%+ price increase
- Within 24-48 hours
- High volume spike

These are the events we should validate the detector against.
"""

import sys
from datetime import datetime, timedelta

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.data_collector import HistoricalDataCollector
from data.binance_client import BinanceClient

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MEME COIN PUMP FINDER")
    print("=" * 80 + "\n")

    client = BinanceClient()
    collector = HistoricalDataCollector(client)

    # Valid meme coins from our symbol check
    # Plus other high-volatility altcoins that pump
    meme_coins = [
        "DOGEUSDT",
        "1000PEPEUSDT",
        "1000SHIBUSDT",
        "WIFUSDT",
        "BOMEUSDT",
        "MEMEUSDT",
        "1000FLOKIUSDT",
        "1000BONKUSDT",
        "ZECUSDT",  # ZCASH - added per user request
    ]

    # Scan last 90 days (3 months)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)

    print(
        f"Scanning {len(meme_coins)} meme coins from {start_time.date()} to {end_time.date()}\n"
    )
    print(f"Looking for pumps: 40%+ price increase within 24 candles (1h timeframe)\n")

    all_pumps = []

    for symbol in meme_coins:
        print(f"Fetching {symbol}...")

        try:
            # Fetch 1h data
            df = collector.fetch_historical_klines(
                symbol=symbol,
                interval="1h",
                start_time=start_time,
                end_time=end_time,
                save_to_file=False,  # Don't save, just analyze
            )

            if df.empty:
                print(f"  No data\n")
                continue

            # Find pumps (40%+ in 24h = 24 candles on 1h chart)
            # Lowered threshold from 50% to 40% to catch more pumps
            pumps = collector.identify_pump_periods(
                df,
                min_pump_pct=40.0,  # 40%+ pump (lowered from 50%)
                lookback_candles=24,  # Within 24 hours
            )

            if pumps:
                print(f"  ✓ Found {len(pumps)} pump(s)")
                for pump in pumps:
                    duration = (pump.end_time - pump.start_time).total_seconds() / 3600
                    print(
                        f"    • {pump.start_time.date()} to {pump.end_time.date()}: "
                        f"+{pump.price_change_pct:.1f}% in {duration:.1f}h "
                        f"(volume {pump.volume_spike:.1f}x)"
                    )
                    all_pumps.append(pump)
            else:
                print(f"  No significant pumps")

            print()

        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    print("\n" + "=" * 80)
    print(f"TOTAL PUMPS FOUND: {len(all_pumps)}")
    print("=" * 80 + "\n")

    if all_pumps:
        # Export to CSV
        collector.export_pump_dataset(all_pumps, "recent_meme_pumps.csv")
        print(f"✓ Exported to: {collector.data_dir / 'recent_meme_pumps.csv'}")

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Review the CSV file with pump details")
        print("2. Pick 3-5 specific pumps to validate")
        print("3. Update pattern_validator test cases with exact dates")
        print("4. Re-run validation on SPECIFIC pump events")
        print("\nExample test case format:")
        print("{")
        print("    'symbol': 'WIFUSDT',")
        print("    'start_date': datetime(2024, 10, 28, 0, 0),  # Exact pump start")
        print("    'end_date': datetime(2024, 10, 29, 0, 0),    # Exact pump end")
        print("    'is_good_setup': True,")
        print("}")
        print()
    else:
        print("No pumps found in the last 30 days.")
        print("\nTry:")
        print("1. Extending date range (60-90 days)")
        print("2. Lowering pump threshold (30%+ instead of 50%+)")
        print("3. Looking at different timeframes")
        print()
=======
"""
PUMP FINDER
===========
Scans recent meme coin data to find specific pump events (not long-term trends).

This identifies SHORT-TERM pumps that fit your "3 pumps rule" profile:
- 50%+ price increase
- Within 24-48 hours
- High volume spike

These are the events we should validate the detector against.
"""

import sys
from datetime import datetime, timedelta

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.data_collector import HistoricalDataCollector
from data.binance_client import BinanceClient

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MEME COIN PUMP FINDER")
    print("=" * 80 + "\n")

    client = BinanceClient()
    collector = HistoricalDataCollector(client)

    # Valid meme coins from our symbol check
    # Plus other high-volatility altcoins that pump
    meme_coins = [
        "DOGEUSDT",
        "1000PEPEUSDT",
        "1000SHIBUSDT",
        "WIFUSDT",
        "BOMEUSDT",
        "MEMEUSDT",
        "1000FLOKIUSDT",
        "1000BONKUSDT",
        "ZECUSDT",  # ZCASH - added per user request
    ]

    # Scan last 90 days (3 months)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)

    print(
        f"Scanning {len(meme_coins)} meme coins from {start_time.date()} to {end_time.date()}\n"
    )
    print(f"Looking for pumps: 40%+ price increase within 24 candles (1h timeframe)\n")

    all_pumps = []

    for symbol in meme_coins:
        print(f"Fetching {symbol}...")

        try:
            # Fetch 1h data
            df = collector.fetch_historical_klines(
                symbol=symbol,
                interval="1h",
                start_time=start_time,
                end_time=end_time,
                save_to_file=False,  # Don't save, just analyze
            )

            if df.empty:
                print(f"  No data\n")
                continue

            # Find pumps (40%+ in 24h = 24 candles on 1h chart)
            # Lowered threshold from 50% to 40% to catch more pumps
            pumps = collector.identify_pump_periods(
                df,
                min_pump_pct=40.0,  # 40%+ pump (lowered from 50%)
                lookback_candles=24,  # Within 24 hours
            )

            if pumps:
                print(f"  ✓ Found {len(pumps)} pump(s)")
                for pump in pumps:
                    duration = (pump.end_time - pump.start_time).total_seconds() / 3600
                    print(
                        f"    • {pump.start_time.date()} to {pump.end_time.date()}: "
                        f"+{pump.price_change_pct:.1f}% in {duration:.1f}h "
                        f"(volume {pump.volume_spike:.1f}x)"
                    )
                    all_pumps.append(pump)
            else:
                print(f"  No significant pumps")

            print()

        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    print("\n" + "=" * 80)
    print(f"TOTAL PUMPS FOUND: {len(all_pumps)}")
    print("=" * 80 + "\n")

    if all_pumps:
        # Export to CSV
        collector.export_pump_dataset(all_pumps, "recent_meme_pumps.csv")
        print(f"✓ Exported to: {collector.data_dir / 'recent_meme_pumps.csv'}")

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Review the CSV file with pump details")
        print("2. Pick 3-5 specific pumps to validate")
        print("3. Update pattern_validator test cases with exact dates")
        print("4. Re-run validation on SPECIFIC pump events")
        print("\nExample test case format:")
        print("{")
        print("    'symbol': 'WIFUSDT',")
        print("    'start_date': datetime(2024, 10, 28, 0, 0),  # Exact pump start")
        print("    'end_date': datetime(2024, 10, 29, 0, 0),    # Exact pump end")
        print("    'is_good_setup': True,")
        print("}")
        print()
    else:
        print("No pumps found in the last 30 days.")
        print("\nTry:")
        print("1. Extending date range (60-90 days)")
        print("2. Lowering pump threshold (30%+ instead of 50%+)")
        print("3. Looking at different timeframes")
        print()
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
