<<<<<<< HEAD
"""
FIND AVAILABLE DATA RANGES
==========================
Discovers when historical data is actually available for user's symbols.
"""

import sys
from datetime import datetime, timedelta

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.binance_client import BinanceClient


def find_earliest_data(symbol: str) -> tuple:
    """
    Find the earliest available data for a symbol.

    Returns:
        (earliest_date, latest_date, has_data)
    """
    client = BinanceClient()

    # Try to get earliest available data
    # Start from 30 days ago and work backwards
    end_time = datetime.now()

    # Try last 7 days first
    try:
        start_time = end_time - timedelta(days=7)
        klines = client.futures_klines(
            symbol=symbol,
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )

        if klines:
            earliest = klines[0]["timestamp"]
            latest = klines[-1]["timestamp"]
            return (earliest, latest, True)
    except:
        pass

    # Try last 30 days
    try:
        start_time = end_time - timedelta(days=30)
        klines = client.futures_klines(
            symbol=symbol,
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )

        if klines:
            earliest = klines[0]["timestamp"]
            latest = klines[-1]["timestamp"]
            return (earliest, latest, True)
    except:
        pass

    # Try last 90 days
    try:
        start_time = end_time - timedelta(days=90)
        klines = client.futures_klines(
            symbol=symbol,
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )

        if klines:
            earliest = klines[0]["timestamp"]
            latest = klines[-1]["timestamp"]
            return (earliest, latest, True)
    except:
        pass

    return (None, None, False)


if __name__ == "__main__":
    symbols = [
        "ETHUSDT",
        "BNBUSDT",
        "BANANAS31USDT",
        "PENGUUSDT",
        "FARTCOINUSDT",
        "POPCATUSDT",
    ]

    print("\n" + "=" * 80)
    print("FINDING AVAILABLE DATA RANGES FOR USER'S SYMBOLS")
    print("=" * 80 + "\n")

    available_symbols = []

    for symbol in symbols:
        print(f"Checking {symbol}...")

        earliest, latest, has_data = find_earliest_data(symbol)

        if has_data:
            print(f"  ✓ Data available from {earliest.date()} to {latest.date()}")
            available_symbols.append(
                {"symbol": symbol, "earliest": earliest, "latest": latest}
            )
        else:
            print(f"  ✗ No data available (might be newly listed or delisted)")

        print()

    print("=" * 80)
    print(f"SUMMARY: {len(available_symbols)}/{len(symbols)} symbols have data")
    print("=" * 80 + "\n")

    if available_symbols:
        print("AVAILABLE SYMBOLS:")
        for s in available_symbols:
            days_available = (s["latest"] - s["earliest"]).days
            print(
                f"  {s['symbol']:20s} {s['earliest'].date()} to {s['latest'].date()} ({days_available} days)"
            )

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Use these date ranges to fetch historical data")
        print("2. Scan for pumps within available periods")
        print("3. Test detector on found pumps")
        print()
    else:
        print("⚠️  No data available for any symbols.")
        print("These coins might be:")
        print("- Very newly listed (< 7 days)")
        print("- Delisted from Binance Futures")
        print("- Spot-only (not available on futures)")
        print()
=======
"""
FIND AVAILABLE DATA RANGES
==========================
Discovers when historical data is actually available for user's symbols.
"""

import sys
from datetime import datetime, timedelta

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.binance_client import BinanceClient


def find_earliest_data(symbol: str) -> tuple:
    """
    Find the earliest available data for a symbol.

    Returns:
        (earliest_date, latest_date, has_data)
    """
    client = BinanceClient()

    # Try to get earliest available data
    # Start from 30 days ago and work backwards
    end_time = datetime.now()

    # Try last 7 days first
    try:
        start_time = end_time - timedelta(days=7)
        klines = client.futures_klines(
            symbol=symbol,
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )

        if klines:
            earliest = klines[0]["timestamp"]
            latest = klines[-1]["timestamp"]
            return (earliest, latest, True)
    except:
        pass

    # Try last 30 days
    try:
        start_time = end_time - timedelta(days=30)
        klines = client.futures_klines(
            symbol=symbol,
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )

        if klines:
            earliest = klines[0]["timestamp"]
            latest = klines[-1]["timestamp"]
            return (earliest, latest, True)
    except:
        pass

    # Try last 90 days
    try:
        start_time = end_time - timedelta(days=90)
        klines = client.futures_klines(
            symbol=symbol,
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000,
        )

        if klines:
            earliest = klines[0]["timestamp"]
            latest = klines[-1]["timestamp"]
            return (earliest, latest, True)
    except:
        pass

    return (None, None, False)


if __name__ == "__main__":
    symbols = [
        "ETHUSDT",
        "BNBUSDT",
        "BANANAS31USDT",
        "PENGUUSDT",
        "FARTCOINUSDT",
        "POPCATUSDT",
    ]

    print("\n" + "=" * 80)
    print("FINDING AVAILABLE DATA RANGES FOR USER'S SYMBOLS")
    print("=" * 80 + "\n")

    available_symbols = []

    for symbol in symbols:
        print(f"Checking {symbol}...")

        earliest, latest, has_data = find_earliest_data(symbol)

        if has_data:
            print(f"  ✓ Data available from {earliest.date()} to {latest.date()}")
            available_symbols.append(
                {"symbol": symbol, "earliest": earliest, "latest": latest}
            )
        else:
            print(f"  ✗ No data available (might be newly listed or delisted)")

        print()

    print("=" * 80)
    print(f"SUMMARY: {len(available_symbols)}/{len(symbols)} symbols have data")
    print("=" * 80 + "\n")

    if available_symbols:
        print("AVAILABLE SYMBOLS:")
        for s in available_symbols:
            days_available = (s["latest"] - s["earliest"]).days
            print(
                f"  {s['symbol']:20s} {s['earliest'].date()} to {s['latest'].date()} ({days_available} days)"
            )

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Use these date ranges to fetch historical data")
        print("2. Scan for pumps within available periods")
        print("3. Test detector on found pumps")
        print()
    else:
        print("⚠️  No data available for any symbols.")
        print("These coins might be:")
        print("- Very newly listed (< 7 days)")
        print("- Delisted from Binance Futures")
        print("- Spot-only (not available on futures)")
        print()
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
