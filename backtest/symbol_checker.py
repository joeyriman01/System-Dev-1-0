<<<<<<< HEAD
"""
SYMBOL CHECKER
==============
Checks which symbols are actually available on Binance Futures.

Use this before testing to avoid "Invalid symbol" errors.
"""

import sys

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.binance_client import BinanceClient


def get_all_futures_symbols():
    """Get all available Binance Futures symbols"""
    client = BinanceClient()

    try:
        # Get exchange info
        exchange_info = client._request("GET", "/fapi/v1/exchangeInfo")

        symbols = []
        for symbol_info in exchange_info["symbols"]:
            if (
                symbol_info["status"] == "TRADING"
                and symbol_info["contractType"] == "PERPETUAL"
            ):
                symbols.append(symbol_info["symbol"])

        return sorted(symbols)
    except Exception as e:
        print(f"Error: {e}")
        return []


def find_meme_coins(all_symbols):
    """Filter for likely meme coin symbols"""

    # Common meme coin names
    meme_keywords = [
        "DOGE",
        "SHIB",
        "PEPE",
        "FLOKI",
        "BONK",
        "MEME",
        "DOGE",
        "ELON",
        "BABYDOGE",
        "SAFEMOON",
        "MOON",
        "WOJAK",
        "PEPE",
        "WIF",
        "BOME",
        "RATS",
        "ORDI",
        "SATS",
        "PEOPLE",
        "LADYS",
        "TURBO",
        "AIDOGE",
    ]

    meme_coins = []
    for symbol in all_symbols:
        for keyword in meme_keywords:
            if keyword in symbol and symbol.endswith("USDT"):
                meme_coins.append(symbol)
                break

    return sorted(set(meme_coins))


def check_symbol_exists(symbol):
    """Check if a specific symbol exists"""
    client = BinanceClient()

    try:
        # Try to fetch recent kline
        client.futures_klines(symbol, "1h", limit=1)
        return True
    except Exception as e:
        if "Invalid symbol" in str(e) or "-1121" in str(e):
            return False
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BINANCE FUTURES SYMBOL CHECKER")
    print("=" * 80 + "\n")

    print("Fetching all available Binance Futures symbols...\n")

    all_symbols = get_all_futures_symbols()

    if not all_symbols:
        print("❌ Failed to fetch symbols")
    else:
        print(f"✓ Found {len(all_symbols)} total perpetual futures symbols\n")

        # Find meme coins
        print("Filtering for meme coins...\n")
        meme_coins = find_meme_coins(all_symbols)

        print(f"✓ Found {len(meme_coins)} potential meme coin perpetuals:\n")

        for symbol in meme_coins:
            print(f"  {symbol}")

        print("\n" + "=" * 80)
        print("CHECKING SPECIFIC SYMBOLS")
        print("=" * 80 + "\n")

        # Check the symbols user tried
        test_symbols = [
            "PIGGYUSDT",
            "SHIBUSDT",
            "PEPEUSDT",
            "1000PEPEUSDT",
            "1000SHIBUSDT",
        ]

        for symbol in test_symbols:
            exists = check_symbol_exists(symbol)
            status = "✓ EXISTS" if exists else "❌ NOT FOUND"
            print(f"{symbol:20s} {status}")

        print("\n" + "=" * 80)
        print("\nUse the symbols from the meme coin list above for testing.")
        print("=" * 80 + "\n")
=======
"""
SYMBOL CHECKER
==============
Checks which symbols are actually available on Binance Futures.

Use this before testing to avoid "Invalid symbol" errors.
"""

import sys

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.binance_client import BinanceClient


def get_all_futures_symbols():
    """Get all available Binance Futures symbols"""
    client = BinanceClient()

    try:
        # Get exchange info
        exchange_info = client._request("GET", "/fapi/v1/exchangeInfo")

        symbols = []
        for symbol_info in exchange_info["symbols"]:
            if (
                symbol_info["status"] == "TRADING"
                and symbol_info["contractType"] == "PERPETUAL"
            ):
                symbols.append(symbol_info["symbol"])

        return sorted(symbols)
    except Exception as e:
        print(f"Error: {e}")
        return []


def find_meme_coins(all_symbols):
    """Filter for likely meme coin symbols"""

    # Common meme coin names
    meme_keywords = [
        "DOGE",
        "SHIB",
        "PEPE",
        "FLOKI",
        "BONK",
        "MEME",
        "DOGE",
        "ELON",
        "BABYDOGE",
        "SAFEMOON",
        "MOON",
        "WOJAK",
        "PEPE",
        "WIF",
        "BOME",
        "RATS",
        "ORDI",
        "SATS",
        "PEOPLE",
        "LADYS",
        "TURBO",
        "AIDOGE",
    ]

    meme_coins = []
    for symbol in all_symbols:
        for keyword in meme_keywords:
            if keyword in symbol and symbol.endswith("USDT"):
                meme_coins.append(symbol)
                break

    return sorted(set(meme_coins))


def check_symbol_exists(symbol):
    """Check if a specific symbol exists"""
    client = BinanceClient()

    try:
        # Try to fetch recent kline
        client.futures_klines(symbol, "1h", limit=1)
        return True
    except Exception as e:
        if "Invalid symbol" in str(e) or "-1121" in str(e):
            return False
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BINANCE FUTURES SYMBOL CHECKER")
    print("=" * 80 + "\n")

    print("Fetching all available Binance Futures symbols...\n")

    all_symbols = get_all_futures_symbols()

    if not all_symbols:
        print("❌ Failed to fetch symbols")
    else:
        print(f"✓ Found {len(all_symbols)} total perpetual futures symbols\n")

        # Find meme coins
        print("Filtering for meme coins...\n")
        meme_coins = find_meme_coins(all_symbols)

        print(f"✓ Found {len(meme_coins)} potential meme coin perpetuals:\n")

        for symbol in meme_coins:
            print(f"  {symbol}")

        print("\n" + "=" * 80)
        print("CHECKING SPECIFIC SYMBOLS")
        print("=" * 80 + "\n")

        # Check the symbols user tried
        test_symbols = [
            "PIGGYUSDT",
            "SHIBUSDT",
            "PEPEUSDT",
            "1000PEPEUSDT",
            "1000SHIBUSDT",
        ]

        for symbol in test_symbols:
            exists = check_symbol_exists(symbol)
            status = "✓ EXISTS" if exists else "❌ NOT FOUND"
            print(f"{symbol:20s} {status}")

        print("\n" + "=" * 80)
        print("\nUse the symbols from the meme coin list above for testing.")
        print("=" * 80 + "\n")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
