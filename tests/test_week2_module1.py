"""
COMPREHENSIVE TEST - WEEK 2 MODULE 1
=====================================
Tests the complete new listing scanner stack:
1. Settings configuration
2. BinanceClient API methods
3. ListingScanner functionality

Run this to verify everything works before continuing to Module 2.

Author: Grim
"""

import os
import sys

# Add paths if running from different directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_settings():
    """Test 1: Settings configuration"""
    print("\n" + "=" * 70)
    print("TEST 1: SETTINGS CONFIGURATION")
    print("=" * 70)

    try:
        from config.settings import settings

        print(f"✓ Settings loaded successfully")
        print(f"  Environment: {settings.ENV}")
        print(f"  Testnet: {settings.BINANCE_TESTNET}")
        print(f"  Paper Trading: {settings.PAPER_TRADING}")

        # Check API keys present
        if settings.BINANCE_API_KEY:
            print(
                f"  API Key: {settings.BINANCE_API_KEY[:8]}...{settings.BINANCE_API_KEY[-4:]}"
            )
        else:
            print(f"  API Key: NOT SET (expected for testnet)")

        return True

    except Exception as e:
        print(f"✗ Settings test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_binance_client():
    """Test 2: Binance API client"""
    print("\n" + "=" * 70)
    print("TEST 2: BINANCE API CLIENT")
    print("=" * 70)

    try:
        from config.settings import settings
        from data.binance_client import BinanceClient, BinanceClientError

        # Initialize client
        print("Initializing Binance client...")
        client = BinanceClient(testnet=settings.BINANCE_TESTNET)
        print("✓ Client initialized")

        # Test connectivity
        print("\nTesting connectivity...")
        if client.ping():
            print("✓ Ping successful")
        else:
            print("✗ Ping failed")
            return False

        # Test public endpoints
        print("\nTesting public endpoints...")

        # Get BTC price
        ticker = client.futures_ticker_price(symbol="BTCUSDT")
        btc_price = float(ticker["price"])
        print(f"✓ BTC Price: ${btc_price:,.2f}")

        # Get 24h stats
        stats = client.futures_ticker_24hr(symbol="BTCUSDT")
        change_pct = float(stats["priceChangePercent"])
        volume = float(stats["quoteVolume"])
        print(f"✓ BTC 24h Change: {change_pct:+.2f}%")
        print(f"✓ BTC 24h Volume: ${volume:,.0f}")

        # Get mark price and funding
        mark = client.futures_mark_price(symbol="BTCUSDT")
        funding = float(mark["lastFundingRate"]) * 100
        print(f"✓ BTC Funding Rate: {funding:.4f}%")

        # Get order book
        book = client.futures_orderbook(symbol="BTCUSDT", limit=5)
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        spread = (best_ask - best_bid) / best_bid * 10000
        print(f"✓ Order Book Spread: {spread:.1f} bps")

        # Get klines
        klines = client.futures_klines(symbol="BTCUSDT", interval="1h", limit=5)
        print(f"✓ Fetched {len(klines)} candles")

        # Test authenticated endpoints (may fail on testnet)
        print("\nTesting authenticated endpoints...")
        try:
            balance = client.futures_balance()
            usdt_balance = next((b for b in balance if b["asset"] == "USDT"), None)
            if usdt_balance:
                print(f"✓ USDT Balance: ${float(usdt_balance['balance']):,.2f}")
            else:
                print("✓ Balance retrieved (no USDT)")
        except BinanceClientError as e:
            print(f"⚠ Authentication failed: {str(e)[:100]}")
            print("  (Expected if using wrong testnet or no API keys)")

        return True

    except Exception as e:
        print(f"✗ Binance client test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_listing_scanner():
    """Test 3: Listing scanner"""
    print("\n" + "=" * 70)
    print("TEST 3: NEW LISTING SCANNER")
    print("=" * 70)

    try:
        from config.settings import settings
        from data.binance_client import BinanceClient
        from data.listing_scanner import ListingScanner

        # Initialize
        print("Initializing scanner...")
        client = BinanceClient(testnet=settings.BINANCE_TESTNET)
        scanner = ListingScanner(client)
        print("✓ Scanner initialized")

        # Test getting all perpetuals
        print("\nFetching perpetual contracts...")
        symbols = scanner.get_all_perpetuals()
        print(f"✓ Found {len(symbols)} perpetual contracts")
        print(f"  Sample symbols: {', '.join(symbols[:10])}")

        # Test market data fetch
        print("\nTesting market data fetch...")
        test_symbols = ["BTCUSDT", "ETHUSDT"]

        for symbol in test_symbols:
            details = scanner.get_listing_details(symbol)

            if details:
                print(f"\n✓ {symbol} Details:")
                print(f"  Price: ${details['current_price']:,.4f}")
                print(f"  Volume 24h: ${details['volume_24h']:,.0f}")
                print(f"  Price Change: {details['price_change_24h']:+.2f}%")
                print(f"  Volatility: {details['volatility_1h']:.2f}%")
                print(f"  Spread: {details['spread_bps']:.1f} bps")
                print(f"  Funding: {details['funding_rate']:.4f}%")

                # Test tradability check
                tradable, reason = scanner.is_tradable(details)
                print(f"  Tradable: {tradable} - {reason}")

                # Test priority scoring
                score = scanner.calculate_priority_score(details)
                print(f"  Priority Score: {score:.1f}/100")
            else:
                print(f"✗ Failed to fetch details for {symbol}")

        # Test with meme coins if available
        print("\nChecking for meme coins...")
        meme_keywords = ["PEPE", "DOGE", "SHIB", "FLOKI", "BONK", "WIF"]
        meme_symbols = [s for s in symbols if any(m in s for m in meme_keywords)]

        if meme_symbols:
            print(
                f"Found {len(meme_symbols)} meme coins: {', '.join(meme_symbols[:5])}"
            )

            # Analyze first meme coin
            test_meme = meme_symbols[0]
            print(f"\nAnalyzing {test_meme}...")
            meme_details = scanner.get_listing_details(test_meme)

            if meme_details:
                score = scanner.calculate_priority_score(meme_details)
                tradable, reason = scanner.is_tradable(meme_details)

                print(f"✓ {test_meme} Analysis:")
                print(f"  Price: ${meme_details['current_price']}")
                print(f"  Volume: ${meme_details['volume_24h']:,.0f}")
                print(f"  Change: {meme_details['price_change_24h']:+.2f}%")
                print(f"  Score: {score:.1f}/100")
                print(f"  Tradable: {tradable} - {reason}")
        else:
            print("No meme coins found in current listings")

        return True

    except Exception as e:
        print(f"✗ Scanner test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("NULLSPECTRE V2.0 - WEEK 2 MODULE 1 COMPREHENSIVE TEST")
    print("=" * 70)
    print("\nTesting new listing scanner and infrastructure...\n")

    results = {"settings": False, "binance_client": False, "listing_scanner": False}

    # Run tests
    results["settings"] = test_settings()

    if results["settings"]:
        results["binance_client"] = test_binance_client()

    if results["binance_client"]:
        results["listing_scanner"] = test_listing_scanner()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\nWeek 2 Module 1 is COMPLETE and READY.")
        print("\nNext Steps:")
        print("1. Move files to your project directory")
        print("2. Update imports in your existing code")
        print("3. Continue to Week 2 Module 2: Market Data Aggregator")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("=" * 70)
        print("\nFix the errors above before continuing.")
        print("Check:")
        print("- API connectivity")
        print("- Environment variables in .env")
        print("- Import paths")

    print("")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
