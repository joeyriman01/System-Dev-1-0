#!/usr/bin/env python3
"""Test Binance client."""

from data.binance_client import BinanceClient
from config.settings import settings

if __name__ == "__main__":
    print("Testing Binance client...")
    
    # Initialize client
    client = BinanceClient(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        testnet=settings.binance_testnet
    )
    print("✓ Client initialized")
    
    # Test public endpoint (no API key needed)
    try:
        ticker = client.get_ticker_price("ETHUSDT")
        price = float(ticker['price'])
        print(f"✓ ETHUSDT Price: ${price:.2f}")
    except Exception as e:
        print(f"✗ Price fetch failed: {e}")
    
    # Test authenticated endpoint (requires API keys)
    try:
        balance = client.get_usdt_balance()
        print(f"✓ USDT Balance: ${balance:.2f}")
    except Exception as e:
        print(f"⚠ Account access failed: {e}")
        print("  (Add API keys to .env to test this)")
    
    print("\n✓ Binance client test complete!")