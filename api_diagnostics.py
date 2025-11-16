"""
API DIAGNOSTICS
===============
Comprehensive test of Binance API to find exact issue.
"""

import os
import sys

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime, timedelta

import requests

from data.binance_client import BinanceClient


def test_api_comprehensive():
    """Run comprehensive API diagnostics"""

    print("\n" + "=" * 80)
    print("BINANCE API COMPREHENSIVE DIAGNOSTICS")
    print("=" * 80 + "\n")

    client = BinanceClient()

    # Test 1: Check if API keys are loaded
    print("TEST 1: API Key Configuration")
    print("-" * 80)
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if api_key:
        print(f"✓ API Key found: {api_key[:8]}...")
    else:
        print("✗ API Key NOT FOUND in environment")

    if api_secret:
        print(f"✓ API Secret found: {api_secret[:8]}...")
    else:
        print("✗ API Secret NOT FOUND in environment")

    print()

    # Test 2: Test public endpoint (no auth required)
    print("TEST 2: Public Endpoint (No Auth)")
    print("-" * 80)
    try:
        response = requests.get("https://fapi.binance.com/fapi/v1/ping")
        if response.status_code == 200:
            print("✓ Public endpoint works (Binance API is reachable)")
        else:
            print(f"✗ Public endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Network error: {e}")
    print()

    # Test 3: Get exchange info (public, no signature)
    print("TEST 3: Exchange Info (Public)")
    print("-" * 80)
    try:
        info = client._request("GET", "/fapi/v1/exchangeInfo")
        if "symbols" in info:
            print(f"✓ Exchange info retrieved ({len(info['symbols'])} symbols)")
        else:
            print("✗ Unexpected response format")
    except Exception as e:
        print(f"✗ Failed: {e}")
    print()

    # Test 4: Get account info (requires auth + signature)
    print("TEST 4: Account Info (Requires Auth)")
    print("-" * 80)
    try:
        account = client.futures_account()
        if "totalWalletBalance" in account:
            print(f"✓ Account info retrieved")
            print(f"  Wallet Balance: ${float(account['totalWalletBalance']):.2f}")
            print(f"  Can Trade: {account.get('canTrade', False)}")
        else:
            print("✗ Unexpected account response")
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Account info failed: {error_msg}")

        if "API-key" in error_msg:
            print("\n⚠️  API KEY ISSUE DETECTED")
            print("  Your API key is invalid or not recognized by Binance")
            print("  → Go to Binance > API Management")
            print("  → Delete old keys")
            print("  → Create NEW API key with 'Enable Futures' checked")
        elif "Signature" in error_msg:
            print("\n⚠️  SIGNATURE ISSUE DETECTED")
            print("  API Secret might be wrong or timestamp issue")
            print("  → Check API Secret is correct")
            print("  → Check system time is synchronized")
        elif "IP" in error_msg:
            print("\n⚠️  IP RESTRICTION ISSUE")
            print("  Your IP might not be whitelisted")
            print("  → Go to API Management")
            print("  → Check 'Restrict access to trusted IPs only' is UNCHECKED")
            print("  → Or add your current IP to whitelist")
    print()

    # Test 5: Get klines (public but often fails with bad keys)
    print("TEST 5: Historical Klines (ETHUSDT)")
    print("-" * 80)
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        klines = client.futures_klines(
            symbol="ETHUSDT",
            interval="1h",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=24,
        )

        if klines:
            print(f"✓ Klines retrieved ({len(klines)} candles)")
            print(f"  First candle: {klines[0]['timestamp']}")
            print(f"  Last candle: {klines[-1]['timestamp']}")
            print(
                f"  Price range: ${klines[0]['close']:.2f} - ${klines[-1]['close']:.2f}"
            )
        else:
            print("✗ Empty klines response")
    except Exception as e:
        print(f"✗ Klines failed: {e}")
    print()

    # Test 6: Test Futures-specific endpoint
    print("TEST 6: Futures Position Risk (Requires Futures Permission)")
    print("-" * 80)
    try:
        positions = client._request("GET", "/fapi/v2/positionRisk", signed=True)
        print(f"✓ Position risk retrieved ({len(positions)} positions)")
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Position risk failed: {error_msg}")

        if "not enabled" in error_msg.lower() or "futures" in error_msg.lower():
            print("\n⚠️  FUTURES NOT ENABLED ON API KEY")
            print("  Your API key doesn't have Futures trading permissions")
            print("  → Go to Binance > API Management")
            print("  → Edit API key")
            print("  → CHECK 'Enable Futures' ✓")
            print("  → Save changes")
    print()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print()
    print("If multiple tests failed, most likely issues:")
    print("1. API key not created with 'Enable Futures' checked")
    print("2. API key has IP restrictions blocking your connection")
    print("3. API Secret is incorrect in .env file")
    print("4. API key was created for Spot only, not Futures")
    print()
    print("FIX:")
    print("1. Go to: https://www.binance.com/en/my/settings/api-management")
    print("2. DELETE current API key")
    print("3. Create NEW key:")
    print("   - Name it 'NullSpectre_Futures'")
    print("   - CHECK 'Enable Futures' ✓")
    print("   - UNCHECK 'Restrict access to trusted IPs only'")
    print("   - READ-ONLY permissions are fine for testing")
    print("4. Copy new API Key and Secret to .env file")
    print("5. Restart terminal and re-run this diagnostic")
    print()


if __name__ == "__main__":
    test_api_comprehensive()
