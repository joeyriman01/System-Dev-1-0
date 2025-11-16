import os
from dotenv import load_dotenv
from data.binance_client import BinanceClient

# Load .env
load_dotenv()

print("="*60)
print("BINANCE API KEY DIAGNOSTIC TEST")
print("="*60)

# Check if keys exist
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
testnet = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

print("\n1. CHECKING .ENV FILE:")
print(f"   API Key found: {'YES' if api_key else 'NO'}")
if api_key:
    print(f"   API Key starts with: {api_key[:8]}...")
print(f"   API Secret found: {'YES' if api_secret else 'NO'}")
if api_secret:
    print(f"   API Secret starts with: {api_secret[:8]}...")
print(f"   Testnet mode: {testnet}")

if not api_key or not api_secret:
    print("\n❌ ERROR: API keys not found in .env file!")
    print("\nAdd these to your .env file:")
    print("   BINANCE_API_KEY=your_key_here")
    print("   BINANCE_API_SECRET=your_secret_here")
    print("   BINANCE_TESTNET=True")
    exit(1)

# Test connection
print("\n2. TESTING API CONNECTION:")
client = BinanceClient(
    api_key=api_key,
    api_secret=api_secret,
    testnet=testnet
)

print(f"   Connecting to: {'TESTNET' if testnet else 'MAINNET'}")

# Test 1: Public endpoint (no auth needed)
print("\n3. TESTING PUBLIC ENDPOINT (no auth):")
try:
    ticker = client.get_ticker_price("BTCUSDT")
    price = float(ticker['price'])
    print(f"   ✓ SUCCESS: BTC Price = ${price:,.2f}")
    print("   → Public API works!")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    print("   → Check your internet connection")

# Test 2: Private endpoint (auth required)
print("\n4. TESTING AUTHENTICATED ENDPOINT:")
try:
    account = client.get_account()
    print(f"   ✓ SUCCESS: Account data retrieved!")
    
    # Show balance
    balance = client.get_usdt_balance()
    print(f"   → USDT Balance: ${balance:,.2f}")
    
    # Show positions
    positions = client.get_positions()
    open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    print(f"   → Open Positions: {len(open_positions)}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour API keys are working correctly!")
    
except Exception as e:
    error_str = str(e)
    print(f"   ✗ FAILED: {error_str}")
    
    print("\n" + "="*60)
    print("❌ AUTHENTICATION FAILED")
    print("="*60)
    
    if "401" in error_str or "Unauthorized" in error_str:
        print("\n🔍 POSSIBLE CAUSES:")
        print("\n1. WRONG KEYS FOR MODE:")
        if testnet:
            print("   → You're using TESTNET mode")
            print("   → Make sure keys are from: https://testnet.binancefuture.com/")
            print("   → NOT from mainnet Binance.com")
        else:
            print("   → You're using MAINNET mode")
            print("   → Make sure keys are from: https://www.binance.com/")
            print("   → NOT from testnet")
        
        print("\n2. MISSING FUTURES PERMISSIONS:")
        print("   → Go to API key settings")
        print("   → Enable 'Futures Trading' permission")
        print("   → Save and try again")
        
        print("\n3. IP RESTRICTIONS:")
        print("   → Check if API key has IP whitelist")
        print("   → Either disable restrictions or add your IP")
        print("   → Get your IP from: https://whatismyipaddress.com/")
        
        print("\n4. EXPIRED OR INVALID KEYS:")
        print("   → Try regenerating new API keys")
        print("   → Copy them carefully (no extra spaces)")
        
    elif "403" in error_str or "Forbidden" in error_str:
        print("\n🔍 PERMISSION ISSUE:")
        print("   → Your keys are valid but lack permissions")
        print("   → Enable 'Futures Trading' permission in API settings")
    
    print("\n" + "="*60)