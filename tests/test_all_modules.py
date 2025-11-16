#!/usr/bin/env python3
"""
NullSpectre v2.0 - Complete Module Test
========================================
Tests all 4 core infrastructure modules:
1. Configuration
2. Logging
3. Database
4. Binance Client

Run this after installing all modules to verify everything works.
"""

import sys
from pathlib import Path
from datetime import datetime


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: MODULE IMPORTS")
    print("="*60)
    
    try:
        from config.settings import settings, validate_settings
        print("✓ Config module imported")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from utils.logging_config import setup_logging
        print("✓ Logging module imported")
    except ImportError as e:
        print(f"✗ Logging import failed: {e}")
        return False
    
    try:
        from database.models import init_database, Trade
        print("✓ Database module imported")
    except ImportError as e:
        print(f"✗ Database import failed: {e}")
        return False
    
    try:
        from data.binance_client import BinanceClient
        print("✓ Binance client imported")
    except ImportError as e:
        print(f"✗ Binance client import failed: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TEST 2: CONFIGURATION")
    print("="*60)
    
    try:
        from config.settings import settings, validate_settings
        
        # Check if settings loaded
        print(f"✓ Configuration loaded")
        print(f"  - Testnet Mode: {settings.binance_testnet}")
        print(f"  - Initial Capital: ${settings.initial_capital:.2f}")
        print(f"  - Max Leverage: {settings.max_leverage}X")
        print(f"  - Paper Trading: {settings.paper_trading}")
        
        # Validate settings
        if validate_settings():
            print("✓ Configuration validation passed")
        else:
            print("⚠ Configuration has warnings (check above)")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_logging():
    """Test logging system."""
    print("\n" + "="*60)
    print("TEST 3: LOGGING")
    print("="*60)
    
    try:
        from utils.logging_config import setup_logging, LoggingConfig
        from loguru import logger
        
        # Initialize logging
        setup_logging("DEBUG", "data_storage/logs/test.log")
        print("✓ Logging initialized")
        
        # Test different log levels
        logger.debug("Debug message test")
        logger.info("Info message test")
        logger.warning("Warning message test")
        
        # Test specialized logging
        LoggingConfig.log_signal("TEST", "ETHUSDT", 85)
        LoggingConfig.log_api_call("Binance", "/test", True, 0.123)
        
        print("✓ Logging functions work")
        print("  - Console output: Working")
        print("  - Log files: data_storage/logs/")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def test_database():
    """Test database system."""
    print("\n" + "="*60)
    print("TEST 4: DATABASE")
    print("="*60)
    
    try:
        from database.models import init_database, Trade, Signal
        
        # Initialize database
        db_manager = init_database("sqlite:///data_storage/nullspectre.db")
        print("✓ Database created: data_storage/nullspectre.db")
        
        # Test session
        session = db_manager.get_session()
        print("✓ Database session created")
        
        # Test creating a trade record
        test_trade = Trade(
            trade_id=f"TEST_{int(datetime.utcnow().timestamp())}",
            symbol="ETHUSDT",
            entry_time=datetime.utcnow(),
            entry_price=4901.50,
            entry_size=100.0,
            leverage=61,
            side="SHORT",
            conviction_score=85,
            initial_stop=4976.00,
            status="OPEN"
        )
        
        session.add(test_trade)
        session.commit()
        print("✓ Test trade record created")
        
        # Query it back
        trades = session.query(Trade).filter_by(trade_id=test_trade.trade_id).all()
        print(f"✓ Trade retrieved: {len(trades)} record(s)")
        
        # Test creating a signal record
        test_signal = Signal(
            signal_id=f"SIG_{int(datetime.utcnow().timestamp())}",
            symbol="ETHUSDT",
            conviction_score=85,
            conviction_tier="MAX",
            rejection_count=3,
            traded=False
        )
        
        session.add(test_signal)
        session.commit()
        print("✓ Test signal record created")
        
        # Clean up test data
        session.query(Trade).filter_by(trade_id=test_trade.trade_id).delete()
        session.query(Signal).filter_by(signal_id=test_signal.signal_id).delete()
        session.commit()
        session.close()
        print("✓ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_binance_client():
    """Test Binance API client."""
    print("\n" + "="*60)
    print("TEST 5: BINANCE CLIENT")
    print("="*60)
    
    try:
        from data.binance_client import BinanceClient
        from config.settings import settings
        
        # Initialize client
        client = BinanceClient(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=settings.binance_testnet
        )
        print("✓ Binance client initialized")
        print(f"  - Mode: {'TESTNET' if settings.binance_testnet else 'MAINNET'}")
        
        # Test public endpoint (no auth needed)
        try:
            ticker = client.get_ticker_price("ETHUSDT")
            price = float(ticker['price'])
            print(f"✓ Market data accessible")
            print(f"  - ETHUSDT Price: ${price:.2f}")
        except Exception as e:
            print(f"⚠ Market data fetch failed: {e}")
            return False
        
        # Test authenticated endpoint (requires API keys)
        try:
            if settings.binance_api_key and settings.binance_api_secret:
                balance = client.get_usdt_balance()
                print(f"✓ Account access working")
                print(f"  - USDT Balance: ${balance:.2f}")
            else:
                print("⚠ API keys not configured - skipping authenticated tests")
                print("  (Add BINANCE_API_KEY and BINANCE_API_SECRET to .env)")
        except Exception as e:
            print(f"⚠ Account access failed: {e}")
            print("  (This is normal if you haven't added API keys yet)")
        
        return True
        
    except Exception as e:
        print(f"✗ Binance client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("NULLSPECTRE V2.0 - COMPLETE MODULE TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Logging", test_logging),
        ("Database", test_database),
        ("Binance Client", test_binance_client),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {test_name}")
    
    print("-"*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready for the next phase:")
        print("  - New listing scanner")
        print("  - Signal detection")
        print("  - Position management")
        print("\nSay: 'Tests passed, what's next?'")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("⚠ SOME TESTS FAILED")
        print("="*60)
        print("\nFix the errors above, then run this test again.")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())