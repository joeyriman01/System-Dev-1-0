<<<<<<< HEAD
"""
USER TRADES VALIDATOR
====================
Validates pattern detection on user's actual manual trades.

Strategy:
1. Fetch data for the periods shown in user's charts
2. Scan for all pumps within those periods
3. Test detector on each pump
4. Report which pumps were detected vs missed
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.data_collector import HistoricalDataCollector, PumpPeriod
from backtest.pattern_validator import PatternValidator, ValidationResult
from data.binance_client import BinanceClient

# User's actual trades with chart timeframes
USER_TRADES = [
    {
        "symbol": "ETHUSDT",
        "period_start": datetime(2024, 7, 1),
        "period_end": datetime(2024, 12, 1),
        "notes": "ETH - multiple timeframes visible",
    },
    {
        "symbol": "BNBUSDT",
        "period_start": datetime(2024, 9, 1),
        "period_end": datetime(2024, 11, 15),
        "notes": "BNB - resistance at ~$1,100",
    },
    {
        "symbol": "BANANAS31USDT",
        "period_start": datetime(2024, 6, 1),
        "period_end": datetime(2024, 8, 31),
        "notes": "BANANA - big July pump",
    },
    {
        "symbol": "PENGUUSDT",
        "period_start": datetime(2024, 7, 1),
        "period_end": datetime(2024, 9, 30),
        "notes": "PENGU - Jul-Sep period",
    },
    {
        "symbol": "FARTCOINUSDT",
        "period_start": datetime(2024, 5, 1),
        "period_end": datetime(2024, 8, 31),
        "notes": "FARTCOIN - May-Aug period",
    },
    {
        "symbol": "POPCATUSDT",
        "period_start": datetime(2024, 4, 1),
        "period_end": datetime(2024, 7, 31),
        "notes": "POPCAT - Apr-Jul period",
    },
]


def check_symbol_exists(symbol: str) -> bool:
    """Check if symbol exists on Binance Futures"""
    client = BinanceClient()
    try:
        client.futures_klines(symbol, "1h", limit=1)
        return True
    except:
        return False


def validate_user_trades():
    """
    Validate detector on all user's actual trades.

    Process:
    1. Check which symbols exist
    2. For each valid symbol, scan for pumps in the chart period
    3. Test detector on each found pump
    4. Report results
    """
    print("\n" + "=" * 80)
    print("USER TRADES VALIDATION")
    print("=" * 80 + "\n")

    client = BinanceClient()
    collector = HistoricalDataCollector(client)
    validator = PatternValidator(client)

    # Step 1: Check which symbols exist
    print("Step 1: Checking which symbols exist on Binance Futures...\n")

    valid_trades = []
    invalid_trades = []

    for trade in USER_TRADES:
        symbol = trade["symbol"]
        exists = check_symbol_exists(symbol)

        if exists:
            print(f"✓ {symbol:20s} EXISTS - {trade['notes']}")
            valid_trades.append(trade)
        else:
            print(f"✗ {symbol:20s} NOT FOUND - {trade['notes']}")
            invalid_trades.append(trade)

    print(f"\n✓ Found {len(valid_trades)} valid symbols to test")

    if not valid_trades:
        print("\n⚠️  No valid symbols found. Cannot proceed with validation.")
        return

    # Step 2: Scan each valid trade for pumps
    print("\n" + "=" * 80)
    print("Step 2: Scanning for pumps in user's trade periods...")
    print("=" * 80 + "\n")

    all_test_cases = []

    for trade in valid_trades:
        symbol = trade["symbol"]
        start = trade["period_start"]
        end = trade["period_end"]

        print(f"\nScanning {symbol} from {start.date()} to {end.date()}...")

        try:
            # Fetch historical data
            df = collector.fetch_historical_klines(
                symbol=symbol,
                interval="1h",
                start_time=start,
                end_time=end,
                save_to_file=False,
            )

            if df.empty:
                print(f"  ✗ No data returned")
                continue

            # Find pumps (25%+ in 48h = more lenient)
            # Lowered from 40% in 24h to catch more patterns
            pumps = collector.identify_pump_periods(
                df,
                min_pump_pct=25.0,  # 25%+ pump
                lookback_candles=48,  # Within 48 hours
            )

            if pumps:
                print(f"  ✓ Found {len(pumps)} pump(s):")
                for i, pump in enumerate(pumps, 1):
                    duration = (pump.end_time - pump.start_time).total_seconds() / 3600
                    print(
                        f"    #{i}: {pump.start_time.date()} - {pump.end_time.date()}: "
                        f"+{pump.price_change_pct:.1f}% in {duration:.1f}h "
                        f"(vol {pump.volume_spike:.1f}x)"
                    )

                    # Add to test cases
                    all_test_cases.append(
                        {
                            "symbol": symbol,
                            "pump": pump,
                            "historical_data": df,
                            "trade_notes": trade["notes"],
                        }
                    )
            else:
                print(f"  ✗ No significant pumps found")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not all_test_cases:
        print("\n⚠️  No pumps found in any of the trade periods.")
        print("\nTry:")
        print("1. Extending date ranges")
        print("2. Lowering pump threshold (30%)")
        print("3. Providing more specific dates")
        return

    # Step 3: Validate detector on all pumps
    print("\n" + "=" * 80)
    print(f"Step 3: Testing detector on {len(all_test_cases)} pumps...")
    print("=" * 80 + "\n")

    detected_count = 0
    missed_count = 0

    for i, test_case in enumerate(all_test_cases, 1):
        symbol = test_case["symbol"]
        pump = test_case["pump"]

        print(f"\n[{i}/{len(all_test_cases)}] Testing {symbol} pump...")
        print(f"  Period: {pump.start_time.date()} to {pump.end_time.date()}")
        print(f"  Move: +{pump.price_change_pct:.1f}%")

        # Get data for just this pump period (with some buffer)
        pump_start = pump.start_time - timedelta(hours=12)
        pump_end = pump.end_time + timedelta(hours=12)

        pump_data = test_case["historical_data"][
            (test_case["historical_data"]["timestamp"] >= pump_start)
            & (test_case["historical_data"]["timestamp"] <= pump_end)
        ]

        if pump_data.empty:
            print("  ✗ No data for pump period")
            continue

        # Validate
        result = validator.validate_single_pump(
            symbol=symbol,
            pump_period=pump,
            historical_data=pump_data,
            is_known_good_setup=True,
        )

        if result.pattern_detected:
            detected_count += 1
            print(f"  ✓ DETECTED")
        else:
            missed_count += 1
            print(f"  ✗ MISSED")

    # Step 4: Generate final report
    print("\n" + "=" * 80)
    print("FINAL VALIDATION REPORT")
    print("=" * 80 + "\n")

    total_pumps = len(all_test_cases)
    detection_rate = (detected_count / total_pumps * 100) if total_pumps > 0 else 0

    print(f"📊 DETECTION PERFORMANCE:")
    print(f"  Total Pumps Found: {total_pumps}")
    print(f"  Detected: {detected_count}")
    print(f"  Missed: {missed_count}")
    print(f"  Detection Rate: {detection_rate:.1f}%")
    print()

    if detection_rate >= 60:
        print("✅ GOOD: Detector catches most of your manual setups")
        print("   → System is working as designed")
        print("   → Ready to move to next development phase")
    elif detection_rate >= 40:
        print("⚠️  MODERATE: Detector catches some setups but misses others")
        print("   → Need to tune parameters")
        print("   → Review missed pumps to understand why")
    else:
        print("❌ POOR: Detector missing most setups")
        print("   → Rejection detector needs significant fixes")
        print("   → Parameters may be too strict")
        print("   → Pattern logic may not match your manual strategy")

    print("\n" + "=" * 80 + "\n")

    # Generate detailed report
    report = validator.generate_validation_report()

    return report


if __name__ == "__main__":
    """Run validation on user's actual trades"""

    print("\n" + "=" * 80)
    print("VALIDATING DETECTOR ON YOUR ACTUAL MANUAL TRADES")
    print("=" * 80 + "\n")

    print("This will:")
    print("1. Check which of your trade symbols exist on Binance")
    print("2. Scan those periods for pump patterns")
    print("3. Test if the detector would have caught them")
    print("4. Report detection accuracy")
    print()

    input("Press Enter to continue...")

    report = validate_user_trades()

    print("\n✅ VALIDATION COMPLETE\n")
=======
"""
USER TRADES VALIDATOR
====================
Validates pattern detection on user's actual manual trades.

Strategy:
1. Fetch data for the periods shown in user's charts
2. Scan for all pumps within those periods
3. Test detector on each pump
4. Report which pumps were detected vs missed
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List

project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.data_collector import HistoricalDataCollector, PumpPeriod
from backtest.pattern_validator import PatternValidator, ValidationResult
from data.binance_client import BinanceClient

# User's actual trades with chart timeframes
USER_TRADES = [
    {
        "symbol": "ETHUSDT",
        "period_start": datetime(2024, 7, 1),
        "period_end": datetime(2024, 12, 1),
        "notes": "ETH - multiple timeframes visible",
    },
    {
        "symbol": "BNBUSDT",
        "period_start": datetime(2024, 9, 1),
        "period_end": datetime(2024, 11, 15),
        "notes": "BNB - resistance at ~$1,100",
    },
    {
        "symbol": "BANANAS31USDT",
        "period_start": datetime(2024, 6, 1),
        "period_end": datetime(2024, 8, 31),
        "notes": "BANANA - big July pump",
    },
    {
        "symbol": "PENGUUSDT",
        "period_start": datetime(2024, 7, 1),
        "period_end": datetime(2024, 9, 30),
        "notes": "PENGU - Jul-Sep period",
    },
    {
        "symbol": "FARTCOINUSDT",
        "period_start": datetime(2024, 5, 1),
        "period_end": datetime(2024, 8, 31),
        "notes": "FARTCOIN - May-Aug period",
    },
    {
        "symbol": "POPCATUSDT",
        "period_start": datetime(2024, 4, 1),
        "period_end": datetime(2024, 7, 31),
        "notes": "POPCAT - Apr-Jul period",
    },
]


def check_symbol_exists(symbol: str) -> bool:
    """Check if symbol exists on Binance Futures"""
    client = BinanceClient()
    try:
        client.futures_klines(symbol, "1h", limit=1)
        return True
    except:
        return False


def validate_user_trades():
    """
    Validate detector on all user's actual trades.

    Process:
    1. Check which symbols exist
    2. For each valid symbol, scan for pumps in the chart period
    3. Test detector on each found pump
    4. Report results
    """
    print("\n" + "=" * 80)
    print("USER TRADES VALIDATION")
    print("=" * 80 + "\n")

    client = BinanceClient()
    collector = HistoricalDataCollector(client)
    validator = PatternValidator(client)

    # Step 1: Check which symbols exist
    print("Step 1: Checking which symbols exist on Binance Futures...\n")

    valid_trades = []
    invalid_trades = []

    for trade in USER_TRADES:
        symbol = trade["symbol"]
        exists = check_symbol_exists(symbol)

        if exists:
            print(f"✓ {symbol:20s} EXISTS - {trade['notes']}")
            valid_trades.append(trade)
        else:
            print(f"✗ {symbol:20s} NOT FOUND - {trade['notes']}")
            invalid_trades.append(trade)

    print(f"\n✓ Found {len(valid_trades)} valid symbols to test")

    if not valid_trades:
        print("\n⚠️  No valid symbols found. Cannot proceed with validation.")
        return

    # Step 2: Scan each valid trade for pumps
    print("\n" + "=" * 80)
    print("Step 2: Scanning for pumps in user's trade periods...")
    print("=" * 80 + "\n")

    all_test_cases = []

    for trade in valid_trades:
        symbol = trade["symbol"]
        start = trade["period_start"]
        end = trade["period_end"]

        print(f"\nScanning {symbol} from {start.date()} to {end.date()}...")

        try:
            # Fetch historical data
            df = collector.fetch_historical_klines(
                symbol=symbol,
                interval="1h",
                start_time=start,
                end_time=end,
                save_to_file=False,
            )

            if df.empty:
                print(f"  ✗ No data returned")
                continue

            # Find pumps (25%+ in 48h = more lenient)
            # Lowered from 40% in 24h to catch more patterns
            pumps = collector.identify_pump_periods(
                df,
                min_pump_pct=25.0,  # 25%+ pump
                lookback_candles=48,  # Within 48 hours
            )

            if pumps:
                print(f"  ✓ Found {len(pumps)} pump(s):")
                for i, pump in enumerate(pumps, 1):
                    duration = (pump.end_time - pump.start_time).total_seconds() / 3600
                    print(
                        f"    #{i}: {pump.start_time.date()} - {pump.end_time.date()}: "
                        f"+{pump.price_change_pct:.1f}% in {duration:.1f}h "
                        f"(vol {pump.volume_spike:.1f}x)"
                    )

                    # Add to test cases
                    all_test_cases.append(
                        {
                            "symbol": symbol,
                            "pump": pump,
                            "historical_data": df,
                            "trade_notes": trade["notes"],
                        }
                    )
            else:
                print(f"  ✗ No significant pumps found")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not all_test_cases:
        print("\n⚠️  No pumps found in any of the trade periods.")
        print("\nTry:")
        print("1. Extending date ranges")
        print("2. Lowering pump threshold (30%)")
        print("3. Providing more specific dates")
        return

    # Step 3: Validate detector on all pumps
    print("\n" + "=" * 80)
    print(f"Step 3: Testing detector on {len(all_test_cases)} pumps...")
    print("=" * 80 + "\n")

    detected_count = 0
    missed_count = 0

    for i, test_case in enumerate(all_test_cases, 1):
        symbol = test_case["symbol"]
        pump = test_case["pump"]

        print(f"\n[{i}/{len(all_test_cases)}] Testing {symbol} pump...")
        print(f"  Period: {pump.start_time.date()} to {pump.end_time.date()}")
        print(f"  Move: +{pump.price_change_pct:.1f}%")

        # Get data for just this pump period (with some buffer)
        pump_start = pump.start_time - timedelta(hours=12)
        pump_end = pump.end_time + timedelta(hours=12)

        pump_data = test_case["historical_data"][
            (test_case["historical_data"]["timestamp"] >= pump_start)
            & (test_case["historical_data"]["timestamp"] <= pump_end)
        ]

        if pump_data.empty:
            print("  ✗ No data for pump period")
            continue

        # Validate
        result = validator.validate_single_pump(
            symbol=symbol,
            pump_period=pump,
            historical_data=pump_data,
            is_known_good_setup=True,
        )

        if result.pattern_detected:
            detected_count += 1
            print(f"  ✓ DETECTED")
        else:
            missed_count += 1
            print(f"  ✗ MISSED")

    # Step 4: Generate final report
    print("\n" + "=" * 80)
    print("FINAL VALIDATION REPORT")
    print("=" * 80 + "\n")

    total_pumps = len(all_test_cases)
    detection_rate = (detected_count / total_pumps * 100) if total_pumps > 0 else 0

    print(f"📊 DETECTION PERFORMANCE:")
    print(f"  Total Pumps Found: {total_pumps}")
    print(f"  Detected: {detected_count}")
    print(f"  Missed: {missed_count}")
    print(f"  Detection Rate: {detection_rate:.1f}%")
    print()

    if detection_rate >= 60:
        print("✅ GOOD: Detector catches most of your manual setups")
        print("   → System is working as designed")
        print("   → Ready to move to next development phase")
    elif detection_rate >= 40:
        print("⚠️  MODERATE: Detector catches some setups but misses others")
        print("   → Need to tune parameters")
        print("   → Review missed pumps to understand why")
    else:
        print("❌ POOR: Detector missing most setups")
        print("   → Rejection detector needs significant fixes")
        print("   → Parameters may be too strict")
        print("   → Pattern logic may not match your manual strategy")

    print("\n" + "=" * 80 + "\n")

    # Generate detailed report
    report = validator.generate_validation_report()

    return report


if __name__ == "__main__":
    """Run validation on user's actual trades"""

    print("\n" + "=" * 80)
    print("VALIDATING DETECTOR ON YOUR ACTUAL MANUAL TRADES")
    print("=" * 80 + "\n")

    print("This will:")
    print("1. Check which of your trade symbols exist on Binance")
    print("2. Scan those periods for pump patterns")
    print("3. Test if the detector would have caught them")
    print("4. Report detection accuracy")
    print()

    input("Press Enter to continue...")

    report = validate_user_trades()

    print("\n✅ VALIDATION COMPLETE\n")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
