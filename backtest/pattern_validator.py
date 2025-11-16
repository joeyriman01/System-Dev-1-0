"""
PATTERN VALIDATOR
=================
Validates pattern detection accuracy on historical pump-and-dump data.

This module tests whether your Week 3 strategy modules actually catch
real pump patterns by running them on historical data.

Tests:
- Does rejection detector catch the "3 pumps rule"?
- Does CVD analyzer detect bearish divergence?
- Does conviction scorer produce reasonable scores?
- Does signal generator create profitable signals?

Output:
- Detection accuracy metrics
- False positive/negative rates
- Example signal analysis
- Validation report

Author: Grim (Institutional Standards)
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = r"C:\Users\lenovo\Desktop\Nullspectre_v2"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.data_collector import HistoricalDataCollector, PumpPeriod
from data.binance_client import BinanceClient
from data.funding_tracker import FundingRateTracker
from data.market_data import MarketDataAggregator
from data.orderbook_monitor import OrderBookMonitor
from strategy.conviction_scorer import ConvictionScorer, ConvictionSignal
from strategy.cvd_analyzer import CVDAnalysis, CVDAnalyzer
from strategy.rejection_detector import RejectionPattern, RejectionPatternDetector
from strategy.signal_generator import SignalGenerator, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of pattern validation on a single pump"""

    symbol: str
    pump_period: PumpPeriod

    # Detection results
    pattern_detected: bool
    rejection_pattern: Optional[RejectionPattern]
    cvd_analysis: Optional[CVDAnalysis]
    conviction_signal: Optional[ConvictionSignal]
    trade_signal: Optional[TradeSignal]

    # Validation metrics
    detection_quality: str  # 'TRUE_POSITIVE', 'FALSE_POSITIVE', 'FALSE_NEGATIVE'

    # Simulated outcome (if signal generated)
    would_have_traded: bool
    simulated_pnl_pct: Optional[float]
    hit_stop: bool
    hit_targets: List[bool]  # Which TP levels hit

    # Notes
    notes: List[str]


@dataclass
class ValidationReport:
    """Aggregate validation metrics"""

    total_pumps: int
    patterns_detected: int
    signals_generated: int

    # Accuracy metrics
    true_positives: int  # Detected real patterns
    false_positives: int  # Flagged non-patterns
    false_negatives: int  # Missed real patterns

    detection_accuracy: float  # (TP + TN) / Total
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)

    # Performance metrics
    avg_conviction_score: float
    signals_above_threshold: int

    # Simulated trading results
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl_pct: float
    max_drawdown_pct: float

    # Individual results
    results: List[ValidationResult]


class PatternValidator:
    """
    Validates pattern detection on historical pump data.

    This is CRITICAL - it tells you if your strategy actually works
    before you risk real money.
    """

    def __init__(self, client: BinanceClient, data_dir: Optional[str] = None):
        """
        Initialize pattern validator.

        Args:
            client: BinanceClient instance
            data_dir: Directory with historical data (default: data/historical)
        """
        self.client = client

        # Initialize all strategy components
        self.market_data = MarketDataAggregator(client)
        self.rejection_detector = RejectionPatternDetector(self.market_data)
        self.cvd_analyzer = CVDAnalyzer(client)
        self.funding_tracker = FundingRateTracker(client)
        self.orderbook_monitor = OrderBookMonitor(client)

        self.conviction_scorer = ConvictionScorer(
            rejection_detector=self.rejection_detector,
            cvd_analyzer=self.cvd_analyzer,
            funding_tracker=self.funding_tracker,
            orderbook_monitor=self.orderbook_monitor,
            market_data=self.market_data,
        )

        self.signal_generator = SignalGenerator(
            conviction_scorer=self.conviction_scorer, account_balance=10000.0
        )

        # Data collector
        self.data_collector = HistoricalDataCollector(client, data_dir)

        # Results storage
        self.validation_results: List[ValidationResult] = []

        logger.info("PatternValidator initialized")

    def validate_single_pump(
        self,
        symbol: str,
        pump_period: PumpPeriod,
        historical_data: pd.DataFrame,
        is_known_good_setup: bool = True,
    ) -> ValidationResult:
        """
        Validate pattern detection on a single historical pump.

        Args:
            symbol: Trading pair
            pump_period: PumpPeriod object with pump details
            historical_data: DataFrame with OHLCV data covering the pump
            is_known_good_setup: Whether this is a manually validated good setup

        Returns:
            ValidationResult with detection and simulation results
        """
        logger.info(
            f"\nValidating {symbol} pump: {pump_period.start_time} to {pump_period.end_time}"
        )

        notes = []

        try:
            # Temporarily override market data aggregator to use historical data
            # This is a simplification - in production you'd mock the entire data pipeline

            # Run rejection detector
            # For now, we'll use a simplified approach where we pass the data directly
            rejection_pattern = self._detect_pattern_from_historical(
                historical_data, symbol
            )

            pattern_detected = (
                rejection_pattern.has_pattern if rejection_pattern else False
            )

            if pattern_detected:
                notes.append(
                    f"✓ Pattern detected: {rejection_pattern.num_rejections} rejections"
                )
                notes.append(f"  Resistance: ${rejection_pattern.resistance_level:.8f}")
                notes.append(
                    f"  Strength: {rejection_pattern.rejection_strength:.1f}/100"
                )
            else:
                notes.append("✗ No pattern detected")

            # For historical validation, we can't run live CVD/funding/orderbook
            # So we'll mark these as N/A for now
            cvd_analysis = None
            conviction_signal = None
            trade_signal = None

            # Determine detection quality
            if is_known_good_setup and pattern_detected:
                detection_quality = "TRUE_POSITIVE"
            elif is_known_good_setup and not pattern_detected:
                detection_quality = "FALSE_NEGATIVE"
                notes.append("⚠️  MISSED: This was a good setup but wasn't detected")
            elif not is_known_good_setup and pattern_detected:
                detection_quality = "FALSE_POSITIVE"
                notes.append("⚠️  FALSE ALARM: Detected pattern in non-setup")
            else:
                detection_quality = "TRUE_NEGATIVE"

            # Simulate trade outcome if pattern was detected
            would_have_traded = pattern_detected
            simulated_pnl_pct = None
            hit_stop = False
            hit_targets = [False, False, False, False]

            if pattern_detected and rejection_pattern:
                # Simulate what would have happened
                entry_price = (
                    rejection_pattern.resistance_level * 0.95
                )  # Enter slight below resistance
                stop_loss = rejection_pattern.resistance_level * 1.05  # +5% stop

                # Check price action after detection
                detection_time = rejection_pattern.latest_rejection_time
                subsequent_data = historical_data[
                    historical_data["timestamp"] > detection_time
                ]

                if not subsequent_data.empty:
                    # Check if stop was hit
                    max_price_after = subsequent_data["high"].max()
                    if max_price_after >= stop_loss:
                        hit_stop = True
                        simulated_pnl_pct = -5.0
                        notes.append(f"  ✗ Stop hit at ${stop_loss:.8f}")
                    else:
                        # Check which targets were hit
                        min_price_after = subsequent_data["low"].min()

                        tp1 = entry_price * 0.5  # -50%
                        tp2 = entry_price * 0.0  # -100% (can't go below 0)
                        tp3 = entry_price * 0.0  # -200% (theoretical)

                        if min_price_after <= tp1:
                            hit_targets[0] = True
                            simulated_pnl_pct = 50.0
                            notes.append(f"  ✓ TP1 hit: +50%")

                        # For simplicity, estimate total PnL
                        pnl = (entry_price - min_price_after) / entry_price * 100
                        simulated_pnl_pct = min(pnl, 200.0)  # Cap at 200%
                        notes.append(f"  📊 Simulated P&L: {simulated_pnl_pct:+.1f}%")

            result = ValidationResult(
                symbol=symbol,
                pump_period=pump_period,
                pattern_detected=pattern_detected,
                rejection_pattern=rejection_pattern,
                cvd_analysis=cvd_analysis,
                conviction_signal=conviction_signal,
                trade_signal=trade_signal,
                detection_quality=detection_quality,
                would_have_traded=would_have_traded,
                simulated_pnl_pct=simulated_pnl_pct,
                hit_stop=hit_stop,
                hit_targets=hit_targets,
                notes=notes,
            )

            self.validation_results.append(result)

            # Print summary
            print(f"\n{'='*60}")
            print(f"VALIDATION: {symbol}")
            print(f"{'='*60}")
            print(
                f"Pump: {pump_period.price_change_pct:+.1f}% over {(pump_period.end_time - pump_period.start_time).total_seconds() / 3600:.1f}h"
            )
            print(f"Detection: {detection_quality}")
            for note in notes:
                print(note)
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            logger.error(f"Error validating {symbol}: {e}", exc_info=True)

            # Return failed result
            result = ValidationResult(
                symbol=symbol,
                pump_period=pump_period,
                pattern_detected=False,
                rejection_pattern=None,
                cvd_analysis=None,
                conviction_signal=None,
                trade_signal=None,
                detection_quality="ERROR",
                would_have_traded=False,
                simulated_pnl_pct=None,
                hit_stop=False,
                hit_targets=[False, False, False, False],
                notes=[f"Error: {str(e)}"],
            )

            self.validation_results.append(result)
            return result

    def _detect_pattern_from_historical(
        self, df: pd.DataFrame, symbol: str
    ) -> Optional[RejectionPattern]:
        """
        Run rejection detector on historical data.

        This is a simplified version that works directly with DataFrame.
        """
        try:
            # Find resistance level
            resistance, confidence = self.rejection_detector.detect_resistance_level(df)

            if resistance is None:
                return RejectionPattern(
                    symbol=symbol,
                    has_pattern=False,
                    resistance_level=None,
                    resistance_confidence=0.0,
                    num_rejections=0,
                    rejections=[],
                    rejection_strength=0.0,
                    time_span_hours=0.0,
                    avg_volume_ratio=0.0,
                    avg_wick_percent=0.0,
                    first_rejection_time=None,
                    latest_rejection_time=None,
                    conviction_factors={},
                )

            # Find rejections
            rejections = self.rejection_detector.find_rejections(df, resistance)
            num_rejections = len(rejections)

            if num_rejections < 2:
                return RejectionPattern(
                    symbol=symbol,
                    has_pattern=False,
                    resistance_level=resistance,
                    resistance_confidence=confidence,
                    num_rejections=num_rejections,
                    rejections=rejections,
                    rejection_strength=0.0,
                    time_span_hours=0.0,
                    avg_volume_ratio=0.0,
                    avg_wick_percent=0.0,
                    first_rejection_time=None,
                    latest_rejection_time=None,
                    conviction_factors={},
                )

            # Calculate metrics
            first_time = rejections[0].timestamp
            latest_time = rejections[-1].timestamp
            time_span = (latest_time - first_time).total_seconds() / 3600

            # Check time span
            if time_span > 4.0 or time_span < 0.5:
                return RejectionPattern(
                    symbol=symbol,
                    has_pattern=False,
                    resistance_level=resistance,
                    resistance_confidence=confidence,
                    num_rejections=num_rejections,
                    rejections=rejections,
                    rejection_strength=0.0,
                    time_span_hours=time_span,
                    avg_volume_ratio=0.0,
                    avg_wick_percent=0.0,
                    first_rejection_time=first_time,
                    latest_rejection_time=latest_time,
                    conviction_factors={},
                )

            # Calculate strength
            strength, factors = self.rejection_detector.calculate_rejection_strength(
                rejections, num_rejections
            )

            avg_volume_ratio = np.mean([r.volume_ratio for r in rejections])
            avg_wick_percent = np.mean([r.wick_percent for r in rejections])

            return RejectionPattern(
                symbol=symbol,
                has_pattern=True,
                resistance_level=resistance,
                resistance_confidence=confidence,
                num_rejections=num_rejections,
                rejections=rejections,
                rejection_strength=strength,
                time_span_hours=time_span,
                avg_volume_ratio=avg_volume_ratio,
                avg_wick_percent=avg_wick_percent,
                first_rejection_time=first_time,
                latest_rejection_time=latest_time,
                conviction_factors=factors,
            )

        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return None

    def validate_multiple_pumps(self, test_cases: List[Dict]) -> ValidationReport:
        """
        Validate pattern detection on multiple pumps.

        Args:
            test_cases: List of dicts with keys:
                - symbol: str
                - start_date: datetime
                - end_date: datetime
                - is_good_setup: bool

        Returns:
            ValidationReport with aggregate metrics
        """
        logger.info(f"\nValidating {len(test_cases)} test cases...")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Processing {test_case['symbol']}...")

            try:
                # Fetch historical data
                df = self.data_collector.fetch_historical_klines(
                    symbol=test_case["symbol"],
                    interval="1h",
                    start_time=test_case["start_date"],
                    end_time=test_case["end_date"],
                    save_to_file=True,
                )

                if df.empty:
                    logger.warning(f"No data for {test_case['symbol']}")
                    continue

                # Create pump period
                pump = PumpPeriod(
                    symbol=test_case["symbol"],
                    start_time=test_case["start_date"],
                    end_time=test_case["end_date"],
                    price_change_pct=(df["close"].max() / df["close"].min() - 1) * 100,
                    max_price=df["high"].max(),
                    min_price=df["low"].min(),
                    volume_spike=df["volume"].max() / df["volume"].mean(),
                )

                # Validate
                self.validate_single_pump(
                    symbol=test_case["symbol"],
                    pump_period=pump,
                    historical_data=df,
                    is_known_good_setup=test_case.get("is_good_setup", True),
                )

            except Exception as e:
                logger.error(f"Error processing {test_case['symbol']}: {e}")
                continue

        # Generate report
        return self.generate_validation_report()

    def generate_validation_report(self) -> ValidationReport:
        """
        Generate aggregate validation metrics.

        Returns:
            ValidationReport with all metrics
        """
        if not self.validation_results:
            logger.warning("No validation results to report")
            return None

        total_pumps = len(self.validation_results)
        patterns_detected = sum(
            1 for r in self.validation_results if r.pattern_detected
        )
        signals_generated = sum(
            1 for r in self.validation_results if r.would_have_traded
        )

        # Calculate detection metrics
        true_positives = sum(
            1 for r in self.validation_results if r.detection_quality == "TRUE_POSITIVE"
        )
        false_positives = sum(
            1
            for r in self.validation_results
            if r.detection_quality == "FALSE_POSITIVE"
        )
        false_negatives = sum(
            1
            for r in self.validation_results
            if r.detection_quality == "FALSE_NEGATIVE"
        )
        true_negatives = sum(
            1 for r in self.validation_results if r.detection_quality == "TRUE_NEGATIVE"
        )

        detection_accuracy = (
            (true_positives + true_negatives) / total_pumps if total_pumps > 0 else 0
        )
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        # Calculate trading metrics
        trades_with_pnl = [
            r for r in self.validation_results if r.simulated_pnl_pct is not None
        ]
        winning_trades = sum(1 for r in trades_with_pnl if r.simulated_pnl_pct > 0)
        losing_trades = sum(1 for r in trades_with_pnl if r.simulated_pnl_pct <= 0)

        win_rate = winning_trades / len(trades_with_pnl) if trades_with_pnl else 0
        avg_pnl_pct = (
            np.mean([r.simulated_pnl_pct for r in trades_with_pnl])
            if trades_with_pnl
            else 0
        )

        # Calculate max drawdown (simplified)
        pnls = [
            r.simulated_pnl_pct
            for r in trades_with_pnl
            if r.simulated_pnl_pct is not None
        ]
        max_drawdown_pct = min(pnls) if pnls else 0

        # Average conviction (if available)
        convictions = [
            r.conviction_signal.conviction_score
            for r in self.validation_results
            if r.conviction_signal is not None
        ]
        avg_conviction_score = np.mean(convictions) if convictions else 0
        signals_above_threshold = sum(1 for c in convictions if c >= 60)

        report = ValidationReport(
            total_pumps=total_pumps,
            patterns_detected=patterns_detected,
            signals_generated=signals_generated,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            detection_accuracy=detection_accuracy,
            precision=precision,
            recall=recall,
            avg_conviction_score=avg_conviction_score,
            signals_above_threshold=signals_above_threshold,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl_pct,
            max_drawdown_pct=max_drawdown_pct,
            results=self.validation_results,
        )

        # Print report
        self._print_validation_report(report)

        return report

    def _print_validation_report(self, report: ValidationReport):
        """Print formatted validation report"""

        print("\n" + "=" * 80)
        print("PATTERN VALIDATION REPORT")
        print("=" * 80 + "\n")

        print(f"📊 DETECTION METRICS:")
        print(f"  Total Pumps Tested: {report.total_pumps}")
        print(
            f"  Patterns Detected: {report.patterns_detected} ({report.patterns_detected/report.total_pumps*100:.1f}%)"
        )
        print(f"  Signals Generated: {report.signals_generated}")
        print()

        print(f"  True Positives: {report.true_positives} (detected real patterns)")
        print(f"  False Positives: {report.false_positives} (flagged non-patterns)")
        print(f"  False Negatives: {report.false_negatives} (missed real patterns)")
        print()

        print(f"  Detection Accuracy: {report.detection_accuracy*100:.1f}%")
        print(
            f"  Precision: {report.precision*100:.1f}% (of detected, how many were real)"
        )
        print(
            f"  Recall: {report.recall*100:.1f}% (of real patterns, how many detected)"
        )
        print()

        if report.avg_conviction_score > 0:
            print(f"🎯 CONVICTION METRICS:")
            print(f"  Average Conviction Score: {report.avg_conviction_score:.1f}/100")
            print(f"  Signals Above Threshold (60): {report.signals_above_threshold}")
            print()

        print(f"💰 SIMULATED TRADING RESULTS:")
        print(f"  Winning Trades: {report.winning_trades}")
        print(f"  Losing Trades: {report.losing_trades}")
        print(f"  Win Rate: {report.win_rate*100:.1f}%")
        print(f"  Average P&L: {report.avg_pnl_pct:+.1f}%")
        print(f"  Max Drawdown: {report.max_drawdown_pct:.1f}%")
        print()

        print("=" * 80 + "\n")


# ============================================================================
# TESTING & USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """Test the pattern validator"""

    print("\n" + "=" * 80)
    print("PATTERN VALIDATOR - TESTING")
    print("=" * 80 + "\n")

    # Initialize
    client = BinanceClient()
    validator = PatternValidator(client)

    print("Pattern validator initialized\n")

    # Define test cases
    # These are REAL pump events found by the pump_finder
    test_cases = [
        {
            "symbol": "MEMEUSDT",
            "start_date": datetime(2024, 8, 21, 0, 0),  # Start of +59.6% pump
            "end_date": datetime(2024, 8, 22, 23, 59),  # End of pump
            "is_good_setup": True,  # Known pump event
        },
        {
            "symbol": "MEMEUSDT",
            "start_date": datetime(2024, 8, 22, 0, 0),  # Another pump
            "end_date": datetime(2024, 8, 23, 23, 59),  # +50.5%
            "is_good_setup": True,
        },
        {
            "symbol": "MEMEUSDT",
            "start_date": datetime(2024, 8, 19, 0, 0),  # Earlier pump
            "end_date": datetime(2024, 8, 20, 23, 59),  # +46.5%
            "is_good_setup": True,
        },
    ]

    print(f"Running validation on {len(test_cases)} test cases...\n")

    # Run validation
    report = validator.validate_multiple_pumps(test_cases)

    print("\n✅ VALIDATION COMPLETE\n")

    print("Next steps:")
    print("1. Add more historical pump test cases")
    print("2. Review false negatives (missed patterns)")
    print("3. Analyze false positives (wrong signals)")
    print("4. Optimize parameters based on results")
    print()
