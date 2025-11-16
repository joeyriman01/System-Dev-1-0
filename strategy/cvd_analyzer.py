"""
CUMULATIVE VOLUME DELTA (CVD) ANALYZER
======================================
Analyzes order flow to detect bearish divergence.

Supports both Binance and MEXC exchanges.

Author: Grim (Institutional Standards)
"""

import os

# Import both exchange clients
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv("USE_MEXC", "False").lower() == "true"


class CVDAnalyzer:
    """
    Analyzes Cumulative Volume Delta for divergence signals.

    CVD tracks net buying/selling pressure:
    - Buy volume - Sell volume over time
    - Bearish divergence: Price making higher highs but CVD making lower highs
    - Indicates weakening buying pressure at resistance
    """

    def __init__(
        self, client=None, lookback_period: int = 24, divergence_threshold: float = 0.15
    ):
        """
        Initialize CVD analyzer.

        Args:
            client: Exchange client (auto-creates if None)
            lookback_period: Candles to analyze
            divergence_threshold: Minimum CVD drop for valid divergence (15%)
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                logger.info("CVDAnalyzer using MEXC")
            else:
                self.client = BinanceClient()
                logger.info("CVDAnalyzer using Binance")
        else:
            self.client = client

        self.lookback_period = lookback_period
        self.divergence_threshold = divergence_threshold

        logger.info(
            f"CVDAnalyzer initialized (lookback={lookback_period}, threshold={divergence_threshold})"
        )

    def analyze_cvd(
        self, symbol: str, timeframe: str = "15m", lookback_candles: int = None
    ) -> Dict:
        """
        Analyze CVD for bearish divergence.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Candlestick interval
            lookback_candles: Override default lookback

        Returns:
            Dict with CVD analysis results
        """
        if lookback_candles is None:
            lookback_candles = self.lookback_period

        logger.info(f"Analyzing CVD for {symbol} ({timeframe})")

        # Get historical candles
        klines = self.client.get_klines(
            symbol=symbol, interval=timeframe, limit=lookback_candles
        )

        if not klines:
            logger.warning(f"No kline data for {symbol}")
            return self._empty_result(symbol)

        # Convert to DataFrame
        df = self._klines_to_df(klines)

        # Calculate CVD (simplified: use volume * direction)
        df["cvd"] = self._calculate_cvd(df)

        # Find divergence
        divergence = self._detect_divergence(df)

        logger.info(f"  CVD divergence detected: {divergence['detected']}")

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "divergence_detected": divergence["detected"],
            "cvd_trend": divergence["cvd_trend"],
            "price_trend": divergence["price_trend"],
            "divergence_strength": divergence["strength"],
            "reason": divergence["reason"],
        }

    def _calculate_cvd(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Cumulative Volume Delta.

        Simplified calculation:
        - Green candles (close > open): +volume
        - Red candles (close < open): -volume
        - Cumulative sum over time
        """
        # Determine candle direction
        df["direction"] = (df["close"] > df["open"]).astype(int) * 2 - 1  # +1 or -1

        # Volume delta = volume * direction
        df["delta"] = df["volume"] * df["direction"]

        # Cumulative sum
        cvd = df["delta"].cumsum()

        return cvd

    def _detect_divergence(self, df: pd.DataFrame) -> Dict:
        """
        Detect bearish divergence between price and CVD.

        Bearish divergence:
        - Price making higher highs
        - CVD making lower highs
        """
        # Get last 10 candles for divergence check
        recent = df.tail(10)

        if len(recent) < 5:
            return {
                "detected": False,
                "cvd_trend": "insufficient_data",
                "price_trend": "insufficient_data",
                "strength": 0,
                "reason": "Not enough data",
            }

        # Find price peaks
        price_peaks = recent[
            recent["high"] == recent["high"].rolling(3, center=True).max()
        ]

        # Find CVD peaks
        cvd_peaks = recent[recent["cvd"] == recent["cvd"].rolling(3, center=True).max()]

        if len(price_peaks) < 2 or len(cvd_peaks) < 2:
            return {
                "detected": False,
                "cvd_trend": "no_peaks",
                "price_trend": "no_peaks",
                "strength": 0,
                "reason": "Insufficient peaks for divergence",
            }

        # Check if price making higher highs
        price_trend = (
            "rising"
            if price_peaks["high"].iloc[-1] > price_peaks["high"].iloc[0]
            else "falling"
        )

        # Check if CVD making lower highs
        cvd_trend = (
            "falling"
            if cvd_peaks["cvd"].iloc[-1] < cvd_peaks["cvd"].iloc[0]
            else "rising"
        )

        # Bearish divergence = price rising, CVD falling
        divergence_detected = price_trend == "rising" and cvd_trend == "falling"

        # Calculate strength
        if divergence_detected:
            cvd_drop = abs(cvd_peaks["cvd"].iloc[-1] - cvd_peaks["cvd"].iloc[0]) / abs(
                cvd_peaks["cvd"].iloc[0]
            )
            strength = min(cvd_drop / self.divergence_threshold, 1.0)  # Normalized 0-1
        else:
            strength = 0

        return {
            "detected": divergence_detected and strength >= 1.0,
            "cvd_trend": cvd_trend,
            "price_trend": price_trend,
            "strength": strength,
            "reason": (
                "Bearish divergence confirmed"
                if divergence_detected
                else "No divergence"
            ),
        }

    def _klines_to_df(self, klines: List[List]) -> pd.DataFrame:
        """Convert klines to DataFrame."""
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def _empty_result(self, symbol: str) -> Dict:
        """Return empty result when no data available."""
        return {
            "symbol": symbol,
            "divergence_detected": False,
            "cvd_trend": "no_data",
            "price_trend": "no_data",
            "divergence_strength": 0,
            "reason": "No data available",
        }


if __name__ == "__main__":
    """Test CVD analyzer"""

    print("\n" + "=" * 80)
    print("CVD ANALYZER - TESTING")
    print("=" * 80 + "\n")

    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")

    # Initialize analyzer (auto-creates client)
    analyzer = CVDAnalyzer()

    # Test CVD analysis
    print("TEST: Analyze CVD on BTC")
    print("-" * 40)

    result = analyzer.analyze_cvd(
        symbol="BTCUSDT", timeframe="15m", lookback_candles=24
    )

    print(f"Symbol: {result['symbol']}")
    print(f"Divergence detected: {result['divergence_detected']}")
    print(f"CVD trend: {result['cvd_trend']}")
    print(f"Price trend: {result['price_trend']}")
    print(f"Strength: {result['divergence_strength']:.2f}")
    print(f"Reason: {result['reason']}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
