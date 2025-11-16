<<<<<<< HEAD
"""
CONVICTION SCORER
=================
Combines all signals into a single conviction score (0-100).

Supports both Binance and MEXC exchanges.

Author: Grim (Institutional Standards)
"""

import os

# Import both exchange clients
import sys
from pathlib import Path
from typing import Dict

from loguru import logger

from .cvd_analyzer import CVDAnalyzer

# Import strategy modules - USE RELATIVE IMPORTS
from .rejection_pattern import RejectionPatternDetector

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv("USE_MEXC", "False").lower() == "true"


class ConvictionScorer:
    """
    Calculates conviction score by combining multiple signals.

    Score breakdown:
    - Rejection pattern: 0-40 points
    - CVD divergence: 0-30 points
    - Funding rate: 0-30 points

    Total: 0-100 points
    """

    def __init__(self, client=None):
        """
        Initialize conviction scorer.

        Args:
            client: Exchange client (auto-creates if None)
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                logger.info("ConvictionScorer using MEXC")
            else:
                self.client = BinanceClient()
                logger.info("ConvictionScorer using Binance")
        else:
            self.client = client

        # Initialize strategy modules
        self.rejection_detector = RejectionPatternDetector(client=self.client)
        self.cvd_analyzer = CVDAnalyzer(client=self.client)

        # Score weights
        self.weights = {
            "rejection_pattern": 40,
            "cvd_divergence": 30,
            "funding_rate": 30,
        }

        logger.info("ConvictionScorer initialized")

    def calculate_score(
        self, symbol: str, rejection_timeframe: str = "1h", cvd_timeframe: str = "15m"
    ) -> Dict:
        """
        Calculate conviction score for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            rejection_timeframe: Timeframe for rejection detection
            cvd_timeframe: Timeframe for CVD analysis

        Returns:
            Dict with score breakdown and total
        """
        logger.info(f"Calculating conviction score for {symbol}")

        scores = {}

        # 1. Rejection Pattern Score (0-40)
        rejection_result = self.rejection_detector.detect_pattern(
            symbol=symbol, timeframe=rejection_timeframe
        )

        rejection_score = self._score_rejection_pattern(rejection_result)
        scores["rejection_pattern"] = rejection_score

        # 2. CVD Divergence Score (0-30)
        cvd_result = self.cvd_analyzer.analyze_cvd(
            symbol=symbol, timeframe=cvd_timeframe
        )

        cvd_score = self._score_cvd_divergence(cvd_result)
        scores["cvd_divergence"] = cvd_score

        # 3. Funding Rate Score (0-30)
        funding_result = self._get_funding_rate(symbol)
        funding_score = self._score_funding_rate(funding_result)
        scores["funding_rate"] = funding_score

        # Calculate total
        total_score = sum(scores.values())

        logger.info(f"  Total conviction score: {total_score}/100")
        logger.info(
            f"    Rejection: {rejection_score}/{self.weights['rejection_pattern']}"
        )
        logger.info(f"    CVD: {cvd_score}/{self.weights['cvd_divergence']}")
        logger.info(f"    Funding: {funding_score}/{self.weights['funding_rate']}")

        return {
            "symbol": symbol,
            "total_score": total_score,
            "scores": scores,
            "weights": self.weights,
            "signals": {
                "rejection": rejection_result,
                "cvd": cvd_result,
                "funding": funding_result,
            },
        }

    def _score_rejection_pattern(self, result: Dict) -> float:
        """
        Score rejection pattern (0-40 points).

        Points based on:
        - Pattern valid: +20 points
        - Number of rejections: +5 per rejection (max +20)
        """
        score = 0

        if result["is_valid"]:
            score += 20  # Base score for valid pattern

            # Bonus for multiple rejections
            rejection_bonus = min((result["rejection_count"] - 2) * 5, 20)
            score += rejection_bonus

        return min(score, self.weights["rejection_pattern"])

    def _score_cvd_divergence(self, result: Dict) -> float:
        """
        Score CVD divergence (0-30 points).

        Points based on:
        - Divergence detected: +15 points
        - Divergence strength: +0 to +15 points
        """
        score = 0

        if result["divergence_detected"]:
            score += 15  # Base score for divergence

            # Bonus based on strength
            strength_bonus = result["divergence_strength"] * 15
            score += strength_bonus

        return min(score, self.weights["cvd_divergence"])

    def _score_funding_rate(self, result: Dict) -> float:
        """
        Score funding rate (0-30 points).

        Points based on:
        - Positive funding rate (longs paying shorts): +15 points
        - Rate magnitude: +0 to +15 points (higher = more overheated)
        """
        score = 0

        funding_rate = result.get("funding_rate", 0)

        if funding_rate > 0:
            score += 15  # Longs paying shorts

            # Bonus for high funding (indicates overheated longs)
            # Typical range: 0.01% to 0.1%
            rate_magnitude = (
                abs(funding_rate) / 0.001
            )  # Normalize to 0.1% = full points
            magnitude_bonus = min(rate_magnitude * 15, 15)
            score += magnitude_bonus

        return min(score, self.weights["funding_rate"])

    def _get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate."""
        try:
            funding_data = self.client.get_funding_rate(symbol)
            funding_rate = float(funding_data.get("fundingRate", 0))

            return {
                "symbol": symbol,
                "funding_rate": funding_rate,
                "funding_time": funding_data.get("fundingTime", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get funding rate: {e}")
            return {"symbol": symbol, "funding_rate": 0, "funding_time": 0}


if __name__ == "__main__":
    """Test conviction scorer"""

    print("\n" + "=" * 80)
    print("CONVICTION SCORER - TESTING")
    print("=" * 80 + "\n")

    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")

    # Initialize scorer (auto-creates client)
    scorer = ConvictionScorer()

    # Calculate conviction score
    print("TEST: Calculate Conviction Score for BTC")
    print("-" * 40)

    result = scorer.calculate_score(
        symbol="BTCUSDT", rejection_timeframe="1h", cvd_timeframe="15m"
    )

    print(f"\nSymbol: {result['symbol']}")
    print(f"Total Score: {result['total_score']}/100")
    print(f"\nBreakdown:")
    print(
        f"  Rejection Pattern: {result['scores']['rejection_pattern']}/{result['weights']['rejection_pattern']}"
    )
    print(
        f"  CVD Divergence: {result['scores']['cvd_divergence']}/{result['weights']['cvd_divergence']}"
    )
    print(
        f"  Funding Rate: {result['scores']['funding_rate']}/{result['weights']['funding_rate']}"
    )

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
=======
"""
CONVICTION SCORER
=================
Combines all signals into a single conviction score (0-100).

Supports both Binance and MEXC exchanges.

Author: Grim (Institutional Standards)
"""

import os

# Import both exchange clients
import sys
from pathlib import Path
from typing import Dict

from loguru import logger

from .cvd_analyzer import CVDAnalyzer

# Import strategy modules - USE RELATIVE IMPORTS
from .rejection_pattern import RejectionPatternDetector

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv("USE_MEXC", "False").lower() == "true"


class ConvictionScorer:
    """
    Calculates conviction score by combining multiple signals.

    Score breakdown:
    - Rejection pattern: 0-40 points
    - CVD divergence: 0-30 points
    - Funding rate: 0-30 points

    Total: 0-100 points
    """

    def __init__(self, client=None):
        """
        Initialize conviction scorer.

        Args:
            client: Exchange client (auto-creates if None)
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                logger.info("ConvictionScorer using MEXC")
            else:
                self.client = BinanceClient()
                logger.info("ConvictionScorer using Binance")
        else:
            self.client = client

        # Initialize strategy modules
        self.rejection_detector = RejectionPatternDetector(client=self.client)
        self.cvd_analyzer = CVDAnalyzer(client=self.client)

        # Score weights
        self.weights = {
            "rejection_pattern": 40,
            "cvd_divergence": 30,
            "funding_rate": 30,
        }

        logger.info("ConvictionScorer initialized")

    def calculate_score(
        self, symbol: str, rejection_timeframe: str = "1h", cvd_timeframe: str = "15m"
    ) -> Dict:
        """
        Calculate conviction score for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            rejection_timeframe: Timeframe for rejection detection
            cvd_timeframe: Timeframe for CVD analysis

        Returns:
            Dict with score breakdown and total
        """
        logger.info(f"Calculating conviction score for {symbol}")

        scores = {}

        # 1. Rejection Pattern Score (0-40)
        rejection_result = self.rejection_detector.detect_pattern(
            symbol=symbol, timeframe=rejection_timeframe
        )

        rejection_score = self._score_rejection_pattern(rejection_result)
        scores["rejection_pattern"] = rejection_score

        # 2. CVD Divergence Score (0-30)
        cvd_result = self.cvd_analyzer.analyze_cvd(
            symbol=symbol, timeframe=cvd_timeframe
        )

        cvd_score = self._score_cvd_divergence(cvd_result)
        scores["cvd_divergence"] = cvd_score

        # 3. Funding Rate Score (0-30)
        funding_result = self._get_funding_rate(symbol)
        funding_score = self._score_funding_rate(funding_result)
        scores["funding_rate"] = funding_score

        # Calculate total
        total_score = sum(scores.values())

        logger.info(f"  Total conviction score: {total_score}/100")
        logger.info(
            f"    Rejection: {rejection_score}/{self.weights['rejection_pattern']}"
        )
        logger.info(f"    CVD: {cvd_score}/{self.weights['cvd_divergence']}")
        logger.info(f"    Funding: {funding_score}/{self.weights['funding_rate']}")

        return {
            "symbol": symbol,
            "total_score": total_score,
            "scores": scores,
            "weights": self.weights,
            "signals": {
                "rejection": rejection_result,
                "cvd": cvd_result,
                "funding": funding_result,
            },
        }

    def _score_rejection_pattern(self, result: Dict) -> float:
        """
        Score rejection pattern (0-40 points).

        Points based on:
        - Pattern valid: +20 points
        - Number of rejections: +5 per rejection (max +20)
        """
        score = 0

        if result["is_valid"]:
            score += 20  # Base score for valid pattern

            # Bonus for multiple rejections
            rejection_bonus = min((result["rejection_count"] - 2) * 5, 20)
            score += rejection_bonus

        return min(score, self.weights["rejection_pattern"])

    def _score_cvd_divergence(self, result: Dict) -> float:
        """
        Score CVD divergence (0-30 points).

        Points based on:
        - Divergence detected: +15 points
        - Divergence strength: +0 to +15 points
        """
        score = 0

        if result["divergence_detected"]:
            score += 15  # Base score for divergence

            # Bonus based on strength
            strength_bonus = result["divergence_strength"] * 15
            score += strength_bonus

        return min(score, self.weights["cvd_divergence"])

    def _score_funding_rate(self, result: Dict) -> float:
        """
        Score funding rate (0-30 points).

        Points based on:
        - Positive funding rate (longs paying shorts): +15 points
        - Rate magnitude: +0 to +15 points (higher = more overheated)
        """
        score = 0

        funding_rate = result.get("funding_rate", 0)

        if funding_rate > 0:
            score += 15  # Longs paying shorts

            # Bonus for high funding (indicates overheated longs)
            # Typical range: 0.01% to 0.1%
            rate_magnitude = (
                abs(funding_rate) / 0.001
            )  # Normalize to 0.1% = full points
            magnitude_bonus = min(rate_magnitude * 15, 15)
            score += magnitude_bonus

        return min(score, self.weights["funding_rate"])

    def _get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate."""
        try:
            funding_data = self.client.get_funding_rate(symbol)
            funding_rate = float(funding_data.get("fundingRate", 0))

            return {
                "symbol": symbol,
                "funding_rate": funding_rate,
                "funding_time": funding_data.get("fundingTime", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get funding rate: {e}")
            return {"symbol": symbol, "funding_rate": 0, "funding_time": 0}


if __name__ == "__main__":
    """Test conviction scorer"""

    print("\n" + "=" * 80)
    print("CONVICTION SCORER - TESTING")
    print("=" * 80 + "\n")

    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")

    # Initialize scorer (auto-creates client)
    scorer = ConvictionScorer()

    # Calculate conviction score
    print("TEST: Calculate Conviction Score for BTC")
    print("-" * 40)

    result = scorer.calculate_score(
        symbol="BTCUSDT", rejection_timeframe="1h", cvd_timeframe="15m"
    )

    print(f"\nSymbol: {result['symbol']}")
    print(f"Total Score: {result['total_score']}/100")
    print(f"\nBreakdown:")
    print(
        f"  Rejection Pattern: {result['scores']['rejection_pattern']}/{result['weights']['rejection_pattern']}"
    )
    print(
        f"  CVD Divergence: {result['scores']['cvd_divergence']}/{result['weights']['cvd_divergence']}"
    )
    print(
        f"  Funding Rate: {result['scores']['funding_rate']}/{result['weights']['funding_rate']}"
    )

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
    print("=" * 80)