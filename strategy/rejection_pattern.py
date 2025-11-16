"""
REJECTION PATTERN DETECTOR
==========================
Detects the "3 pumps rule" - multiple rejections at resistance levels.

WEEK 4 ENHANCEMENTS:
- Quality scoring for individual rejections (0-100 points)
- Weighted scoring (3rd rejection = highest weight)
- 5+ rejection warning (breakout risk)

Supports both Binance and MEXC exchanges.

Author: Grim (Institutional Standards)
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from loguru import logger

# Import both exchange clients
from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv('USE_MEXC', 'False').lower() == 'true'


class RejectionPatternDetector:
    """
    Detects exhaustion patterns via resistance rejections.
    
    The "3 pumps rule":
    - Price makes 2+ attempts to break resistance
    - Each attempt forms a rejection candle (long wick)
    - Rejections cluster at same price level (~0.5% tolerance)
    
    WEEK 4: Now scores rejection QUALITY, not just quantity
    """
    
    def __init__(
        self, 
        client=None,
        lookback_period: int = 24,
        min_rejections: int = 2,
        resistance_tolerance: float = 0.005
    ):
        """
        Initialize rejection pattern detector.
        
        Args:
            client: Exchange client (auto-creates if None)
            lookback_period: Number of candles to analyze
            min_rejections: Minimum rejection count for valid pattern
            resistance_tolerance: Price tolerance for clustering (0.5%)
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                logger.info("RejectionPatternDetector using MEXC")
            else:
                self.client = BinanceClient()
                logger.info("RejectionPatternDetector using Binance")
        else:
            self.client = client
        
        self.lookback_period = lookback_period
        self.min_rejections = min_rejections
        self.resistance_tolerance = resistance_tolerance
        
        logger.info(f"RejectionPatternDetector initialized (lookback={lookback_period}, min_rejections={min_rejections})")
    
    
    def detect_pattern(
        self,
        symbol: str,
        timeframe: str = '1h',
        lookback_candles: int = None
    ) -> Dict:
        """
        Detect rejection pattern for given symbol.
        
        WEEK 4: Enhanced with quality scoring
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candlestick interval
            lookback_candles: Override default lookback period
            
        Returns:
            Dict with pattern detection results + quality scores
        """
        if lookback_candles is None:
            lookback_candles = self.lookback_period
        
        logger.info(f"Detecting pattern for {symbol} ({timeframe})")
        
        # Get historical candles
        klines = self.client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=lookback_candles
        )
        
        if not klines:
            logger.warning(f"No kline data for {symbol}")
            return self._empty_result(symbol)
        
        # Convert to DataFrame
        df = self._klines_to_df(klines)
        
        # Find rejection candles (now with quality data)
        rejections = self._find_rejections(df)
        
        if len(rejections) < self.min_rejections:
            logger.info(f"  Only {len(rejections)} rejections found (need {self.min_rejections})")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'rejection_count': len(rejections),
                'is_valid': False,
                'resistance_level': 0,
                'rejections': rejections,
                'reason': f'Insufficient rejections ({len(rejections)} < {self.min_rejections})'
            }
        
        # Cluster rejections to find resistance level
        resistance_level = self._find_resistance_level(rejections)
        
        # Validate that rejections cluster at resistance
        clustered_rejections = [
            r for r in rejections 
            if abs(r['high'] - resistance_level) / resistance_level <= self.resistance_tolerance
        ]
        
        # WEEK 4: Score quality of each rejection
        for rejection in clustered_rejections:
            quality = self.score_rejection_quality(
                candle=rejection['candle_data'],
                resistance_level=resistance_level,
                avg_volume=rejection['avg_volume']
            )
            rejection['quality_score'] = quality['quality_score']
            rejection['quality_breakdown'] = quality
        
        # WEEK 4: Get weighted score for entire pattern
        weighted_score = self.get_weighted_rejection_score(clustered_rejections)
        
        is_valid = len(clustered_rejections) >= self.min_rejections
        
        logger.info(f"  Found {len(clustered_rejections)} rejections at ${resistance_level:.2f}")
        logger.info(f"  Average quality: {weighted_score['avg_quality']:.1f}/100")
        logger.info(f"  Weighted score: {weighted_score['total_score']:.1f}")
        logger.info(f"  Pattern valid: {is_valid}")
        
        if weighted_score['warning']:
            logger.warning(f"  âš ï¸ {weighted_score['warning']}")
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'rejection_count': len(clustered_rejections),
            'is_valid': is_valid,
            'resistance_level': resistance_level,
            'rejections': clustered_rejections,
            'weighted_score': weighted_score,  # WEEK 4 addition
            'reason': 'Valid pattern' if is_valid else 'Rejections not clustered at resistance'
        }
    
    
    def score_rejection_quality(self, candle: Dict, resistance_level: float, avg_volume: float = None) -> Dict[str, float]:
        """
        Score individual rejection quality (0-100 points)
        
        WEEK 4 ENHANCEMENT: Quality matters more than quantity
        
        Quality indicators:
        - Wick length (longer = stronger rejection)
        - Volume (higher = more conviction)
        - Close position (lower close = stronger rejection)
        - Distance from resistance (closer = cleaner test)
        
        Args:
            candle: OHLCV dict with keys: open, high, low, close, volume
            resistance_level: Key resistance being tested
            avg_volume: Recent average volume (optional)
            
        Returns:
            Dict with quality_score and breakdown
        """
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close = candle['close']
        volume = candle['volume']
        
        # Calculate candle components
        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        full_range = high - low
        
        # Avoid division by zero
        if full_range == 0:
            return {"quality_score": 0, "reason": "No price movement"}
        
        # === WICK LENGTH SCORE (0-40 points) ===
        # Upper wick should be significant relative to body
        wick_to_body_ratio = upper_wick / body if body > 0 else 10  # Large wick, tiny body = bearish
        
        if wick_to_body_ratio >= 3:
            wick_score = 40  # Huge wick, tiny body = strong rejection
        elif wick_to_body_ratio >= 2:
            wick_score = 30  # Decent wick
        elif wick_to_body_ratio >= 1:
            wick_score = 20  # Wick = body size
        else:
            wick_score = 10  # Small wick
        
        # === CLOSE POSITION SCORE (0-25 points) ===
        # Close should be in lower half of candle (bearish)
        close_position = (close - low) / full_range  # 0 = closed at low, 1 = closed at high
        
        if close_position <= 0.25:
            close_score = 25  # Closed in bottom 25%
        elif close_position <= 0.40:
            close_score = 20  # Closed in bottom 40%
        elif close_position <= 0.50:
            close_score = 15  # Closed in bottom 50%
        else:
            close_score = 5  # Closed above midpoint (weak)
        
        # === RESISTANCE TEST ACCURACY (0-20 points) ===
        # High should be very close to resistance level
        distance_from_resistance = abs(high - resistance_level) / resistance_level
        
        if distance_from_resistance <= 0.002:  # Within 0.2%
            resistance_score = 20  # Perfect test
        elif distance_from_resistance <= 0.005:  # Within 0.5%
            resistance_score = 15  # Good test
        elif distance_from_resistance <= 0.01:  # Within 1%
            resistance_score = 10  # Acceptable test
        else:
            resistance_score = 5  # Loose test
        
        # === VOLUME CONFIRMATION (0-15 points) ===
        if avg_volume:
            if volume >= avg_volume * 2:
                volume_score = 15  # 2X+ volume
            elif volume >= avg_volume * 1.5:
                volume_score = 10  # 1.5X+ volume
            elif volume >= avg_volume:
                volume_score = 5  # Normal volume
            else:
                volume_score = 0  # Below average (weak signal)
        else:
            volume_score = 5  # No comparison data, neutral score
        
        # === TOTAL QUALITY SCORE ===
        total_score = wick_score + close_score + resistance_score + volume_score
        
        return {
            "quality_score": total_score,
            "wick_score": wick_score,
            "close_score": close_score,
            "resistance_score": resistance_score,
            "volume_score": volume_score,
            "wick_to_body_ratio": round(wick_to_body_ratio, 2),
            "close_position_pct": round(close_position * 100, 1),
            "distance_from_resistance_pct": round(distance_from_resistance * 100, 2)
        }

    def get_weighted_rejection_score(self, rejections: List[Dict]) -> Dict[str, float]:
        """
        Score entire rejection sequence with quality weighting
        
        WEEK 4 ENHANCEMENT: 3rd rejection = biggest conviction jump
        
        Scoring philosophy:
        - 3rd rejection = highest weight (critical exhaustion point)
        - Quality matters more than quantity
        - 5+ rejections = warning (might break through)
        
        Args:
            rejections: List of rejection dicts with quality_score
            
        Returns:
            Dict with total_score, avg_quality, warning flags
        """
        if not rejections:
            return {"total_score": 0, "avg_quality": 0, "count": 0}
        
        count = len(rejections)
        
        # Calculate average quality
        avg_quality = sum(r.get('quality_score', 50) for r in rejections) / count
        
        # Base score from count (3 rejections = baseline)
        if count >= 5:
            count_score = 25  # Cap at 5 (more might break through)
            warning = "5+ rejections - risk of breakout"
        elif count == 4:
            count_score = 30  # Strong exhaustion
            warning = None
        elif count == 3:
            count_score = 25  # Qualified setup
            warning = None
        elif count == 2:
            count_score = 15  # Early signal
            warning = "Only 2 rejections - wait for 3rd"
        else:
            count_score = 5   # Too early
            warning = "Insufficient rejections"
        
        # Quality multiplier (avg quality 50+ = 1.0X, 70+ = 1.2X, 80+ = 1.4X)
        if avg_quality >= 80:
            quality_mult = 1.4
        elif avg_quality >= 70:
            quality_mult = 1.2
        elif avg_quality >= 50:
            quality_mult = 1.0
        else:
            quality_mult = 0.8  # Low quality rejections
        
        # Weight 3rd rejection most heavily (THE CRITICAL ONE)
        if count >= 3:
            third_rejection_bonus = rejections[2].get('quality_score', 50) * 0.3  # 30% bonus from 3rd
        else:
            third_rejection_bonus = 0
        
        total_score = (count_score * quality_mult) + third_rejection_bonus
        
        return {
            "total_score": round(total_score, 1),
            "avg_quality": round(avg_quality, 1),
            "count": count,
            "quality_multiplier": quality_mult,
            "third_rejection_bonus": round(third_rejection_bonus, 1),
            "warning": warning
        }
    
    
    def _find_rejections(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find rejection candles (long upper wicks).
        
        ENHANCED: Now includes quality scoring data for each rejection
        
        A rejection candle has:
        - Upper wick > 1.5% of candle body
        - High is local peak (higher than neighbors)
        """
        rejections = []
        
        # Calculate average volume for quality scoring
        avg_volume = df['volume'].mean()
        
        for i in range(1, len(df) - 1):
            row = df.iloc[i]
            
            # Calculate wick and body sizes
            upper_wick = row['high'] - max(row['open'], row['close'])
            body = abs(row['close'] - row['open'])
            
            # Rejection criteria
            if body > 0 and upper_wick / body >= 1.5:
                # Check if high is local peak
                prev_high = df.iloc[i-1]['high']
                next_high = df.iloc[i+1]['high']
                
                if row['high'] >= prev_high and row['high'] >= next_high:
                    # Create candle dict for quality scoring
                    candle_dict = {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    
                    rejections.append({
                        'timestamp': row['timestamp'],
                        'high': row['high'],
                        'wick_size': upper_wick,
                        'body_size': body,
                        'wick_ratio': upper_wick / body,
                        'candle_data': candle_dict,  # Store for quality scoring later
                        'avg_volume': avg_volume
                    })
        
        return rejections
    
    
    def _find_resistance_level(self, rejections: List[Dict]) -> float:
        """
        Find resistance level by clustering rejection highs.
        
        Uses median of rejection highs as resistance.
        """
        if not rejections:
            return 0
        
        highs = [r['high'] for r in rejections]
        return sorted(highs)[len(highs) // 2]  # Median
    
    
    def _klines_to_df(self, klines: List[List]) -> pd.DataFrame:
        """Convert klines to DataFrame."""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    
    def _empty_result(self, symbol: str) -> Dict:
        """Return empty result when no data available."""
        return {
            'symbol': symbol,
            'rejection_count': 0,
            'is_valid': False,
            'resistance_level': 0,
            'rejections': [],
            'reason': 'No data available'
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test rejection pattern detector with Week 4 enhancements"""
    
    print("\n" + "="*80)
    print("REJECTION PATTERN DETECTOR - TESTING (WEEK 4 ENHANCED)")
    print("="*80 + "\n")
    
    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")
    
    # Initialize detector (auto-creates client)
    detector = RejectionPatternDetector()
    
    # Test pattern detection
    print("TEST: Detect Pattern on BTC with Quality Scoring")
    print("-" * 40)
    
    result = detector.detect_pattern(
        symbol='BTCUSDT',
        timeframe='1h',
        lookback_candles=24
    )
    
    print(f"Symbol: {result['symbol']}")
    print(f"Rejections found: {result['rejection_count']}")
    print(f"Resistance level: ${result['resistance_level']:,.2f}")
    print(f"Pattern valid: {result['is_valid']}")
    
    # WEEK 4: Show quality metrics
    if 'weighted_score' in result:
        ws = result['weighted_score']
        print(f"\nQuality Metrics:")
        print(f"  Average quality: {ws['avg_quality']:.1f}/100")
        print(f"  Weighted score: {ws['total_score']:.1f}")
        print(f"  Quality multiplier: {ws['quality_multiplier']:.2f}X")
        if ws.get('third_rejection_bonus'):
            print(f"  3rd rejection bonus: +{ws['third_rejection_bonus']:.1f} points")
        if ws.get('warning'):
            print(f"  âš ï¸ WARNING: {ws['warning']}")
    
    if result['rejections']:
        print(f"\nRejection details:")
        for idx, r in enumerate(result['rejections'][:3], 1):
            quality = r.get('quality_breakdown', {})
            print(f"  #{idx} - {r['timestamp']}: High ${r['high']:.2f}")
            print(f"       Wick: {r['wick_ratio']:.2f}x body")
            if quality:
                print(f"       Quality: {quality.get('quality_score', 0)}/100")
                print(f"         - Wick: {quality.get('wick_score', 0)}/40")
                print(f"         - Close: {quality.get('close_score', 0)}/25")
                print(f"         - Resistance test: {quality.get('resistance_score', 0)}/20")
                print(f"         - Volume: {quality.get('volume_score', 0)}/15")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)