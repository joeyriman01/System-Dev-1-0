<<<<<<< HEAD
"""
SIGNAL GENERATOR
================
Generates complete trade signals with entry, stops, and targets.

WEEK 4 ENHANCEMENTS:
- Volume profile support detection
- Swing low identification
- R:R ratio calculation for each target
- Intelligent partial sizing recommendations

Supports both Binance and MEXC exchanges.

Author: Grim (Institutional Standards)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

# Add parent directory to path for data imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import conviction scorer - handle both module and script execution
try:
    from .conviction_scorer import ConvictionScorer
except ImportError:
    from conviction_scorer import ConvictionScorer

from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv("USE_MEXC", "False").lower() == "true"


class SignalGenerator:
    """
    Generates actionable trade signals.

    WEEK 4: Enhanced with volume profile support detection
    
    Signal structure:
    - Entry price: Current market price
    - Stop loss: Just above resistance level
    - Take profit levels: Based on support zones (volume profile + swing lows)
    - Position size: Based on conviction score
    """

    def __init__(
        self,
        client=None,
        min_conviction: float = 60,
        base_position_size: float = 100,  # Base position in USD
    ):
        """
        Initialize signal generator.

        Args:
            client: Exchange client (auto-creates if None)
            min_conviction: Minimum score to generate signal (0-100)
            base_position_size: Base position size in USD
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                logger.info("SignalGenerator using MEXC")
            else:
                self.client = BinanceClient()
                logger.info("SignalGenerator using Binance")
        else:
            self.client = client

        self.min_conviction = min_conviction
        self.base_position_size = base_position_size

        # Initialize conviction scorer
        self.scorer = ConvictionScorer(client=self.client)

        logger.info(f"SignalGenerator initialized (min_conviction={min_conviction})")

    def generate_signal(
        self, 
        symbol: str, 
        rejection_timeframe: str = "1h", 
        cvd_timeframe: str = "15m",
        support_lookback_days: int = 30
    ) -> Optional[Dict]:
        """
        Generate trade signal for symbol.
        
        WEEK 4: Enhanced with volume profile support detection

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            rejection_timeframe: Timeframe for pattern detection
            cvd_timeframe: Timeframe for CVD analysis
            support_lookback_days: Days of history for support detection

        Returns:
            Trade signal dict or None if conviction too low
        """
        logger.info(f"Generating signal for {symbol}")

        # Calculate conviction score
        conviction_result = self.scorer.calculate_score(
            symbol=symbol,
            rejection_timeframe=rejection_timeframe,
            cvd_timeframe=cvd_timeframe,
        )

        total_score = conviction_result["total_score"]

        # Check if conviction meets minimum
        if total_score < self.min_conviction:
            logger.info(f"  Conviction too low: {total_score} < {self.min_conviction}")
            return None

        # Get current price
        ticker = self.client.get_ticker(symbol)
        current_price = float(ticker.get("lastPrice", 0))

        if current_price == 0:
            logger.error(f"Failed to get price for {symbol}")
            return None

        # Get resistance level from rejection pattern
        rejection_data = conviction_result["signals"]["rejection"]
        resistance_level = rejection_data.get("resistance_level", current_price * 1.02)

        # Calculate position size based on conviction
        position_multiplier = total_score / 100  # 0.6 to 1.0 for scores 60-100
        position_size_usd = self.base_position_size * position_multiplier

        # Calculate stop loss (2% above resistance)
        stop_loss = resistance_level * 1.02
        stop_distance_pct = abs(stop_loss - current_price) / current_price * 100

        # WEEK 4: Detect support levels using volume profile
        logger.info(f"  Detecting support levels...")
        support_levels = self.detect_support_levels(
            symbol=symbol,
            current_price=current_price,
            lookback_days=support_lookback_days
        )

        # Calculate profit targets with R:R ratios
        targets = self.calculate_profit_targets(
            support_levels=support_levels,
            entry_price=current_price,
            stop_price=stop_loss
        )

        # If we have good targets, use them; otherwise fall back to % targets
        if len(targets) >= 3:
            tp1 = targets[0]['target_price']
            tp2 = targets[1]['target_price']
            tp3 = targets[2]['target_price']
            
            tp1_r = targets[0]['r_ratio']
            tp2_r = targets[1]['r_ratio']
            tp3_r = targets[2]['r_ratio']
        else:
            # Fallback to simple % targets
            logger.warning("  Insufficient support levels, using % targets")
            tp1 = current_price * 0.98  # -2%
            tp2 = current_price * 0.96  # -4%
            tp3 = current_price * 0.94  # -6%
            
            stop_dist = abs(stop_loss - current_price)
            tp1_r = abs(current_price - tp1) / stop_dist
            tp2_r = abs(current_price - tp2) / stop_dist
            tp3_r = abs(current_price - tp3) / stop_dist

        # Build signal
        signal = {
            "symbol": symbol,
            "direction": "SHORT",
            "conviction_score": total_score,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "stop_distance_pct": round(stop_distance_pct, 2),
            "take_profits": {
                "tp1": {
                    "price": tp1, 
                    "size": 0.25,  # 25% at TP1
                    "r_ratio": round(tp1_r, 2),
                    "distance_pct": round(abs(current_price - tp1) / current_price * 100, 2)
                },
                "tp2": {
                    "price": tp2, 
                    "size": 0.25,  # 25% at TP2
                    "r_ratio": round(tp2_r, 2),
                    "distance_pct": round(abs(current_price - tp2) / current_price * 100, 2)
                },
                "tp3": {
                    "price": tp3, 
                    "size": 0.25,  # 25% at TP3
                    "r_ratio": round(tp3_r, 2),
                    "distance_pct": round(abs(current_price - tp3) / current_price * 100, 2)
                },
                "runner": {
                    "size": 0.25,  # 25% runner
                    "target": "exhaustion_reversal"
                }
            },
            "position_size_usd": position_size_usd,
            "avg_risk_reward_ratio": round((tp1_r + tp2_r + tp3_r) / 3, 2),
            "resistance_level": resistance_level,
            "support_levels": support_levels[:5],  # Top 5 supports
            "timestamp": "now",
            "signals_breakdown": conviction_result,
        }

        logger.info(f"  [OK] SIGNAL GENERATED")
        logger.info(f"    Conviction: {total_score}/100")
        logger.info(f"    Entry: ${current_price:.4f}")
        logger.info(f"    Stop: ${stop_loss:.4f} ({stop_distance_pct:.2f}%)")
        logger.info(f"    TP1: ${tp1:.4f} (25%, {tp1_r:.2f}R)")
        logger.info(f"    TP2: ${tp2:.4f} (25%, {tp2_r:.2f}R)")
        logger.info(f"    TP3: ${tp3:.4f} (25%, {tp3_r:.2f}R)")
        logger.info(f"    Avg R:R: {signal['avg_risk_reward_ratio']:.2f}")

        return signal

    def detect_support_levels(
        self, 
        symbol: str, 
        current_price: float,
        lookback_days: int = 30
    ) -> List[Dict]:
        """
        Detect high-volume price levels that act as support (profit targets)
        
        WEEK 4 ENHANCEMENT: Volume Profile analysis
        
        Uses Volume Profile to find price levels where most trading occurred.
        These levels = likely support on the way down.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            lookback_days: Days of historical data to analyze
            
        Returns:
            List of support levels sorted by strength
        """
        # Get historical data for volume profile
        # Use daily candles for volume profile (more stable)
        limit = lookback_days  # 30 days = 30 candles
        
        klines = self.client.get_klines(
            symbol=symbol,
            interval='1d',
            limit=limit
        )
        
        if not klines:
            logger.warning(f"No historical data for {symbol}")
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate price bins (1% increments)
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_range = max_price - min_price
        
        # Create 100 price bins
        num_bins = 100
        bin_size = price_range / num_bins
        
        # Initialize volume profile
        volume_profile = {}
        
        for _, candle in df.iterrows():
            # Determine which bins this candle touched
            candle_low = candle['low']
            candle_high = candle['high']
            candle_volume = candle['volume']
            
            # Distribute volume across bins touched by this candle
            low_bin = int((candle_low - min_price) / bin_size)
            high_bin = int((candle_high - min_price) / bin_size)
            
            bins_touched = max(1, high_bin - low_bin + 1)
            volume_per_bin = candle_volume / bins_touched
            
            for bin_idx in range(low_bin, high_bin + 1):
                if bin_idx not in volume_profile:
                    volume_profile[bin_idx] = 0
                volume_profile[bin_idx] += volume_per_bin
        
        # Find high volume nodes (HVN = support levels)
        total_volume = sum(volume_profile.values())
        avg_volume_per_bin = total_volume / len(volume_profile) if volume_profile else 1
        
        support_levels = []
        
        for bin_idx, volume in volume_profile.items():
            # Only consider bins with >1.5X average volume
            if volume >= avg_volume_per_bin * 1.5:
                # Calculate price at center of bin
                price_level = min_price + (bin_idx * bin_size) + (bin_size / 2)
                
                # Only include levels BELOW current price (support, not resistance)
                if price_level < current_price * 0.95:  # At least 5% below
                    
                    # Calculate distance from current price
                    distance_pct = ((current_price - price_level) / current_price) * 100
                    
                    # Strength score based on volume concentration
                    strength = (volume / avg_volume_per_bin) * 10  # Scale to ~10-50 range
                    
                    support_levels.append({
                        "price": round(price_level, 8),
                        "volume": volume,
                        "strength": round(strength, 1),
                        "distance_pct": round(distance_pct, 2),
                        "type": "volume_profile_hvn"
                    })
        
        # Sort by strength (strongest first)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        # Add previous swing lows as support
        swing_lows = self._find_swing_lows(df, current_price)
        for low in swing_lows:
            support_levels.append(low)
        
        # Deduplicate levels that are too close (within 2%)
        deduplicated = []
        for level in support_levels:
            # Check if similar level already exists
            is_duplicate = False
            for existing in deduplicated:
                if abs(level['price'] - existing['price']) / existing['price'] < 0.02:
                    is_duplicate = True
                    # Keep the stronger one
                    if level['strength'] > existing['strength']:
                        deduplicated.remove(existing)
                        deduplicated.append(level)
                    break
            
            if not is_duplicate:
                deduplicated.append(level)
        
        # Re-sort by distance (closest first = first target)
        deduplicated.sort(key=lambda x: x['distance_pct'])
        
        logger.info(f"  Detected {len(deduplicated)} support levels below ${current_price:.4f}")
        
        return deduplicated[:5]  # Return top 5 targets

    def _find_swing_lows(
        self, 
        df: pd.DataFrame, 
        current_price: float, 
        window: int = 5
    ) -> List[Dict]:
        """
        Find significant swing lows that act as support
        
        Args:
            df: OHLCV dataframe
            current_price: Current price
            window: Lookback window for swing detection
            
        Returns:
            List of swing low support levels
        """
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Check if this is a local minimum
            current_low = df['low'].iloc[i]
            
            # Must be lower than surrounding candles
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] < current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low and current_low < current_price * 0.95:
                distance_pct = ((current_price - current_low) / current_price) * 100
                
                # Strength based on how many times price bounced off this level
                touches = self._count_support_touches(df, current_low, tolerance=0.02)
                
                swing_lows.append({
                    "price": round(current_low, 8),
                    "strength": touches * 15,  # Each touch = +15 strength
                    "distance_pct": round(distance_pct, 2),
                    "type": "swing_low",
                    "touches": touches
                })
        
        return swing_lows

    def _count_support_touches(
        self, 
        df: pd.DataFrame, 
        level: float, 
        tolerance: float = 0.02
    ) -> int:
        """Count how many times price touched a support level"""
        touches = 0
        for _, candle in df.iterrows():
            if abs(candle['low'] - level) / level <= tolerance:
                touches += 1
        return touches

    def calculate_profit_targets(
        self, 
        support_levels: List[Dict], 
        entry_price: float, 
        stop_price: float
    ) -> List[Dict]:
        """
        Calculate profit targets based on support levels and R:R ratios
        
        WEEK 4 ENHANCEMENT: R:R aware target selection
        
        Args:
            support_levels: Detected support levels
            entry_price: Trade entry price
            stop_price: Stop loss price
            
        Returns:
            List of profit targets with R:R ratios
        """
        stop_distance = abs(entry_price - stop_price)
        
        targets = []
        for level in support_levels:
            target_price = level['price']
            profit_distance = abs(entry_price - target_price)
            
            # Calculate R:R ratio
            r_ratio = profit_distance / stop_distance if stop_distance > 0 else 0
            
            # Only include targets that are worth at least 1R
            if r_ratio >= 1.0:
                targets.append({
                    "target_price": target_price,
                    "distance_pct": level['distance_pct'],
                    "r_ratio": round(r_ratio, 2),
                    "strength": level['strength'],
                    "type": level['type'],
                    "recommended_partial": self._suggest_partial_size(r_ratio)
                })
        
        # Sort by R:R (take best targets first)
        targets.sort(key=lambda x: x['r_ratio'])
        
        return targets

    def _suggest_partial_size(self, r_ratio: float) -> str:
        """Suggest what % to take off at this target"""
        if r_ratio >= 5:
            return "25% (5R+ runner material)"
        elif r_ratio >= 3:
            return "25% (strong 3R+ target)"
        elif r_ratio >= 2:
            return "33% (decent 2R target)"
        elif r_ratio >= 1:
            return "50% (1R safety partial)"
        else:
            return "SKIP (not worth 1R)"


if __name__ == "__main__":
    """Test signal generator with Week 4 enhancements"""

    print("\n" + "=" * 80)
    print("SIGNAL GENERATOR - TESTING (WEEK 4 ENHANCED)")
    print("=" * 80 + "\n")

    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")

    # Initialize generator (auto-creates client)
    generator = SignalGenerator(min_conviction=40)  # Lower threshold for testing

    # Generate signal
    print("TEST: Generate Signal for BTC with Volume Profile Targets")
    print("-" * 40)

    signal = generator.generate_signal(
        symbol="BTCUSDT", 
        rejection_timeframe="1h", 
        cvd_timeframe="15m",
        support_lookback_days=30
    )

    if signal:
        print(f"\nâœ" SIGNAL GENERATED:")
        print(f"  Symbol: {signal['symbol']}")
        print(f"  Direction: {signal['direction']}")
        print(f"  Conviction: {signal['conviction_score']}/100")
        print(f"  Entry: ${signal['entry_price']:,.4f}")
        print(f"  Stop Loss: ${signal['stop_loss']:,.4f} ({signal['stop_distance_pct']:.2f}%)")
        print(f"\n  Take Profits:")
        for tp_name, tp_data in signal["take_profits"].items():
            if tp_name != 'runner':
                print(f"    {tp_name.upper()}: ${tp_data['price']:,.4f}")
                print(f"      - Size: {tp_data['size']*100:.0f}%")
                print(f"      - R:R: {tp_data['r_ratio']:.2f}R")
                print(f"      - Distance: {tp_data['distance_pct']:.2f}%")
            else:
                print(f"    RUNNER: {tp_data['size']*100:.0f}% to {tp_data['target']}")
        
        print(f"\n  Position Size: ${signal['position_size_usd']:.2f}")
        print(f"  Avg Risk:Reward: {signal['avg_risk_reward_ratio']:.2f}")
        
        # Show support levels detected
        if signal.get('support_levels'):
            print(f"\n  Support Levels Detected:")
            for idx, support in enumerate(signal['support_levels'][:3], 1):
                print(f"    #{idx}: ${support['price']:.4f} ({support['type']})")
                print(f"        Strength: {support['strength']:.1f}, Distance: {support['distance_pct']:.2f}%")
    else:
        print("\nâœ— NO SIGNAL (conviction too low)")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
=======
"""
SIGNAL GENERATOR
================
Generates complete trade signals with entry, stops, and targets.

WEEK 4 ENHANCEMENTS:
- Volume profile support detection
- Swing low identification
- R:R ratio calculation for each target
- Intelligent partial sizing recommendations

Supports both Binance and MEXC exchanges.

Author: Grim (Institutional Standards)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

# Import conviction scorer - USE RELATIVE IMPORT
from .conviction_scorer import ConvictionScorer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv("USE_MEXC", "False").lower() == "true"


class SignalGenerator:
    """
    Generates actionable trade signals.

    WEEK 4: Enhanced with volume profile support detection
    
    Signal structure:
    - Entry price: Current market price
    - Stop loss: Just above resistance level
    - Take profit levels: Based on support zones (volume profile + swing lows)
    - Position size: Based on conviction score
    """

    def __init__(
        self,
        client=None,
        min_conviction: float = 60,
        base_position_size: float = 100,  # Base position in USD
    ):
        """
        Initialize signal generator.

        Args:
            client: Exchange client (auto-creates if None)
            min_conviction: Minimum score to generate signal (0-100)
            base_position_size: Base position size in USD
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                logger.info("SignalGenerator using MEXC")
            else:
                self.client = BinanceClient()
                logger.info("SignalGenerator using Binance")
        else:
            self.client = client

        self.min_conviction = min_conviction
        self.base_position_size = base_position_size

        # Initialize conviction scorer
        self.scorer = ConvictionScorer(client=self.client)

        logger.info(f"SignalGenerator initialized (min_conviction={min_conviction})")

    def generate_signal(
        self, 
        symbol: str, 
        rejection_timeframe: str = "1h", 
        cvd_timeframe: str = "15m",
        support_lookback_days: int = 30
    ) -> Optional[Dict]:
        """
        Generate trade signal for symbol.
        
        WEEK 4: Enhanced with volume profile support detection

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            rejection_timeframe: Timeframe for pattern detection
            cvd_timeframe: Timeframe for CVD analysis
            support_lookback_days: Days of history for support detection

        Returns:
            Trade signal dict or None if conviction too low
        """
        logger.info(f"Generating signal for {symbol}")

        # Calculate conviction score
        conviction_result = self.scorer.calculate_score(
            symbol=symbol,
            rejection_timeframe=rejection_timeframe,
            cvd_timeframe=cvd_timeframe,
        )

        total_score = conviction_result["total_score"]

        # Check if conviction meets minimum
        if total_score < self.min_conviction:
            logger.info(f"  Conviction too low: {total_score} < {self.min_conviction}")
            return None

        # Get current price
        ticker = self.client.get_ticker(symbol)
        current_price = float(ticker.get("lastPrice", 0))

        if current_price == 0:
            logger.error(f"Failed to get price for {symbol}")
            return None

        # Get resistance level from rejection pattern
        rejection_data = conviction_result["signals"]["rejection"]
        resistance_level = rejection_data.get("resistance_level", current_price * 1.02)

        # Calculate position size based on conviction
        position_multiplier = total_score / 100  # 0.6 to 1.0 for scores 60-100
        position_size_usd = self.base_position_size * position_multiplier

        # Calculate stop loss (2% above resistance)
        stop_loss = resistance_level * 1.02
        stop_distance_pct = abs(stop_loss - current_price) / current_price * 100

        # WEEK 4: Detect support levels using volume profile
        logger.info(f"  Detecting support levels...")
        support_levels = self.detect_support_levels(
            symbol=symbol,
            current_price=current_price,
            lookback_days=support_lookback_days
        )

        # Calculate profit targets with R:R ratios
        targets = self.calculate_profit_targets(
            support_levels=support_levels,
            entry_price=current_price,
            stop_price=stop_loss
        )

        # If we have good targets, use them; otherwise fall back to % targets
        if len(targets) >= 3:
            tp1 = targets[0]['target_price']
            tp2 = targets[1]['target_price']
            tp3 = targets[2]['target_price']
            
            tp1_r = targets[0]['r_ratio']
            tp2_r = targets[1]['r_ratio']
            tp3_r = targets[2]['r_ratio']
        else:
            # Fallback to simple % targets
            logger.warning("  Insufficient support levels, using % targets")
            tp1 = current_price * 0.98  # -2%
            tp2 = current_price * 0.96  # -4%
            tp3 = current_price * 0.94  # -6%
            
            stop_dist = abs(stop_loss - current_price)
            tp1_r = abs(current_price - tp1) / stop_dist
            tp2_r = abs(current_price - tp2) / stop_dist
            tp3_r = abs(current_price - tp3) / stop_dist

        # Build signal
        signal = {
            "symbol": symbol,
            "direction": "SHORT",
            "conviction_score": total_score,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "stop_distance_pct": round(stop_distance_pct, 2),
            "take_profits": {
                "tp1": {
                    "price": tp1, 
                    "size": 0.25,  # 25% at TP1
                    "r_ratio": round(tp1_r, 2),
                    "distance_pct": round(abs(current_price - tp1) / current_price * 100, 2)
                },
                "tp2": {
                    "price": tp2, 
                    "size": 0.25,  # 25% at TP2
                    "r_ratio": round(tp2_r, 2),
                    "distance_pct": round(abs(current_price - tp2) / current_price * 100, 2)
                },
                "tp3": {
                    "price": tp3, 
                    "size": 0.25,  # 25% at TP3
                    "r_ratio": round(tp3_r, 2),
                    "distance_pct": round(abs(current_price - tp3) / current_price * 100, 2)
                },
                "runner": {
                    "size": 0.25,  # 25% runner
                    "target": "exhaustion_reversal"
                }
            },
            "position_size_usd": position_size_usd,
            "avg_risk_reward_ratio": round((tp1_r + tp2_r + tp3_r) / 3, 2),
            "resistance_level": resistance_level,
            "support_levels": support_levels[:5],  # Top 5 supports
            "timestamp": "now",
            "signals_breakdown": conviction_result,
        }

        logger.info(f"  âœ" SIGNAL GENERATED")
        logger.info(f"    Conviction: {total_score}/100")
        logger.info(f"    Entry: ${current_price:.4f}")
        logger.info(f"    Stop: ${stop_loss:.4f} ({stop_distance_pct:.2f}%)")
        logger.info(f"    TP1: ${tp1:.4f} (25%, {tp1_r:.2f}R)")
        logger.info(f"    TP2: ${tp2:.4f} (25%, {tp2_r:.2f}R)")
        logger.info(f"    TP3: ${tp3:.4f} (25%, {tp3_r:.2f}R)")
        logger.info(f"    Avg R:R: {signal['avg_risk_reward_ratio']:.2f}")

        return signal

    def detect_support_levels(
        self, 
        symbol: str, 
        current_price: float,
        lookback_days: int = 30
    ) -> List[Dict]:
        """
        Detect high-volume price levels that act as support (profit targets)
        
        WEEK 4 ENHANCEMENT: Volume Profile analysis
        
        Uses Volume Profile to find price levels where most trading occurred.
        These levels = likely support on the way down.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            lookback_days: Days of historical data to analyze
            
        Returns:
            List of support levels sorted by strength
        """
        # Get historical data for volume profile
        # Use daily candles for volume profile (more stable)
        limit = lookback_days  # 30 days = 30 candles
        
        klines = self.client.get_klines(
            symbol=symbol,
            interval='1d',
            limit=limit
        )
        
        if not klines:
            logger.warning(f"No historical data for {symbol}")
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate price bins (1% increments)
        min_price = df['low'].min()
        max_price = df['high'].max()
        price_range = max_price - min_price
        
        # Create 100 price bins
        num_bins = 100
        bin_size = price_range / num_bins
        
        # Initialize volume profile
        volume_profile = {}
        
        for _, candle in df.iterrows():
            # Determine which bins this candle touched
            candle_low = candle['low']
            candle_high = candle['high']
            candle_volume = candle['volume']
            
            # Distribute volume across bins touched by this candle
            low_bin = int((candle_low - min_price) / bin_size)
            high_bin = int((candle_high - min_price) / bin_size)
            
            bins_touched = max(1, high_bin - low_bin + 1)
            volume_per_bin = candle_volume / bins_touched
            
            for bin_idx in range(low_bin, high_bin + 1):
                if bin_idx not in volume_profile:
                    volume_profile[bin_idx] = 0
                volume_profile[bin_idx] += volume_per_bin
        
        # Find high volume nodes (HVN = support levels)
        total_volume = sum(volume_profile.values())
        avg_volume_per_bin = total_volume / len(volume_profile) if volume_profile else 1
        
        support_levels = []
        
        for bin_idx, volume in volume_profile.items():
            # Only consider bins with >1.5X average volume
            if volume >= avg_volume_per_bin * 1.5:
                # Calculate price at center of bin
                price_level = min_price + (bin_idx * bin_size) + (bin_size / 2)
                
                # Only include levels BELOW current price (support, not resistance)
                if price_level < current_price * 0.95:  # At least 5% below
                    
                    # Calculate distance from current price
                    distance_pct = ((current_price - price_level) / current_price) * 100
                    
                    # Strength score based on volume concentration
                    strength = (volume / avg_volume_per_bin) * 10  # Scale to ~10-50 range
                    
                    support_levels.append({
                        "price": round(price_level, 8),
                        "volume": volume,
                        "strength": round(strength, 1),
                        "distance_pct": round(distance_pct, 2),
                        "type": "volume_profile_hvn"
                    })
        
        # Sort by strength (strongest first)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        # Add previous swing lows as support
        swing_lows = self._find_swing_lows(df, current_price)
        for low in swing_lows:
            support_levels.append(low)
        
        # Deduplicate levels that are too close (within 2%)
        deduplicated = []
        for level in support_levels:
            # Check if similar level already exists
            is_duplicate = False
            for existing in deduplicated:
                if abs(level['price'] - existing['price']) / existing['price'] < 0.02:
                    is_duplicate = True
                    # Keep the stronger one
                    if level['strength'] > existing['strength']:
                        deduplicated.remove(existing)
                        deduplicated.append(level)
                    break
            
            if not is_duplicate:
                deduplicated.append(level)
        
        # Re-sort by distance (closest first = first target)
        deduplicated.sort(key=lambda x: x['distance_pct'])
        
        logger.info(f"  Detected {len(deduplicated)} support levels below ${current_price:.4f}")
        
        return deduplicated[:5]  # Return top 5 targets

    def _find_swing_lows(
        self, 
        df: pd.DataFrame, 
        current_price: float, 
        window: int = 5
    ) -> List[Dict]:
        """
        Find significant swing lows that act as support
        
        Args:
            df: OHLCV dataframe
            current_price: Current price
            window: Lookback window for swing detection
            
        Returns:
            List of swing low support levels
        """
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Check if this is a local minimum
            current_low = df['low'].iloc[i]
            
            # Must be lower than surrounding candles
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] < current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low and current_low < current_price * 0.95:
                distance_pct = ((current_price - current_low) / current_price) * 100
                
                # Strength based on how many times price bounced off this level
                touches = self._count_support_touches(df, current_low, tolerance=0.02)
                
                swing_lows.append({
                    "price": round(current_low, 8),
                    "strength": touches * 15,  # Each touch = +15 strength
                    "distance_pct": round(distance_pct, 2),
                    "type": "swing_low",
                    "touches": touches
                })
        
        return swing_lows

    def _count_support_touches(
        self, 
        df: pd.DataFrame, 
        level: float, 
        tolerance: float = 0.02
    ) -> int:
        """Count how many times price touched a support level"""
        touches = 0
        for _, candle in df.iterrows():
            if abs(candle['low'] - level) / level <= tolerance:
                touches += 1
        return touches

    def calculate_profit_targets(
        self, 
        support_levels: List[Dict], 
        entry_price: float, 
        stop_price: float
    ) -> List[Dict]:
        """
        Calculate profit targets based on support levels and R:R ratios
        
        WEEK 4 ENHANCEMENT: R:R aware target selection
        
        Args:
            support_levels: Detected support levels
            entry_price: Trade entry price
            stop_price: Stop loss price
            
        Returns:
            List of profit targets with R:R ratios
        """
        stop_distance = abs(entry_price - stop_price)
        
        targets = []
        for level in support_levels:
            target_price = level['price']
            profit_distance = abs(entry_price - target_price)
            
            # Calculate R:R ratio
            r_ratio = profit_distance / stop_distance if stop_distance > 0 else 0
            
            # Only include targets that are worth at least 1R
            if r_ratio >= 1.0:
                targets.append({
                    "target_price": target_price,
                    "distance_pct": level['distance_pct'],
                    "r_ratio": round(r_ratio, 2),
                    "strength": level['strength'],
                    "type": level['type'],
                    "recommended_partial": self._suggest_partial_size(r_ratio)
                })
        
        # Sort by R:R (take best targets first)
        targets.sort(key=lambda x: x['r_ratio'])
        
        return targets

    def _suggest_partial_size(self, r_ratio: float) -> str:
        """Suggest what % to take off at this target"""
        if r_ratio >= 5:
            return "25% (5R+ runner material)"
        elif r_ratio >= 3:
            return "25% (strong 3R+ target)"
        elif r_ratio >= 2:
            return "33% (decent 2R target)"
        elif r_ratio >= 1:
            return "50% (1R safety partial)"
        else:
            return "SKIP (not worth 1R)"


if __name__ == "__main__":
    """Test signal generator with Week 4 enhancements"""

    print("\n" + "=" * 80)
    print("SIGNAL GENERATOR - TESTING (WEEK 4 ENHANCED)")
    print("=" * 80 + "\n")

    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")

    # Initialize generator (auto-creates client)
    generator = SignalGenerator(min_conviction=40)  # Lower threshold for testing

    # Generate signal
    print("TEST: Generate Signal for BTC with Volume Profile Targets")
    print("-" * 40)

    signal = generator.generate_signal(
        symbol="BTCUSDT", 
        rejection_timeframe="1h", 
        cvd_timeframe="15m",
        support_lookback_days=30
    )

    if signal:
        print(f"\nâœ" SIGNAL GENERATED:")
        print(f"  Symbol: {signal['symbol']}")
        print(f"  Direction: {signal['direction']}")
        print(f"  Conviction: {signal['conviction_score']}/100")
        print(f"  Entry: ${signal['entry_price']:,.4f}")
        print(f"  Stop Loss: ${signal['stop_loss']:,.4f} ({signal['stop_distance_pct']:.2f}%)")
        print(f"\n  Take Profits:")
        for tp_name, tp_data in signal["take_profits"].items():
            if tp_name != 'runner':
                print(f"    {tp_name.upper()}: ${tp_data['price']:,.4f}")
                print(f"      - Size: {tp_data['size']*100:.0f}%")
                print(f"      - R:R: {tp_data['r_ratio']:.2f}R")
                print(f"      - Distance: {tp_data['distance_pct']:.2f}%")
            else:
                print(f"    RUNNER: {tp_data['size']*100:.0f}% to {tp_data['target']}")
        
        print(f"\n  Position Size: ${signal['position_size_usd']:.2f}")
        print(f"  Avg Risk:Reward: {signal['avg_risk_reward_ratio']:.2f}")
        
        # Show support levels detected
        if signal.get('support_levels'):
            print(f"\n  Support Levels Detected:")
            for idx, support in enumerate(signal['support_levels'][:3], 1):
                print(f"    #{idx}: ${support['price']:.4f} ({support['type']})")
                print(f"        Strength: {support['strength']:.1f}, Distance: {support['distance_pct']:.2f}%")
    else:
        print("\nâœ— NO SIGNAL (conviction too low)")
        print("\n" + "=" * 80)
    print("TESTING COMPLETE")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
    print("=" * 80)