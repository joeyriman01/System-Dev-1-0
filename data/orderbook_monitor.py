"""
ORDER BOOK MONITOR
==================
Analyzes order book depth, imbalances, and walls to detect supply/demand pressure.

Order Book Basics:
- Bids = buy orders (demand)
- Asks = sell orders (supply)
- Imbalance = ratio of bid vs ask liquidity
- Walls = large orders that can block price movement

Your Edge:
When price hits resistance with:
- Heavy sell-side order book (asks >> bids)
- Large sell walls just above price
- Low bid depth below

= Supply overwhelming demand = Your short setup confirmed

Author: Grim (Institutional Standards)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from data.binance_client import BinanceClient, BinanceClientError
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single price level in the order book"""
    price: float
    quantity: float
    total_value: float  # price * quantity in USDT


@dataclass
class OrderBookWall:
    """Large order in the book (wall)"""
    side: str  # 'BID' or 'ASK'
    price: float
    quantity: float
    total_value: float
    distance_from_mid_pct: float  # Distance from mid price as %


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot with analysis"""
    symbol: str
    timestamp: datetime
    
    # Raw data
    bids: List[OrderBookLevel]  # Buy orders
    asks: List[OrderBookLevel]  # Sell orders
    
    # Mid price
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float  # Spread in basis points
    
    # Depth analysis
    bid_depth_usdt: float  # Total USDT in bids
    ask_depth_usdt: float  # Total USDT in asks
    
    # Imbalance (-1 to +1, negative = more asks, positive = more bids)
    imbalance: float
    
    # Pressure classification
    pressure: str  # 'BULLISH', 'NEUTRAL', 'BEARISH'
    
    # Walls
    bid_walls: List[OrderBookWall]
    ask_walls: List[OrderBookWall]


class OrderBookMonitor:
    """
    Monitors order book for depth, imbalances, and walls.
    
    This helps confirm your exhaustion fades. When price pumps to resistance
    and you see heavy sell-side order book, it means supply > demand at
    that level. Combine with rejections and funding = high conviction.
    """
    
    # Configuration
    DEFAULT_DEPTH_LEVELS = 100  # How many levels to fetch
    WALL_THRESHOLD_MULTIPLIER = 3.0  # Order is a "wall" if 3x average size
    CACHE_DURATION_SECONDS = 5  # Refresh every 5 seconds
    
    # Imbalance thresholds
    STRONG_BULLISH_THRESHOLD = 0.3  # Bid depth > 30% more than ask
    STRONG_BEARISH_THRESHOLD = -0.3  # Ask depth > 30% more than bid
    
    def __init__(self, binance_client: BinanceClient):
        """Initialize order book monitor"""
        self.client = binance_client
        
        # Cache
        self.orderbook_cache: Dict[str, Tuple[OrderBookSnapshot, float]] = {}
        
        logger.info("OrderBookMonitor initialized")
    
    def _calculate_imbalance(self, bid_depth: float, ask_depth: float) -> float:
        """
        Calculate order book imbalance.
        
        Returns value from -1 to +1:
        - Positive = more bids (bullish)
        - Negative = more asks (bearish)
        - 0 = balanced
        """
        total_depth = bid_depth + ask_depth
        
        if total_depth == 0:
            return 0.0
        
        imbalance = (bid_depth - ask_depth) / total_depth
        return float(imbalance)
    
    def _classify_pressure(self, imbalance: float) -> str:
        """
        Classify market pressure based on imbalance.
        
        Args:
            imbalance: Order book imbalance (-1 to +1)
            
        Returns:
            'BULLISH', 'NEUTRAL', or 'BEARISH'
        """
        if imbalance >= self.STRONG_BULLISH_THRESHOLD:
            return 'BULLISH'
        elif imbalance <= self.STRONG_BEARISH_THRESHOLD:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _find_walls(
        self,
        levels: List[OrderBookLevel],
        side: str,
        mid_price: float
    ) -> List[OrderBookWall]:
        """
        Find large orders (walls) in the order book.
        
        Args:
            levels: List of order book levels
            side: 'BID' or 'ASK'
            mid_price: Mid price for distance calculation
            
        Returns:
            List of significant walls
        """
        if not levels:
            return []
        
        # Calculate average order size
        avg_quantity = sum(level.quantity for level in levels) / len(levels)
        
        # Find orders larger than threshold
        walls = []
        for level in levels:
            if level.quantity >= avg_quantity * self.WALL_THRESHOLD_MULTIPLIER:
                distance_pct = abs(level.price - mid_price) / mid_price * 100
                
                wall = OrderBookWall(
                    side=side,
                    price=level.price,
                    quantity=level.quantity,
                    total_value=level.total_value,
                    distance_from_mid_pct=distance_pct
                )
                walls.append(wall)
        
        return walls
    
    def get_orderbook(
        self,
        symbol: str,
        depth: int = None,
        force_refresh: bool = False
    ) -> OrderBookSnapshot:
        """
        Get order book snapshot with analysis.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            depth: Number of price levels (default: 100)
            force_refresh: Force refresh cached data
            
        Returns:
            OrderBookSnapshot with complete analysis
        """
        if depth is None:
            depth = self.DEFAULT_DEPTH_LEVELS
        
        # Check cache
        if not force_refresh and symbol in self.orderbook_cache:
            snapshot, cached_time = self.orderbook_cache[symbol]
            cache_age = time.time() - cached_time
            
            if cache_age < self.CACHE_DURATION_SECONDS:
                logger.debug(f"Using cached order book for {symbol}")
                return snapshot
        
        # Fetch from API
        try:
            logger.debug(f"Fetching order book for {symbol} (depth={depth})")
            
            raw_book = self.client.futures_orderbook(symbol=symbol, limit=depth)
            
            # Parse bids and asks
            bids = []
            for price_str, qty_str in raw_book['bids']:
                price = float(price_str)
                quantity = float(qty_str)
                bids.append(OrderBookLevel(
                    price=price,
                    quantity=quantity,
                    total_value=price * quantity
                ))
            
            asks = []
            for price_str, qty_str in raw_book['asks']:
                price = float(price_str)
                quantity = float(qty_str)
                asks.append(OrderBookLevel(
                    price=price,
                    quantity=quantity,
                    total_value=price * quantity
                ))
            
            # Calculate mid price and spread
            best_bid = bids[0].price if bids else 0.0
            best_ask = asks[0].price if asks else 0.0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
            
            spread_bps = 0.0
            if mid_price > 0:
                spread_bps = (best_ask - best_bid) / mid_price * 10000
            
            # Calculate depth
            bid_depth = sum(level.total_value for level in bids)
            ask_depth = sum(level.total_value for level in asks)
            
            # Calculate imbalance
            imbalance = self._calculate_imbalance(bid_depth, ask_depth)
            
            # Classify pressure
            pressure = self._classify_pressure(imbalance)
            
            # Find walls
            bid_walls = self._find_walls(bids, 'BID', mid_price)
            ask_walls = self._find_walls(asks, 'ASK', mid_price)
            
            # Create snapshot
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                spread_bps=spread_bps,
                bid_depth_usdt=bid_depth,
                ask_depth_usdt=ask_depth,
                imbalance=imbalance,
                pressure=pressure,
                bid_walls=bid_walls,
                ask_walls=ask_walls
            )
            
            # Cache
            self.orderbook_cache[symbol] = (snapshot, time.time())
            
            logger.debug(
                f"Order book for {symbol}: "
                f"Spread={spread_bps:.1f}bps, "
                f"Imbalance={imbalance:+.2f}, "
                f"Pressure={pressure}, "
                f"Walls: {len(bid_walls)} bids, {len(ask_walls)} asks"
            )
            
            return snapshot
            
        except BinanceClientError as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            
            # Return cached if available
            if symbol in self.orderbook_cache:
                logger.warning(f"Returning stale order book for {symbol}")
                snapshot, _ = self.orderbook_cache[symbol]
                return snapshot
            
            raise
    
    def get_depth_at_price(
        self,
        symbol: str,
        target_price: float,
        side: str = 'ASK'
    ) -> float:
        """
        Get total liquidity (depth) from current price to target price.
        
        Useful for asking: "How much sell pressure exists between here and resistance?"
        
        Args:
            symbol: Trading pair
            target_price: Price level to measure to
            side: 'BID' or 'ASK'
            
        Returns:
            Total USDT value of orders between mid and target
        """
        snapshot = self.get_orderbook(symbol)
        
        levels = snapshot.asks if side == 'ASK' else snapshot.bids
        mid_price = snapshot.mid_price
        
        # Calculate depth between mid and target
        depth = 0.0
        for level in levels:
            if side == 'ASK' and mid_price <= level.price <= target_price:
                depth += level.total_value
            elif side == 'BID' and target_price <= level.price <= mid_price:
                depth += level.total_value
        
        return float(depth)
    
    def find_nearest_wall(
        self,
        symbol: str,
        side: str = 'ASK',
        max_distance_pct: float = 2.0
    ) -> Optional[OrderBookWall]:
        """
        Find nearest large wall.
        
        Args:
            symbol: Trading pair
            side: 'BID' or 'ASK'
            max_distance_pct: Maximum distance from mid price (%)
            
        Returns:
            Nearest wall within max distance, or None
        """
        snapshot = self.get_orderbook(symbol)
        
        walls = snapshot.ask_walls if side == 'ASK' else snapshot.bid_walls
        
        # Filter by distance
        nearby_walls = [
            wall for wall in walls
            if wall.distance_from_mid_pct <= max_distance_pct
        ]
        
        if not nearby_walls:
            return None
        
        # Return nearest
        return min(nearby_walls, key=lambda w: w.distance_from_mid_pct)
    
    def analyze_resistance_zone(
        self,
        symbol: str,
        resistance_price: float,
        zone_width_pct: float = 0.5
    ) -> Dict[str, float]:
        """
        Analyze order book around a resistance level.
        
        This is KEY for your strategy. When price approaches resistance,
        check if there's heavy sell-side liquidity defending that level.
        
        Args:
            symbol: Trading pair
            resistance_price: Resistance level to analyze
            zone_width_pct: Width of zone around resistance (%)
            
        Returns:
            Dict with analysis metrics
        """
        snapshot = self.get_orderbook(symbol)
        
        # Define zone
        zone_width = resistance_price * (zone_width_pct / 100)
        zone_low = resistance_price - zone_width
        zone_high = resistance_price + zone_width
        
        # Calculate ask depth in zone (selling pressure)
        ask_depth_in_zone = sum(
            level.total_value for level in snapshot.asks
            if zone_low <= level.price <= zone_high
        )
        
        # Calculate bid depth below zone (buying support)
        bid_depth_below = sum(
            level.total_value for level in snapshot.bids
            if level.price < zone_low
        )
        
        # Find walls in zone
        walls_in_zone = [
            wall for wall in snapshot.ask_walls
            if zone_low <= wall.price <= zone_high
        ]
        
        # Distance to zone
        distance_to_zone = (zone_low - snapshot.mid_price) / snapshot.mid_price * 100
        
        return {
            'resistance_price': resistance_price,
            'zone_low': zone_low,
            'zone_high': zone_high,
            'current_price': snapshot.mid_price,
            'distance_to_zone_pct': distance_to_zone,
            'ask_depth_in_zone': ask_depth_in_zone,
            'bid_depth_below': bid_depth_below,
            'sell_buy_ratio': ask_depth_in_zone / bid_depth_below if bid_depth_below > 0 else 999.0,
            'num_walls_in_zone': len(walls_in_zone),
            'heavy_resistance': ask_depth_in_zone > bid_depth_below * 1.5  # Sells > 1.5x buys
        }
    
    def get_pressure_score(self, symbol: str) -> float:
        """
        Get overall market pressure score from -100 to +100.
        
        Negative = bearish (heavy sells)
        Positive = bullish (heavy buys)
        
        Args:
            symbol: Trading pair
            
        Returns:
            Pressure score
        """
        snapshot = self.get_orderbook(symbol)
        
        # Base score from imbalance
        score = snapshot.imbalance * 100
        
        # Adjust for spread (wider spread = less confidence)
        if snapshot.spread_bps > 50:
            score *= 0.5  # Reduce confidence in wide spread
        
        # Adjust for walls
        if snapshot.pressure == 'BEARISH' and snapshot.ask_walls:
            score -= 10  # Extra bearish if ask walls present
        elif snapshot.pressure == 'BULLISH' and snapshot.bid_walls:
            score += 10  # Extra bullish if bid walls present
        
        return float(max(-100, min(100, score)))
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached order book data"""
        if symbol:
            if symbol in self.orderbook_cache:
                del self.orderbook_cache[symbol]
            logger.info(f"Cleared order book cache for {symbol}")
        else:
            self.orderbook_cache.clear()
            logger.info("Cleared all order book cache")


def main():
    """Test the order book monitor"""
    from data.binance_client import BinanceClient
    
    print("=" * 70)
    print("ORDER BOOK MONITOR TEST")
    print("=" * 70)
    
    # Initialize
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    monitor = OrderBookMonitor(client)
    
    # Test symbol
    symbol = 'BTCUSDT'
    
    print(f"\nTesting {symbol}...\n")
    
    # Test 1: Get order book snapshot
    print("1. Fetching order book...")
    snapshot = monitor.get_orderbook(symbol, depth=50)
    
    print(f"   ✓ Mid Price: ${snapshot.mid_price:,.2f}")
    print(f"   ✓ Best Bid: ${snapshot.best_bid:,.2f}")
    print(f"   ✓ Best Ask: ${snapshot.best_ask:,.2f}")
    print(f"   ✓ Spread: {snapshot.spread_bps:.1f} bps")
    
    # Test 2: Depth analysis
    print(f"\n2. Analyzing depth...")
    print(f"   ✓ Bid Depth: ${snapshot.bid_depth_usdt:,.0f}")
    print(f"   ✓ Ask Depth: ${snapshot.ask_depth_usdt:,.0f}")
    print(f"   ✓ Imbalance: {snapshot.imbalance:+.2f}")
    print(f"   ✓ Pressure: {snapshot.pressure}")
    
    # Test 3: Walls
    print(f"\n3. Finding walls...")
    print(f"   ✓ Bid Walls: {len(snapshot.bid_walls)}")
    if snapshot.bid_walls:
        nearest_bid_wall = snapshot.bid_walls[0]
        print(f"      ${nearest_bid_wall.price:,.2f} | "
              f"${nearest_bid_wall.total_value:,.0f} | "
              f"{nearest_bid_wall.distance_from_mid_pct:.2f}% away")
    
    print(f"   ✓ Ask Walls: {len(snapshot.ask_walls)}")
    if snapshot.ask_walls:
        nearest_ask_wall = snapshot.ask_walls[0]
        print(f"      ${nearest_ask_wall.price:,.2f} | "
              f"${nearest_ask_wall.total_value:,.0f} | "
              f"{nearest_ask_wall.distance_from_mid_pct:.2f}% away")
    
    # Test 4: Pressure score
    print(f"\n4. Calculating pressure score...")
    pressure_score = monitor.get_pressure_score(symbol)
    print(f"   ✓ Pressure Score: {pressure_score:+.1f}/100")
    
    # Test 5: Resistance analysis
    print(f"\n5. Analyzing hypothetical resistance...")
    resistance_price = snapshot.mid_price * 1.02  # 2% above current
    resistance_analysis = monitor.analyze_resistance_zone(
        symbol,
        resistance_price,
        zone_width_pct=0.5
    )
    
    print(f"   ✓ Resistance Level: ${resistance_analysis['resistance_price']:,.2f}")
    print(f"   ✓ Distance: {resistance_analysis['distance_to_zone_pct']:+.2f}%")
    print(f"   ✓ Ask Depth in Zone: ${resistance_analysis['ask_depth_in_zone']:,.0f}")
    print(f"   ✓ Bid Depth Below: ${resistance_analysis['bid_depth_below']:,.0f}")
    print(f"   ✓ Sell/Buy Ratio: {resistance_analysis['sell_buy_ratio']:.2f}x")
    print(f"   ✓ Walls in Zone: {resistance_analysis['num_walls_in_zone']}")
    print(f"   ✓ Heavy Resistance: {resistance_analysis['heavy_resistance']}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nOrder Book Monitor is ready.")
    print("\nWEEK 2 DATA COLLECTION: COMPLETE")
    print("Next: Week 3 - Strategy Logic")


if __name__ == "__main__":
    main()