<<<<<<< HEAD
"""
NEW LISTING SCANNER MODULE
===========================
Detects new Binance perpetual futures listings and filters for tradable contracts.

This is the foundation of your edge - you trade newly listed meme coins during pump cycles.
If this module misses listings or gives false signals, everything downstream fails.

Key Requirements:
- Detect new listings within minutes (not hours)
- Filter out illiquid garbage (min volume, liquidity thresholds)
- Prioritize by tradable volume and volatility
- Handle API rate limits gracefully
- Cache to avoid redundant calls
- Log everything for debugging

Author: Grim (Institutional Standards)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from config.settings import settings
from data.binance_client import BinanceClient

logger = logging.getLogger(__name__)


@dataclass
class NewListing:
    """Represents a newly listed perpetual futures contract"""

    symbol: str
    listed_time: datetime
    current_price: float
    volume_24h: float  # USDT volume
    price_change_24h: float  # Percentage
    volatility_1h: float  # Price range percentage
    mark_price: float
    funding_rate: float
    open_interest: float  # USDT value
    tradable: bool
    priority_score: float  # 0-100, higher = better setup


class ListingScanner:
    """
    Scans for new perpetual futures listings on Binance.

    This is NOT a fire-and-forget module. It maintains state, caches listings,
    and intelligently filters for tradable contracts.
    """

    # Minimum thresholds for tradable contracts
    MIN_VOLUME_24H = 100000  # $100k USDT (adjust based on testing)
    MIN_OPEN_INTEREST = 50000  # $50k USDT
    MIN_PRICE_CHANGE = 10.0  # 10% minimum pump
    MAX_SPREAD_BPS = 50  # Max 50 bps spread (0.5%)

    # Rate limiting
    CACHE_DURATION_SECONDS = 60  # Cache exchange info for 60s
    MAX_REQUESTS_PER_MINUTE = 50  # Conservative limit

    def __init__(self, binance_client: BinanceClient):
        """Initialize scanner with Binance client"""
        self.client = binance_client
        self.known_symbols: Set[str] = set()  # Track all known symbols
        self.new_listings: List[NewListing] = []  # Store detected listings
        self.last_scan_time: Optional[datetime] = None
        self.cache: Dict[str, Any] = {}  # Simple cache for exchange info
        self.request_count: int = 0  # Rate limit tracking
        self.request_window_start: float = time.time()

        logger.info("ListingScanner initialized")

    def _check_rate_limit(self):
        """Enforce rate limiting to avoid API bans"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.request_window_start > 60:
            self.request_count = 0
            self.request_window_start = current_time

        if self.request_count >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()

        self.request_count += 1

    def get_all_perpetuals(self) -> List[str]:
        """
        Fetch all perpetual futures symbols from Binance.

        Uses caching to avoid hammering the API.

        Returns:
            List of symbol strings (e.g. ['BTCUSDT', 'ETHUSDT', 'PEPEUSDT'])
        """
        cache_key = "all_perpetuals"
        cache_time = self.cache.get(f"{cache_key}_time", 0)

        # Return cached data if fresh
        if time.time() - cache_time < self.CACHE_DURATION_SECONDS:
            logger.debug("Returning cached perpetual symbols")
            return self.cache[cache_key]

        try:
            self._check_rate_limit()
            exchange_info = self.client.futures_exchange_info()

            # Filter for USDT perpetuals only
            symbols = [
                s["symbol"]
                for s in exchange_info["symbols"]
                if s["symbol"].endswith("USDT")
                and s["contractType"] == "PERPETUAL"
                and s["status"] == "TRADING"
            ]

            # Update cache
            self.cache[cache_key] = symbols
            self.cache[f"{cache_key}_time"] = time.time()

            logger.info(f"Fetched {len(symbols)} perpetual contracts")
            return symbols

        except Exception as e:
            logger.error(f"Failed to fetch perpetual symbols: {e}")
            # Return cached data if available, even if stale
            return self.cache.get(cache_key, [])

    def detect_new_listings(self, lookback_hours: int = 24) -> List[str]:
        """
        Detect newly listed perpetuals by comparing current symbols to known symbols.

        Args:
            lookback_hours: How far back to look for "new" (for initial scan)

        Returns:
            List of newly detected symbol strings
        """
        current_symbols = set(self.get_all_perpetuals())

        # First run - initialize known symbols
        if not self.known_symbols:
            logger.info(
                f"Initializing scanner with {len(current_symbols)} known symbols"
            )
            self.known_symbols = current_symbols
            return []  # No new listings on first run

        # Detect new symbols
        new_symbols = current_symbols - self.known_symbols

        if new_symbols:
            logger.info(f"Detected {len(new_symbols)} new listings: {new_symbols}")
            self.known_symbols.update(new_symbols)

        return list(new_symbols)

    def get_listing_details(self, symbol: str) -> Optional[Dict]:
        """
        Fetch detailed market data for a symbol.

        This is where we gather the data to filter tradable vs garbage listings.

        Args:
            symbol: Contract symbol (e.g. 'PEPEUSDT')

        Returns:
            Dict with market data or None if fetch fails
        """
        try:
            self._check_rate_limit()

            # Fetch 24h ticker stats
            ticker = self.client.futures_ticker_24hr(symbol=symbol)

            # Fetch current mark price and funding
            mark_price_data = self.client.futures_mark_price(symbol=symbol)

            # Fetch order book for spread calculation
            orderbook = self.client.futures_orderbook(symbol=symbol, limit=5)

            # Calculate spread
            best_bid = float(orderbook["bids"][0][0]) if orderbook["bids"] else 0
            best_ask = float(orderbook["asks"][0][0]) if orderbook["asks"] else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread_bps = (
                ((best_ask - best_bid) / mid_price * 10000) if mid_price > 0 else 999
            )

            # Calculate 1h volatility (high-low range)
            price_high = float(ticker["highPrice"])
            price_low = float(ticker["lowPrice"])
            volatility_1h = (
                ((price_high - price_low) / price_low * 100) if price_low > 0 else 0
            )

            return {
                "symbol": symbol,
                "current_price": float(ticker["lastPrice"]),
                "volume_24h": float(ticker["quoteVolume"]),  # USDT volume
                "price_change_24h": float(ticker["priceChangePercent"]),
                "volatility_1h": volatility_1h,
                "mark_price": float(mark_price_data["markPrice"]),
                "funding_rate": float(mark_price_data["lastFundingRate"])
                * 100,  # Convert to percentage
                "spread_bps": spread_bps,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Failed to fetch details for {symbol}: {e}")
            return None

    def is_tradable(self, details: Dict) -> tuple[bool, str]:
        """
        Determine if a listing meets minimum criteria for trading.

        This is CRITICAL. Most new listings are illiquid garbage that will:
        - Slip your stops by 10%+
        - Have no order book depth
        - Get delisted after a week

        Args:
            details: Market data dict from get_listing_details()

        Returns:
            (is_tradable: bool, reason: str)
        """
        symbol = details["symbol"]

        # Check volume
        if details["volume_24h"] < self.MIN_VOLUME_24H:
            return (
                False,
                f"Volume too low: ${details['volume_24h']:,.0f} < ${self.MIN_VOLUME_24H:,.0f}",
            )

        # Check price change (must be pumping)
        if details["price_change_24h"] < self.MIN_PRICE_CHANGE:
            return (
                False,
                f"No pump detected: {details['price_change_24h']:.1f}% < {self.MIN_PRICE_CHANGE}%",
            )

        # Check spread (too wide = illiquid)
        if details["spread_bps"] > self.MAX_SPREAD_BPS:
            return (
                False,
                f"Spread too wide: {details['spread_bps']:.1f} bps > {self.MAX_SPREAD_BPS} bps",
            )

        # Passed all checks
        return True, "TRADABLE"

    def calculate_priority_score(self, details: Dict) -> float:
        """
        Score a listing from 0-100 based on setup quality.

        Higher score = better setup for your exhaustion fade strategy.

        Factors:
        - Volume (more = better liquidity)
        - Price change (bigger pump = more exhaustion potential)
        - Volatility (higher = bigger profit potential)
        - Spread (tighter = better execution)
        - Funding rate (positive = overleveraged longs)

        Returns:
            Score from 0-100
        """
        score = 0.0

        # Volume score (0-30 points)
        # Scale: $100k = 0 points, $1M = 15 points, $10M+ = 30 points
        volume_usd = details["volume_24h"]
        if volume_usd >= 10_000_000:
            score += 30
        elif volume_usd >= 1_000_000:
            score += 15 + (volume_usd - 1_000_000) / 9_000_000 * 15
        else:
            score += (volume_usd - 100_000) / 900_000 * 15

        # Price change score (0-30 points)
        # Scale: 10% = 0 points, 50% = 15 points, 100%+ = 30 points
        price_change = details["price_change_24h"]
        if price_change >= 100:
            score += 30
        elif price_change >= 50:
            score += 15 + (price_change - 50) / 50 * 15
        else:
            score += (price_change - 10) / 40 * 15

        # Volatility score (0-20 points)
        # Higher volatility = more profit potential (but also more risk)
        volatility = details["volatility_1h"]
        if volatility >= 20:
            score += 20
        else:
            score += (volatility / 20) * 20

        # Spread score (0-10 points)
        # Tighter spread = better execution
        spread = details["spread_bps"]
        if spread <= 10:
            score += 10
        elif spread <= 30:
            score += 10 - (spread - 10) / 20 * 5
        else:
            score += 5 - (spread - 30) / 20 * 5

        # Funding rate score (0-10 points)
        # Positive funding = longs paying shorts = overleveraged longs = good for shorts
        funding = details["funding_rate"]
        if funding >= 0.1:  # 0.1% or higher
            score += 10
        elif funding >= 0:
            score += funding / 0.1 * 10
        # Negative funding = longs getting paid = bad for shorts

        return max(0, min(100, score))  # Clamp to 0-100

    def scan_for_new_listings(self) -> List[NewListing]:
        """
        Main scanning function - detect new listings and analyze them.

        This is what you'll call in your main loop.

        Returns:
            List of NewListing objects, sorted by priority score (highest first)
        """
        logger.info("Starting new listing scan...")
        scan_start = time.time()

        # Detect new symbols
        new_symbols = self.detect_new_listings()

        if not new_symbols:
            logger.info("No new listings detected")
            self.last_scan_time = datetime.now()
            return []

        # Analyze each new listing
        new_listings = []
        for symbol in new_symbols:
            logger.info(f"Analyzing new listing: {symbol}")

            details = self.get_listing_details(symbol)
            if not details:
                logger.warning(f"Failed to fetch details for {symbol}, skipping")
                continue

            # Check if tradable
            tradable, reason = self.is_tradable(details)
            logger.info(f"{symbol}: {reason}")

            # Calculate priority score
            priority_score = self.calculate_priority_score(details)

            # Create NewListing object
            listing = NewListing(
                symbol=symbol,
                listed_time=details["timestamp"],
                current_price=details["current_price"],
                volume_24h=details["volume_24h"],
                price_change_24h=details["price_change_24h"],
                volatility_1h=details["volatility_1h"],
                mark_price=details["mark_price"],
                funding_rate=details["funding_rate"],
                open_interest=0,  # We'll add this later
                tradable=tradable,
                priority_score=priority_score,
            )

            new_listings.append(listing)

            logger.info(
                f"{symbol}: Priority={priority_score:.1f}, "
                f"Volume=${details['volume_24h']:,.0f}, "
                f"Change={details['price_change_24h']:.1f}%, "
                f"Tradable={tradable}"
            )

        # Sort by priority score (highest first)
        new_listings.sort(key=lambda x: x.priority_score, reverse=True)

        self.new_listings = new_listings
        self.last_scan_time = datetime.now()

        scan_duration = time.time() - scan_start
        logger.info(
            f"Scan complete: {len(new_listings)} new listings analyzed in {scan_duration:.2f}s"
        )

        return new_listings

    def get_top_opportunities(
        self, min_score: float = 50.0, max_results: int = 5
    ) -> List[NewListing]:
        """
        Get top-ranked tradable listings above minimum score threshold.

        This is what feeds into your strategy modules.

        Args:
            min_score: Minimum priority score (0-100)
            max_results: Maximum number of results to return

        Returns:
            List of top NewListing objects
        """
        tradable_listings = [
            listing
            for listing in self.new_listings
            if listing.tradable and listing.priority_score >= min_score
        ]

        return tradable_listings[:max_results]


def main():
    """Test the scanner"""
    from data.binance_client import BinanceClient

    # Initialize
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    scanner = ListingScanner(client)

    # Run scan
    print("Running new listing scan...")
    listings = scanner.scan_for_new_listings()

    print(f"\nFound {len(listings)} new listings:")
    for listing in listings:
        print(f"\n{listing.symbol}:")
        print(f"  Priority Score: {listing.priority_score:.1f}/100")
        print(f"  Volume 24h: ${listing.volume_24h:,.0f}")
        print(f"  Price Change: {listing.price_change_24h:.1f}%")
        print(f"  Volatility: {listing.volatility_1h:.1f}%")
        print(f"  Funding Rate: {listing.funding_rate:.3f}%")
        print(f"  Tradable: {listing.tradable}")

    # Get top opportunities
    top = scanner.get_top_opportunities(min_score=60.0, max_results=3)
    print(f"\n\nTop {len(top)} opportunities (score >= 60):")
    for listing in top:
        print(f"  {listing.symbol}: {listing.priority_score:.1f} points")


if __name__ == "__main__":
    main()
=======
"""
NEW LISTING SCANNER MODULE
===========================
Detects new Binance perpetual futures listings and filters for tradable contracts.

This is the foundation of your edge - you trade newly listed meme coins during pump cycles.
If this module misses listings or gives false signals, everything downstream fails.

Key Requirements:
- Detect new listings within minutes (not hours)
- Filter out illiquid garbage (min volume, liquidity thresholds)
- Prioritize by tradable volume and volatility
- Handle API rate limits gracefully
- Cache to avoid redundant calls
- Log everything for debugging

Author: Grim (Institutional Standards)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from config.settings import settings
from data.binance_client import BinanceClient

logger = logging.getLogger(__name__)


@dataclass
class NewListing:
    """Represents a newly listed perpetual futures contract"""

    symbol: str
    listed_time: datetime
    current_price: float
    volume_24h: float  # USDT volume
    price_change_24h: float  # Percentage
    volatility_1h: float  # Price range percentage
    mark_price: float
    funding_rate: float
    open_interest: float  # USDT value
    tradable: bool
    priority_score: float  # 0-100, higher = better setup


class ListingScanner:
    """
    Scans for new perpetual futures listings on Binance.

    This is NOT a fire-and-forget module. It maintains state, caches listings,
    and intelligently filters for tradable contracts.
    """

    # Minimum thresholds for tradable contracts
    MIN_VOLUME_24H = 100000  # $100k USDT (adjust based on testing)
    MIN_OPEN_INTEREST = 50000  # $50k USDT
    MIN_PRICE_CHANGE = 10.0  # 10% minimum pump
    MAX_SPREAD_BPS = 50  # Max 50 bps spread (0.5%)

    # Rate limiting
    CACHE_DURATION_SECONDS = 60  # Cache exchange info for 60s
    MAX_REQUESTS_PER_MINUTE = 50  # Conservative limit

    def __init__(self, binance_client: BinanceClient):
        """Initialize scanner with Binance client"""
        self.client = binance_client
        self.known_symbols: Set[str] = set()  # Track all known symbols
        self.new_listings: List[NewListing] = []  # Store detected listings
        self.last_scan_time: Optional[datetime] = None
        self.cache: Dict[str, Any] = {}  # Simple cache for exchange info
        self.request_count: int = 0  # Rate limit tracking
        self.request_window_start: float = time.time()

        logger.info("ListingScanner initialized")

    def _check_rate_limit(self):
        """Enforce rate limiting to avoid API bans"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.request_window_start > 60:
            self.request_count = 0
            self.request_window_start = current_time

        if self.request_count >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()

        self.request_count += 1

    def get_all_perpetuals(self) -> List[str]:
        """
        Fetch all perpetual futures symbols from Binance.

        Uses caching to avoid hammering the API.

        Returns:
            List of symbol strings (e.g. ['BTCUSDT', 'ETHUSDT', 'PEPEUSDT'])
        """
        cache_key = "all_perpetuals"
        cache_time = self.cache.get(f"{cache_key}_time", 0)

        # Return cached data if fresh
        if time.time() - cache_time < self.CACHE_DURATION_SECONDS:
            logger.debug("Returning cached perpetual symbols")
            return self.cache[cache_key]

        try:
            self._check_rate_limit()
            exchange_info = self.client.futures_exchange_info()

            # Filter for USDT perpetuals only
            symbols = [
                s["symbol"]
                for s in exchange_info["symbols"]
                if s["symbol"].endswith("USDT")
                and s["contractType"] == "PERPETUAL"
                and s["status"] == "TRADING"
            ]

            # Update cache
            self.cache[cache_key] = symbols
            self.cache[f"{cache_key}_time"] = time.time()

            logger.info(f"Fetched {len(symbols)} perpetual contracts")
            return symbols

        except Exception as e:
            logger.error(f"Failed to fetch perpetual symbols: {e}")
            # Return cached data if available, even if stale
            return self.cache.get(cache_key, [])

    def detect_new_listings(self, lookback_hours: int = 24) -> List[str]:
        """
        Detect newly listed perpetuals by comparing current symbols to known symbols.

        Args:
            lookback_hours: How far back to look for "new" (for initial scan)

        Returns:
            List of newly detected symbol strings
        """
        current_symbols = set(self.get_all_perpetuals())

        # First run - initialize known symbols
        if not self.known_symbols:
            logger.info(
                f"Initializing scanner with {len(current_symbols)} known symbols"
            )
            self.known_symbols = current_symbols
            return []  # No new listings on first run

        # Detect new symbols
        new_symbols = current_symbols - self.known_symbols

        if new_symbols:
            logger.info(f"Detected {len(new_symbols)} new listings: {new_symbols}")
            self.known_symbols.update(new_symbols)

        return list(new_symbols)

    def get_listing_details(self, symbol: str) -> Optional[Dict]:
        """
        Fetch detailed market data for a symbol.

        This is where we gather the data to filter tradable vs garbage listings.

        Args:
            symbol: Contract symbol (e.g. 'PEPEUSDT')

        Returns:
            Dict with market data or None if fetch fails
        """
        try:
            self._check_rate_limit()

            # Fetch 24h ticker stats
            ticker = self.client.futures_ticker_24hr(symbol=symbol)

            # Fetch current mark price and funding
            mark_price_data = self.client.futures_mark_price(symbol=symbol)

            # Fetch order book for spread calculation
            orderbook = self.client.futures_orderbook(symbol=symbol, limit=5)

            # Calculate spread
            best_bid = float(orderbook["bids"][0][0]) if orderbook["bids"] else 0
            best_ask = float(orderbook["asks"][0][0]) if orderbook["asks"] else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread_bps = (
                ((best_ask - best_bid) / mid_price * 10000) if mid_price > 0 else 999
            )

            # Calculate 1h volatility (high-low range)
            price_high = float(ticker["highPrice"])
            price_low = float(ticker["lowPrice"])
            volatility_1h = (
                ((price_high - price_low) / price_low * 100) if price_low > 0 else 0
            )

            return {
                "symbol": symbol,
                "current_price": float(ticker["lastPrice"]),
                "volume_24h": float(ticker["quoteVolume"]),  # USDT volume
                "price_change_24h": float(ticker["priceChangePercent"]),
                "volatility_1h": volatility_1h,
                "mark_price": float(mark_price_data["markPrice"]),
                "funding_rate": float(mark_price_data["lastFundingRate"])
                * 100,  # Convert to percentage
                "spread_bps": spread_bps,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Failed to fetch details for {symbol}: {e}")
            return None

    def is_tradable(self, details: Dict) -> tuple[bool, str]:
        """
        Determine if a listing meets minimum criteria for trading.

        This is CRITICAL. Most new listings are illiquid garbage that will:
        - Slip your stops by 10%+
        - Have no order book depth
        - Get delisted after a week

        Args:
            details: Market data dict from get_listing_details()

        Returns:
            (is_tradable: bool, reason: str)
        """
        symbol = details["symbol"]

        # Check volume
        if details["volume_24h"] < self.MIN_VOLUME_24H:
            return (
                False,
                f"Volume too low: ${details['volume_24h']:,.0f} < ${self.MIN_VOLUME_24H:,.0f}",
            )

        # Check price change (must be pumping)
        if details["price_change_24h"] < self.MIN_PRICE_CHANGE:
            return (
                False,
                f"No pump detected: {details['price_change_24h']:.1f}% < {self.MIN_PRICE_CHANGE}%",
            )

        # Check spread (too wide = illiquid)
        if details["spread_bps"] > self.MAX_SPREAD_BPS:
            return (
                False,
                f"Spread too wide: {details['spread_bps']:.1f} bps > {self.MAX_SPREAD_BPS} bps",
            )

        # Passed all checks
        return True, "TRADABLE"

    def calculate_priority_score(self, details: Dict) -> float:
        """
        Score a listing from 0-100 based on setup quality.

        Higher score = better setup for your exhaustion fade strategy.

        Factors:
        - Volume (more = better liquidity)
        - Price change (bigger pump = more exhaustion potential)
        - Volatility (higher = bigger profit potential)
        - Spread (tighter = better execution)
        - Funding rate (positive = overleveraged longs)

        Returns:
            Score from 0-100
        """
        score = 0.0

        # Volume score (0-30 points)
        # Scale: $100k = 0 points, $1M = 15 points, $10M+ = 30 points
        volume_usd = details["volume_24h"]
        if volume_usd >= 10_000_000:
            score += 30
        elif volume_usd >= 1_000_000:
            score += 15 + (volume_usd - 1_000_000) / 9_000_000 * 15
        else:
            score += (volume_usd - 100_000) / 900_000 * 15

        # Price change score (0-30 points)
        # Scale: 10% = 0 points, 50% = 15 points, 100%+ = 30 points
        price_change = details["price_change_24h"]
        if price_change >= 100:
            score += 30
        elif price_change >= 50:
            score += 15 + (price_change - 50) / 50 * 15
        else:
            score += (price_change - 10) / 40 * 15

        # Volatility score (0-20 points)
        # Higher volatility = more profit potential (but also more risk)
        volatility = details["volatility_1h"]
        if volatility >= 20:
            score += 20
        else:
            score += (volatility / 20) * 20

        # Spread score (0-10 points)
        # Tighter spread = better execution
        spread = details["spread_bps"]
        if spread <= 10:
            score += 10
        elif spread <= 30:
            score += 10 - (spread - 10) / 20 * 5
        else:
            score += 5 - (spread - 30) / 20 * 5

        # Funding rate score (0-10 points)
        # Positive funding = longs paying shorts = overleveraged longs = good for shorts
        funding = details["funding_rate"]
        if funding >= 0.1:  # 0.1% or higher
            score += 10
        elif funding >= 0:
            score += funding / 0.1 * 10
        # Negative funding = longs getting paid = bad for shorts

        return max(0, min(100, score))  # Clamp to 0-100

    def scan_for_new_listings(self) -> List[NewListing]:
        """
        Main scanning function - detect new listings and analyze them.

        This is what you'll call in your main loop.

        Returns:
            List of NewListing objects, sorted by priority score (highest first)
        """
        logger.info("Starting new listing scan...")
        scan_start = time.time()

        # Detect new symbols
        new_symbols = self.detect_new_listings()

        if not new_symbols:
            logger.info("No new listings detected")
            self.last_scan_time = datetime.now()
            return []

        # Analyze each new listing
        new_listings = []
        for symbol in new_symbols:
            logger.info(f"Analyzing new listing: {symbol}")

            details = self.get_listing_details(symbol)
            if not details:
                logger.warning(f"Failed to fetch details for {symbol}, skipping")
                continue

            # Check if tradable
            tradable, reason = self.is_tradable(details)
            logger.info(f"{symbol}: {reason}")

            # Calculate priority score
            priority_score = self.calculate_priority_score(details)

            # Create NewListing object
            listing = NewListing(
                symbol=symbol,
                listed_time=details["timestamp"],
                current_price=details["current_price"],
                volume_24h=details["volume_24h"],
                price_change_24h=details["price_change_24h"],
                volatility_1h=details["volatility_1h"],
                mark_price=details["mark_price"],
                funding_rate=details["funding_rate"],
                open_interest=0,  # We'll add this later
                tradable=tradable,
                priority_score=priority_score,
            )

            new_listings.append(listing)

            logger.info(
                f"{symbol}: Priority={priority_score:.1f}, "
                f"Volume=${details['volume_24h']:,.0f}, "
                f"Change={details['price_change_24h']:.1f}%, "
                f"Tradable={tradable}"
            )

        # Sort by priority score (highest first)
        new_listings.sort(key=lambda x: x.priority_score, reverse=True)

        self.new_listings = new_listings
        self.last_scan_time = datetime.now()

        scan_duration = time.time() - scan_start
        logger.info(
            f"Scan complete: {len(new_listings)} new listings analyzed in {scan_duration:.2f}s"
        )

        return new_listings

    def get_top_opportunities(
        self, min_score: float = 50.0, max_results: int = 5
    ) -> List[NewListing]:
        """
        Get top-ranked tradable listings above minimum score threshold.

        This is what feeds into your strategy modules.

        Args:
            min_score: Minimum priority score (0-100)
            max_results: Maximum number of results to return

        Returns:
            List of top NewListing objects
        """
        tradable_listings = [
            listing
            for listing in self.new_listings
            if listing.tradable and listing.priority_score >= min_score
        ]

        return tradable_listings[:max_results]


def main():
    """Test the scanner"""
    from data.binance_client import BinanceClient

    # Initialize
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    scanner = ListingScanner(client)

    # Run scan
    print("Running new listing scan...")
    listings = scanner.scan_for_new_listings()

    print(f"\nFound {len(listings)} new listings:")
    for listing in listings:
        print(f"\n{listing.symbol}:")
        print(f"  Priority Score: {listing.priority_score:.1f}/100")
        print(f"  Volume 24h: ${listing.volume_24h:,.0f}")
        print(f"  Price Change: {listing.price_change_24h:.1f}%")
        print(f"  Volatility: {listing.volatility_1h:.1f}%")
        print(f"  Funding Rate: {listing.funding_rate:.3f}%")
        print(f"  Tradable: {listing.tradable}")

    # Get top opportunities
    top = scanner.get_top_opportunities(min_score=60.0, max_results=3)
    print(f"\n\nTop {len(top)} opportunities (score >= 60):")
    for listing in top:
        print(f"  {listing.symbol}: {listing.priority_score:.1f} points")


if __name__ == "__main__":
    main()
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
