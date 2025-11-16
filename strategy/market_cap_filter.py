<<<<<<< HEAD
"""
MARKET CAP FILTER
=================
Research-backed filter from P&D paper 2412.18848

Filters out coins >$60M market cap to identify pump-vulnerable assets.
Lower market cap = less liquidity = better dump fade edge.

Author: Grim (Institutional Standards)
"""

import requests
from typing import Dict, List, Optional
from decimal import Decimal
from loguru import logger
from config.settings import settings


class MarketCapFilter:
    """Filter coins by market cap to identify pump-vulnerable assets"""
    
    def __init__(self, max_market_cap_usd: float = 60_000_000):
        """
        Initialize market cap filter.
        
        Args:
            max_market_cap_usd: Maximum market cap in USD (default: $60M from research)
        """
        self.max_market_cap = Decimal(str(max_market_cap_usd))
        
        # Get API key from settings
        try:
            self.cmc_api_key = settings.COINMARKETCAP_API_KEY
        except AttributeError:
            logger.warning("COINMARKETCAP_API_KEY not found in settings, using placeholder")
            self.cmc_api_key = "YOUR_CMC_API_KEY_HERE"
        
        self.cache: Dict[str, Decimal] = {}
        
        logger.info(f"MarketCapFilter initialized (max_cap=${max_market_cap_usd:,.0f})")
        
    def get_market_cap(self, symbol: str) -> Optional[Decimal]:
        """
        Fetch current market cap from CoinMarketCap API
        
        Args:
            symbol: Trading pair (e.g., "PIGGYUSDT")
            
        Returns:
            Market cap in USD or None if unavailable
        """
        # Strip USDT suffix to get base symbol
        base_symbol = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")
        
        # Check cache first (5 minute TTL)
        if base_symbol in self.cache:
            return self.cache[base_symbol]
        
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            params = {
                "symbol": base_symbol,
                "convert": "USD"
            }
            headers = {
                "X-CMC_PRO_API_KEY": self.cmc_api_key
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if base_symbol in data.get("data", {}):
                market_cap = Decimal(str(data["data"][base_symbol]["quote"]["USD"]["market_cap"]))
                self.cache[base_symbol] = market_cap
                
                logger.debug(f"{symbol} market cap: ${market_cap:,.0f}")
                return market_cap
            else:
                logger.warning(f"Market cap not found for {symbol} (symbol not in CMC response)")
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("CoinMarketCap API authentication failed - check API key")
            else:
                logger.error(f"HTTP error fetching market cap for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch market cap for {symbol}: {str(e)}")
            return None
    
    def is_pump_vulnerable(self, symbol: str) -> tuple[bool, str]:
        """
        Check if coin meets market cap criteria for dump fade strategy
        
        Research shows: Coins <$60M market cap have better dump fade edges
        
        Args:
            symbol: Trading pair to check
            
        Returns:
            (passes_filter, reason)
        """
        market_cap = self.get_market_cap(symbol)
        
        if market_cap is None:
            return False, "Market cap data unavailable"
        
        if market_cap > self.max_market_cap:
            return False, f"Market cap ${market_cap:,.0f} > ${self.max_market_cap:,.0f} threshold"
        
        return True, f"Market cap ${market_cap:,.0f} within target range"
    
    def filter_symbols(self, symbols: List[str]) -> List[Dict[str, any]]:
        """
        Filter list of symbols by market cap criteria
        
        Args:
            symbols: List of trading pairs
            
        Returns:
            List of dicts with symbol, market_cap, passes_filter
        """
        results = []
        
        for symbol in symbols:
            market_cap = self.get_market_cap(symbol)
            passes = market_cap is not None and market_cap <= self.max_market_cap
            
            results.append({
                "symbol": symbol,
                "market_cap": float(market_cap) if market_cap else None,
                "passes_filter": passes,
                "threshold": float(self.max_market_cap)
            })
        
        passed_count = sum(1 for r in results if r['passes_filter'])
        logger.info(f"Market cap filter: {passed_count}/{len(symbols)} symbols passed")
        
        return results
    
    def clear_cache(self):
        """Clear market cap cache (use when data becomes stale)"""
        self.cache.clear()
        logger.debug("Market cap cache cleared")


# Convenience functions
def filter_by_market_cap(symbols: List[str], max_cap: float = 60_000_000) -> List[str]:
    """
    Quick filter - returns only passing symbols
    
    Args:
        symbols: List of trading pairs
        max_cap: Maximum market cap threshold
        
    Returns:
        List of symbols that pass the filter
    """
    filter_instance = MarketCapFilter(max_cap)
    results = filter_instance.filter_symbols(symbols)
    return [r["symbol"] for r in results if r["passes_filter"]]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test market cap filter"""
    
    print("\n" + "="*80)
    print("MARKET CAP FILTER - TESTING")
    print("="*80 + "\n")
    
    # Initialize filter
    filter = MarketCapFilter(max_market_cap_usd=60_000_000)
    
    # Test symbols
    test_symbols = [
        "BTCUSDT",      # Large cap (should fail)
        "ETHUSDT",      # Large cap (should fail)
        "PIGGYUSDT",    # Small cap (should pass)
        "PEPEUSDT",     # Medium cap (depends on current cap)
    ]
    
    print("Testing individual symbols:")
    print("-" * 40)
    
    for symbol in test_symbols:
        passes, reason = filter.is_pump_vulnerable(symbol)
        status = "[PASS]" if passes else "[FAIL]"
        print(f"{symbol:12s} {status}")
        print(f"  {reason}")
        print()
    
    print("\nBatch filtering:")
    print("-" * 40)
    
    results = filter.filter_symbols(test_symbols)
    
    for result in results:
        status = "[PASS]" if result['passes_filter'] else "[FAIL]"
        mcap = result['market_cap']
        mcap_str = f"${mcap:,.0f}" if mcap else "N/A"
        print(f"{status} {result['symbol']:12s} - Market Cap: {mcap_str}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nNOTE: Add COINMARKETCAP_API_KEY to your .env file")
=======
"""
MARKET CAP FILTER
=================
Research-backed filter from P&D paper 2412.18848

Filters out coins >$60M market cap to identify pump-vulnerable assets.
Lower market cap = less liquidity = better dump fade edge.

Author: Grim (Institutional Standards)
"""

import requests
from typing import Dict, List, Optional
from decimal import Decimal
from loguru import logger
from config.settings import settings


class MarketCapFilter:
    """Filter coins by market cap to identify pump-vulnerable assets"""
    
    def __init__(self, max_market_cap_usd: float = 60_000_000):
        """
        Initialize market cap filter.
        
        Args:
            max_market_cap_usd: Maximum market cap in USD (default: $60M from research)
        """
        self.max_market_cap = Decimal(str(max_market_cap_usd))
        
        # Get API key from settings
        try:
            self.cmc_api_key = settings.COINMARKETCAP_API_KEY
        except AttributeError:
            logger.warning("COINMARKETCAP_API_KEY not found in settings, using placeholder")
            self.cmc_api_key = "YOUR_CMC_API_KEY_HERE"
        
        self.cache: Dict[str, Decimal] = {}
        
        logger.info(f"MarketCapFilter initialized (max_cap=${max_market_cap_usd:,.0f})")
        
    def get_market_cap(self, symbol: str) -> Optional[Decimal]:
        """
        Fetch current market cap from CoinMarketCap API
        
        Args:
            symbol: Trading pair (e.g., "PIGGYUSDT")
            
        Returns:
            Market cap in USD or None if unavailable
        """
        # Strip USDT suffix to get base symbol
        base_symbol = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")
        
        # Check cache first (5 minute TTL)
        if base_symbol in self.cache:
            return self.cache[base_symbol]
        
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            params = {
                "symbol": base_symbol,
                "convert": "USD"
            }
            headers = {
                "X-CMC_PRO_API_KEY": self.cmc_api_key
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if base_symbol in data.get("data", {}):
                market_cap = Decimal(str(data["data"][base_symbol]["quote"]["USD"]["market_cap"]))
                self.cache[base_symbol] = market_cap
                
                logger.debug(f"{symbol} market cap: ${market_cap:,.0f}")
                return market_cap
            else:
                logger.warning(f"Market cap not found for {symbol} (symbol not in CMC response)")
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("CoinMarketCap API authentication failed - check API key")
            else:
                logger.error(f"HTTP error fetching market cap for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch market cap for {symbol}: {str(e)}")
            return None
    
    def is_pump_vulnerable(self, symbol: str) -> tuple[bool, str]:
        """
        Check if coin meets market cap criteria for dump fade strategy
        
        Research shows: Coins <$60M market cap have better dump fade edges
        
        Args:
            symbol: Trading pair to check
            
        Returns:
            (passes_filter, reason)
        """
        market_cap = self.get_market_cap(symbol)
        
        if market_cap is None:
            return False, "Market cap data unavailable"
        
        if market_cap > self.max_market_cap:
            return False, f"Market cap ${market_cap:,.0f} > ${self.max_market_cap:,.0f} threshold"
        
        return True, f"Market cap ${market_cap:,.0f} within target range"
    
    def filter_symbols(self, symbols: List[str]) -> List[Dict[str, any]]:
        """
        Filter list of symbols by market cap criteria
        
        Args:
            symbols: List of trading pairs
            
        Returns:
            List of dicts with symbol, market_cap, passes_filter
        """
        results = []
        
        for symbol in symbols:
            market_cap = self.get_market_cap(symbol)
            passes = market_cap is not None and market_cap <= self.max_market_cap
            
            results.append({
                "symbol": symbol,
                "market_cap": float(market_cap) if market_cap else None,
                "passes_filter": passes,
                "threshold": float(self.max_market_cap)
            })
        
        passed_count = sum(1 for r in results if r['passes_filter'])
        logger.info(f"Market cap filter: {passed_count}/{len(symbols)} symbols passed")
        
        return results
    
    def clear_cache(self):
        """Clear market cap cache (use when data becomes stale)"""
        self.cache.clear()
        logger.debug("Market cap cache cleared")


# Convenience functions
def filter_by_market_cap(symbols: List[str], max_cap: float = 60_000_000) -> List[str]:
    """
    Quick filter - returns only passing symbols
    
    Args:
        symbols: List of trading pairs
        max_cap: Maximum market cap threshold
        
    Returns:
        List of symbols that pass the filter
    """
    filter_instance = MarketCapFilter(max_cap)
    results = filter_instance.filter_symbols(symbols)
    return [r["symbol"] for r in results if r["passes_filter"]]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test market cap filter"""
    
    print("\n" + "="*80)
    print("MARKET CAP FILTER - TESTING")
    print("="*80 + "\n")
    
    # Initialize filter
    filter = MarketCapFilter(max_market_cap_usd=60_000_000)
    
    # Test symbols
    test_symbols = [
        "BTCUSDT",      # Large cap (should fail)
        "ETHUSDT",      # Large cap (should fail)
        "PIGGYUSDT",    # Small cap (should pass)
        "PEPEUSDT",     # Medium cap (depends on current cap)
    ]
    
    print("Testing individual symbols:")
    print("-" * 40)
    
    for symbol in test_symbols:
        passes, reason = filter.is_pump_vulnerable(symbol)
        status = "âœ" PASS" if passes else "âœ— FAIL"
        print(f"{symbol:12s} {status}")
        print(f"  {reason}")
        print()
    
    print("\nBatch filtering:")
    print("-" * 40)
    
    results = filter.filter_symbols(test_symbols)
    
    for result in results:
        status = "âœ"" if result['passes_filter'] else "âœ—"
        mcap = result['market_cap']
        mcap_str = f"${mcap:,.0f}" if mcap else "N/A"
        print(f"{status} {result['symbol']:12s} - Market Cap: {mcap_str}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nNOTE: Add COINMARKETCAP_API_KEY to your .env file")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
    print("Get free API key at: https://coinmarketcap.com/api/")