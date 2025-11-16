<<<<<<< HEAD
"""
BINANCE API CLIENT - COMPLETE FUTURES IMPLEMENTATION
====================================================
Wraps Binance futures API with proper error handling, rate limiting, and logging.

This is the CORRECT implementation with all methods needed for NullSpectre v2.0.

Author: Grim (Fixing Week 1's incomplete implementation)
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings

logger = logging.getLogger(__name__)


class BinanceClientError(Exception):
    """Custom exception for Binance API errors"""

    pass


class BinanceClient:
    """
    Binance Futures API client with complete method coverage.

    Handles:
    - Testnet and mainnet endpoints
    - Request signing for authenticated endpoints
    - Rate limiting
    - Error handling and retries
    - Response validation
    """

    # API Endpoints
    MAINNET_BASE_URL = "https://fapi.binance.com"
    TESTNET_BASE_URL = "https://testnet.binancefuture.com"

    # Rate limits (requests per minute)
    RATE_LIMIT_WEIGHT = 1200  # Default weight limit per minute

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ):
        """
        Initialize Binance Futures client.

        Args:
            api_key: API key (optional, loads from settings if not provided)
            api_secret: API secret (optional, loads from settings if not provided)
            testnet: Use testnet (True) or mainnet (False)
        """
        self.api_key = api_key or settings.BINANCE_API_KEY
        self.api_secret = api_secret or settings.BINANCE_API_SECRET
        self.testnet = testnet

        # Set base URL
        self.base_url = self.TESTNET_BASE_URL if testnet else self.MAINNET_BASE_URL

        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Rate limiting
        self.last_request_time = 0
        self.request_weight = 0
        self.weight_reset_time = time.time() + 60

        logger.info(f"BinanceClient initialized (testnet={testnet})")

    def _get_headers(self, signed: bool = False) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json",
        }
        if signed and self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.api_secret:
            raise BinanceClientError("API secret required for signed requests")

        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def _enforce_rate_limit(self, weight: int = 1):
        """Enforce rate limiting to avoid API bans"""
        current_time = time.time()

        # Reset weight counter every minute
        if current_time > self.weight_reset_time:
            self.request_weight = 0
            self.weight_reset_time = current_time + 60

        # Check if we're over the limit
        if self.request_weight + weight > self.RATE_LIMIT_WEIGHT:
            sleep_time = self.weight_reset_time - current_time
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_weight = 0
                self.weight_reset_time = time.time() + 60

        self.request_weight += weight

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Dict[str, Any]:
        """
        Make API request with proper error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/fapi/v1/ticker/24hr")
            params: Request parameters
            signed: Whether to sign the request
            weight: Request weight for rate limiting

        Returns:
            API response as dict

        Raises:
            BinanceClientError: On API errors
        """
        if params is None:
            params = {}

        # Enforce rate limiting
        self._enforce_rate_limit(weight)

        # Add timestamp for signed requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)

        # Build URL
        url = f"{self.base_url}{endpoint}"

        # Make request
        try:
            headers = self._get_headers(signed=signed)

            if method == "GET":
                response = self.session.get(
                    url, params=params, headers=headers, timeout=10
                )
            elif method == "POST":
                response = self.session.post(
                    url, params=params, headers=headers, timeout=10
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url, params=params, headers=headers, timeout=10
                )
            else:
                raise BinanceClientError(f"Unsupported HTTP method: {method}")

            # Check for errors
            if response.status_code != 200:
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise BinanceClientError(error_msg)

            return response.json()

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise BinanceClientError(error_msg)

    # =====================================================================
    # PUBLIC MARKET DATA ENDPOINTS (No authentication required)
    # =====================================================================

    def futures_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.

        Returns:
            Exchange info including all trading pairs
        """
        return self._request("GET", "/fapi/v1/exchangeInfo", weight=1)

    def futures_ticker_24hr(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get 24-hour price change statistics.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT'). If None, returns all symbols.

        Returns:
            24hr ticker data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 40  # All symbols costs 40 weight
        return self._request(
            "GET", "/fapi/v1/ticker/24hr", params=params, weight=weight
        )

    def futures_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest price for a symbol or all symbols.

        Args:
            symbol: Trading pair symbol. If None, returns all symbols.

        Returns:
            Current price data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 2
        return self._request(
            "GET", "/fapi/v1/ticker/price", params=params, weight=weight
        )

    def futures_mark_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get mark price and funding rate.

        Args:
            symbol: Trading pair symbol. If None, returns all symbols.

        Returns:
            Mark price and funding rate data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 10
        return self._request(
            "GET", "/fapi/v1/premiumIndex", params=params, weight=weight
        )

    def futures_funding_rate(
        self, symbol: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get funding rate history.

        Args:
            symbol: Trading pair symbol
            limit: Number of records to return (max 1000)

        Returns:
            List of historical funding rates
        """
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        return self._request("GET", "/fapi/v1/fundingRate", params=params, weight=1)

    def futures_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book depth.

        Args:
            symbol: Trading pair symbol
            limit: Number of levels (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Order book with bids and asks
        """
        params = {"symbol": symbol, "limit": limit}

        # Weight varies by limit
        if limit <= 50:
            weight = 2
        elif limit <= 100:
            weight = 5
        elif limit <= 500:
            weight = 10
        else:
            weight = 20

        return self._request("GET", "/fapi/v1/depth", params=params, weight=weight)

    def futures_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        """
        Get candlestick/kline data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles (max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of klines [open_time, open, high, low, close, volume, ...]
        """
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return self._request("GET", "/fapi/v1/klines", params=params, weight=1)

    def futures_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        Get current open interest.

        Args:
            symbol: Trading pair symbol

        Returns:
            Open interest data
        """
        params = {"symbol": symbol}
        return self._request("GET", "/fapi/v1/openInterest", params=params, weight=1)

    # =====================================================================
    # AUTHENTICATED ACCOUNT/TRADING ENDPOINTS
    # =====================================================================

    def futures_account(self) -> Dict[str, Any]:
        """
        Get current account information including balances and positions.

        Returns:
            Account information
        """
        return self._request("GET", "/fapi/v2/account", signed=True, weight=5)

    def futures_balance(self) -> List[Dict[str, Any]]:
        """
        Get futures account balance.

        Returns:
            List of asset balances
        """
        return self._request("GET", "/fapi/v2/balance", signed=True, weight=1)

    def futures_position_information(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current position information.

        Args:
            symbol: Trading pair symbol. If None, returns all positions.

        Returns:
            Position information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        return self._request(
            "GET", "/fapi/v2/positionRisk", params=params, signed=True, weight=5
        )

    def futures_create_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        order_type: str,  # LIMIT, MARKET, STOP, TAKE_PROFIT, etc.
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        stop_price: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new futures order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: Order type
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: GTC, IOC, FOK
            reduce_only: Reduce-only flag
            stop_price: Stop price for stop orders

        Returns:
            Order creation result
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timeInForce": time_in_force,
        }

        if price:
            params["price"] = price
        if reduce_only:
            params["reduceOnly"] = "true"
        if stop_price:
            params["stopPrice"] = stop_price

        # Add any additional parameters
        params.update(kwargs)

        return self._request(
            "POST", "/fapi/v1/order", params=params, signed=True, weight=1
        )

    def futures_cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an active order.

        Args:
            symbol: Trading pair
            order_id: Order ID to cancel

        Returns:
            Cancellation result
        """
        params = {"symbol": symbol, "orderId": order_id}
        return self._request(
            "DELETE", "/fapi/v1/order", params=params, signed=True, weight=1
        )

    def futures_get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Get order details.

        Args:
            symbol: Trading pair
            order_id: Order ID

        Returns:
            Order details
        """
        params = {"symbol": symbol, "orderId": order_id}
        return self._request(
            "GET", "/fapi/v1/order", params=params, signed=True, weight=1
        )

    def futures_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Args:
            symbol: Trading pair. If None, returns orders for all symbols.

        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 40
        return self._request(
            "GET", "/fapi/v1/openOrders", params=params, signed=True, weight=weight
        )

    def futures_change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Change initial leverage.

        Args:
            symbol: Trading pair
            leverage: Target leverage (1-125)

        Returns:
            Leverage change result
        """
        params = {"symbol": symbol, "leverage": leverage}
        return self._request(
            "POST", "/fapi/v1/leverage", params=params, signed=True, weight=1
        )

    def futures_change_margin_type(
        self, symbol: str, margin_type: str
    ) -> Dict[str, Any]:
        """
        Change margin type (ISOLATED or CROSSED).

        Args:
            symbol: Trading pair
            margin_type: ISOLATED or CROSSED

        Returns:
            Margin type change result
        """
        params = {"symbol": symbol, "marginType": margin_type}
        return self._request(
            "POST", "/fapi/v1/marginType", params=params, signed=True, weight=1
        )

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================

    def ping(self) -> bool:
        """
        Test connectivity to the API.

        Returns:
            True if ping successful
        """
        try:
            self._request("GET", "/fapi/v1/ping", weight=1)
            return True
        except BinanceClientError:
            return False

    def get_server_time(self) -> int:
        """
        Get server time.

        Returns:
            Server time in milliseconds
        """
        result = self._request("GET", "/fapi/v1/time", weight=1)
        return result["serverTime"]


def test_client():
    """Test the Binance client"""
    print("=" * 70)
    print("BINANCE CLIENT TEST")
    print("=" * 70)

    # Initialize client
    print("\n1. Initializing client...")
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    print("✓ Client initialized")

    # Test ping
    print("\n2. Testing connectivity...")
    if client.ping():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return

    # Test public endpoints
    print("\n3. Testing public endpoints...")

    try:
        # Get BTC price
        ticker = client.futures_ticker_price(symbol="BTCUSDT")
        print(f"✓ BTC Price: ${float(ticker['price']):,.2f}")

        # Get 24h stats
        stats = client.futures_ticker_24hr(symbol="BTCUSDT")
        print(f"✓ BTC 24h Change: {float(stats['priceChangePercent']):.2f}%")

        # Get mark price and funding
        mark = client.futures_mark_price(symbol="BTCUSDT")
        print(f"✓ BTC Funding Rate: {float(mark['lastFundingRate']) * 100:.4f}%")

        # Get order book
        book = client.futures_orderbook(symbol="BTCUSDT", limit=5)
        print(f"✓ Order book depth: {len(book['bids'])} bids, {len(book['asks'])} asks")

    except BinanceClientError as e:
        print(f"✗ Public endpoint test failed: {e}")

    # Test authenticated endpoints
    print("\n4. Testing authenticated endpoints...")

    try:
        balance = client.futures_balance()
        usdt_balance = next((b for b in balance if b["asset"] == "USDT"), None)
        if usdt_balance:
            print(f"✓ USDT Balance: ${float(usdt_balance['balance']):,.2f}")
        else:
            print("✓ Balance retrieved (no USDT)")

    except BinanceClientError as e:
        print(f"✗ Authentication test failed: {e}")
        print("   (Expected if using testnet with wrong keys)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_client()
=======
"""
BINANCE API CLIENT - COMPLETE FUTURES IMPLEMENTATION
====================================================
Wraps Binance futures API with proper error handling, rate limiting, and logging.

This is the CORRECT implementation with all methods needed for NullSpectre v2.0.

Author: Grim (Fixing Week 1's incomplete implementation)
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings

logger = logging.getLogger(__name__)


class BinanceClientError(Exception):
    """Custom exception for Binance API errors"""

    pass


class BinanceClient:
    """
    Binance Futures API client with complete method coverage.

    Handles:
    - Testnet and mainnet endpoints
    - Request signing for authenticated endpoints
    - Rate limiting
    - Error handling and retries
    - Response validation
    """

    # API Endpoints
    MAINNET_BASE_URL = "https://fapi.binance.com"
    TESTNET_BASE_URL = "https://testnet.binancefuture.com"

    # Rate limits (requests per minute)
    RATE_LIMIT_WEIGHT = 1200  # Default weight limit per minute

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ):
        """
        Initialize Binance Futures client.

        Args:
            api_key: API key (optional, loads from settings if not provided)
            api_secret: API secret (optional, loads from settings if not provided)
            testnet: Use testnet (True) or mainnet (False)
        """
        self.api_key = api_key or settings.BINANCE_API_KEY
        self.api_secret = api_secret or settings.BINANCE_API_SECRET
        self.testnet = testnet

        # Set base URL
        self.base_url = self.TESTNET_BASE_URL if testnet else self.MAINNET_BASE_URL

        # Setup session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Rate limiting
        self.last_request_time = 0
        self.request_weight = 0
        self.weight_reset_time = time.time() + 60

        logger.info(f"BinanceClient initialized (testnet={testnet})")

    def _get_headers(self, signed: bool = False) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json",
        }
        if signed and self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.api_secret:
            raise BinanceClientError("API secret required for signed requests")

        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def _enforce_rate_limit(self, weight: int = 1):
        """Enforce rate limiting to avoid API bans"""
        current_time = time.time()

        # Reset weight counter every minute
        if current_time > self.weight_reset_time:
            self.request_weight = 0
            self.weight_reset_time = current_time + 60

        # Check if we're over the limit
        if self.request_weight + weight > self.RATE_LIMIT_WEIGHT:
            sleep_time = self.weight_reset_time - current_time
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_weight = 0
                self.weight_reset_time = time.time() + 60

        self.request_weight += weight

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Dict[str, Any]:
        """
        Make API request with proper error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/fapi/v1/ticker/24hr")
            params: Request parameters
            signed: Whether to sign the request
            weight: Request weight for rate limiting

        Returns:
            API response as dict

        Raises:
            BinanceClientError: On API errors
        """
        if params is None:
            params = {}

        # Enforce rate limiting
        self._enforce_rate_limit(weight)

        # Add timestamp for signed requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)

        # Build URL
        url = f"{self.base_url}{endpoint}"

        # Make request
        try:
            headers = self._get_headers(signed=signed)

            if method == "GET":
                response = self.session.get(
                    url, params=params, headers=headers, timeout=10
                )
            elif method == "POST":
                response = self.session.post(
                    url, params=params, headers=headers, timeout=10
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url, params=params, headers=headers, timeout=10
                )
            else:
                raise BinanceClientError(f"Unsupported HTTP method: {method}")

            # Check for errors
            if response.status_code != 200:
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise BinanceClientError(error_msg)

            return response.json()

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise BinanceClientError(error_msg)

    # =====================================================================
    # PUBLIC MARKET DATA ENDPOINTS (No authentication required)
    # =====================================================================

    def futures_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.

        Returns:
            Exchange info including all trading pairs
        """
        return self._request("GET", "/fapi/v1/exchangeInfo", weight=1)

    def futures_ticker_24hr(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get 24-hour price change statistics.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT'). If None, returns all symbols.

        Returns:
            24hr ticker data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 40  # All symbols costs 40 weight
        return self._request(
            "GET", "/fapi/v1/ticker/24hr", params=params, weight=weight
        )

    def futures_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest price for a symbol or all symbols.

        Args:
            symbol: Trading pair symbol. If None, returns all symbols.

        Returns:
            Current price data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 2
        return self._request(
            "GET", "/fapi/v1/ticker/price", params=params, weight=weight
        )

    def futures_mark_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get mark price and funding rate.

        Args:
            symbol: Trading pair symbol. If None, returns all symbols.

        Returns:
            Mark price and funding rate data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 10
        return self._request(
            "GET", "/fapi/v1/premiumIndex", params=params, weight=weight
        )

    def futures_funding_rate(
        self, symbol: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get funding rate history.

        Args:
            symbol: Trading pair symbol
            limit: Number of records to return (max 1000)

        Returns:
            List of historical funding rates
        """
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        return self._request("GET", "/fapi/v1/fundingRate", params=params, weight=1)

    def futures_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book depth.

        Args:
            symbol: Trading pair symbol
            limit: Number of levels (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Order book with bids and asks
        """
        params = {"symbol": symbol, "limit": limit}

        # Weight varies by limit
        if limit <= 50:
            weight = 2
        elif limit <= 100:
            weight = 5
        elif limit <= 500:
            weight = 10
        else:
            weight = 20

        return self._request("GET", "/fapi/v1/depth", params=params, weight=weight)

    def futures_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        """
        Get candlestick/kline data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles (max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of klines [open_time, open, high, low, close, volume, ...]
        """
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return self._request("GET", "/fapi/v1/klines", params=params, weight=1)

    def futures_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        Get current open interest.

        Args:
            symbol: Trading pair symbol

        Returns:
            Open interest data
        """
        params = {"symbol": symbol}
        return self._request("GET", "/fapi/v1/openInterest", params=params, weight=1)

    # =====================================================================
    # AUTHENTICATED ACCOUNT/TRADING ENDPOINTS
    # =====================================================================

    def futures_account(self) -> Dict[str, Any]:
        """
        Get current account information including balances and positions.

        Returns:
            Account information
        """
        return self._request("GET", "/fapi/v2/account", signed=True, weight=5)

    def futures_balance(self) -> List[Dict[str, Any]]:
        """
        Get futures account balance.

        Returns:
            List of asset balances
        """
        return self._request("GET", "/fapi/v2/balance", signed=True, weight=1)

    def futures_position_information(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current position information.

        Args:
            symbol: Trading pair symbol. If None, returns all positions.

        Returns:
            Position information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        return self._request(
            "GET", "/fapi/v2/positionRisk", params=params, signed=True, weight=5
        )

    def futures_create_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        order_type: str,  # LIMIT, MARKET, STOP, TAKE_PROFIT, etc.
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        stop_price: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new futures order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: Order type
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: GTC, IOC, FOK
            reduce_only: Reduce-only flag
            stop_price: Stop price for stop orders

        Returns:
            Order creation result
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timeInForce": time_in_force,
        }

        if price:
            params["price"] = price
        if reduce_only:
            params["reduceOnly"] = "true"
        if stop_price:
            params["stopPrice"] = stop_price

        # Add any additional parameters
        params.update(kwargs)

        return self._request(
            "POST", "/fapi/v1/order", params=params, signed=True, weight=1
        )

    def futures_cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Cancel an active order.

        Args:
            symbol: Trading pair
            order_id: Order ID to cancel

        Returns:
            Cancellation result
        """
        params = {"symbol": symbol, "orderId": order_id}
        return self._request(
            "DELETE", "/fapi/v1/order", params=params, signed=True, weight=1
        )

    def futures_get_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """
        Get order details.

        Args:
            symbol: Trading pair
            order_id: Order ID

        Returns:
            Order details
        """
        params = {"symbol": symbol, "orderId": order_id}
        return self._request(
            "GET", "/fapi/v1/order", params=params, signed=True, weight=1
        )

    def futures_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Args:
            symbol: Trading pair. If None, returns orders for all symbols.

        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        weight = 1 if symbol else 40
        return self._request(
            "GET", "/fapi/v1/openOrders", params=params, signed=True, weight=weight
        )

    def futures_change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Change initial leverage.

        Args:
            symbol: Trading pair
            leverage: Target leverage (1-125)

        Returns:
            Leverage change result
        """
        params = {"symbol": symbol, "leverage": leverage}
        return self._request(
            "POST", "/fapi/v1/leverage", params=params, signed=True, weight=1
        )

    def futures_change_margin_type(
        self, symbol: str, margin_type: str
    ) -> Dict[str, Any]:
        """
        Change margin type (ISOLATED or CROSSED).

        Args:
            symbol: Trading pair
            margin_type: ISOLATED or CROSSED

        Returns:
            Margin type change result
        """
        params = {"symbol": symbol, "marginType": margin_type}
        return self._request(
            "POST", "/fapi/v1/marginType", params=params, signed=True, weight=1
        )

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================

    def ping(self) -> bool:
        """
        Test connectivity to the API.

        Returns:
            True if ping successful
        """
        try:
            self._request("GET", "/fapi/v1/ping", weight=1)
            return True
        except BinanceClientError:
            return False

    def get_server_time(self) -> int:
        """
        Get server time.

        Returns:
            Server time in milliseconds
        """
        result = self._request("GET", "/fapi/v1/time", weight=1)
        return result["serverTime"]


def test_client():
    """Test the Binance client"""
    print("=" * 70)
    print("BINANCE CLIENT TEST")
    print("=" * 70)

    # Initialize client
    print("\n1. Initializing client...")
    client = BinanceClient(testnet=settings.BINANCE_TESTNET)
    print("✓ Client initialized")

    # Test ping
    print("\n2. Testing connectivity...")
    if client.ping():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return

    # Test public endpoints
    print("\n3. Testing public endpoints...")

    try:
        # Get BTC price
        ticker = client.futures_ticker_price(symbol="BTCUSDT")
        print(f"✓ BTC Price: ${float(ticker['price']):,.2f}")

        # Get 24h stats
        stats = client.futures_ticker_24hr(symbol="BTCUSDT")
        print(f"✓ BTC 24h Change: {float(stats['priceChangePercent']):.2f}%")

        # Get mark price and funding
        mark = client.futures_mark_price(symbol="BTCUSDT")
        print(f"✓ BTC Funding Rate: {float(mark['lastFundingRate']) * 100:.4f}%")

        # Get order book
        book = client.futures_orderbook(symbol="BTCUSDT", limit=5)
        print(f"✓ Order book depth: {len(book['bids'])} bids, {len(book['asks'])} asks")

    except BinanceClientError as e:
        print(f"✗ Public endpoint test failed: {e}")

    # Test authenticated endpoints
    print("\n4. Testing authenticated endpoints...")

    try:
        balance = client.futures_balance()
        usdt_balance = next((b for b in balance if b["asset"] == "USDT"), None)
        if usdt_balance:
            print(f"✓ USDT Balance: ${float(usdt_balance['balance']):,.2f}")
        else:
            print("✓ Balance retrieved (no USDT)")

    except BinanceClientError as e:
        print(f"✗ Authentication test failed: {e}")
        print("   (Expected if using testnet with wrong keys)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_client()
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
