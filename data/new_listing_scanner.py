"""
NEW LISTING SCANNER
===================
Scans for new perpetual futures listings and prioritizes by setup quality.

Now works with both Binance and MEXC.

Author: Grim (Institutional Standards)
"""

import os
from typing import List, Dict
from datetime import datetime, timedelta
from loguru import logger

# Import both exchange clients
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.binance_client import BinanceClient
from data.mexc_client import MEXCClient

# Auto-detect which exchange to use
USE_MEXC = os.getenv('USE_MEXC', 'False').lower() == 'true'


class NewListingScanner:
    """
    Scans for new perpetual futures listings.
    
    Prioritizes coins based on:
    - Volume (high volume = retail interest)
    - Volatility (big moves = pump potential)
    - Setup quality (rejections, funding rate)
    """
    
    def __init__(self, client=None):
        """
        Initialize scanner.
        
        Args:
            client: Exchange client (auto-creates if None)
        """
        # Auto-create client if not provided
        if client is None:
            if USE_MEXC:
                self.client = MEXCClient()
                self.exchange_name = "MEXC"
                logger.info("NewListingScanner using MEXC")
            else:
                self.client = BinanceClient()
                self.exchange_name = "Binance"
                logger.info("NewListingScanner using Binance")
        else:
            self.client = client
            self.exchange_name = "MEXC" if USE_MEXC else "Binance"
        
        logger.info("NewListingScanner initialized")
    
    
    def scan_all_perpetuals(self) -> List[Dict]:
        """
        Scan all perpetual contracts and get basic stats.
        
        Returns:
            List of dicts with symbol info
        """
        logger.info(f"Scanning all perpetuals on {self.exchange_name}...")
        
        # Get all available symbols
        symbols = self.client.get_available_symbols()
        
        logger.info(f"Found {len(symbols)} perpetual contracts")
        
        results = []
        
        for symbol in symbols:
            try:
                # Get ticker data
                ticker = self.client.get_ticker(symbol)
                
                if not ticker:
                    continue
                
                # Extract key metrics
                last_price = float(ticker.get('lastPrice', 0))
                volume_24h = float(ticker.get('volume24', 0))
                change_24h = float(ticker.get('riseFallRate', 0))
                
                if last_price == 0:
                    continue
                
                results.append({
                    'symbol': symbol,
                    'price': last_price,
                    'volume_24h': volume_24h,
                    'change_24h_pct': change_24h * 100,
                    'exchange': self.exchange_name
                })
                
            except Exception as e:
                logger.debug(f"Failed to get data for {symbol}: {e}")
                continue
        
        logger.info(f"Successfully scanned {len(results)} symbols")
        
        return results
    
    
    def filter_meme_coins(
        self,
        all_coins: List[Dict],
        min_volume: float = 1_000_000,
        min_change: float = 5.0
    ) -> List[Dict]:
        """
        Filter for potential meme coin pumps.
        
        Args:
            all_coins: List of coin data
            min_volume: Minimum 24h volume (USD)
            min_change: Minimum 24h price change (%)
            
        Returns:
            Filtered list sorted by potential
        """
        logger.info(f"Filtering for meme coins (vol>${min_volume:,.0f}, change>{min_change}%)")
        
        # Filter
        candidates = [
            coin for coin in all_coins
            if coin['volume_24h'] >= min_volume
            and abs(coin['change_24h_pct']) >= min_change
        ]
        
        # Sort by volume (proxy for retail interest)
        candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        logger.info(f"Found {len(candidates)} candidates")
        
        return candidates
    
    
    def scan_with_signals(
        self,
        min_volume: float = 1_000_000,
        min_change: float = 5.0,
        min_conviction: float = 40
    ) -> List[Dict]:
        """
        Scan for coins AND generate signals for top candidates.
        
        Args:
            min_volume: Minimum 24h volume
            min_change: Minimum 24h change
            min_conviction: Minimum conviction score
            
        Returns:
            List of coins with signals
        """
        # Import signal generator here to avoid circular imports
        from strategy.signal_generator import SignalGenerator
        
        logger.info("Running full scan with signal generation...")
        
        # Get all coins
        all_coins = self.scan_all_perpetuals()
        
        # Filter for candidates
        candidates = self.filter_meme_coins(all_coins, min_volume, min_change)
        
        # Take top 20 by volume
        top_candidates = candidates[:20]
        
        logger.info(f"Analyzing top {len(top_candidates)} candidates for signals...")
        
        # Generate signals
        generator = SignalGenerator(client=self.client, min_conviction=min_conviction)
        
        results_with_signals = []
        
        for coin in top_candidates:
            symbol = coin['symbol']
            
            try:
                logger.info(f"  Analyzing {symbol}...")
                
                signal = generator.generate_signal(
                    symbol=symbol,
                    rejection_timeframe='1h',
                    cvd_timeframe='15m'
                )
                
                if signal:
                    coin['signal'] = signal
                    coin['has_signal'] = True
                    logger.info(f"    ✓ Signal generated (conviction: {signal['conviction_score']})")
                else:
                    coin['signal'] = None
                    coin['has_signal'] = False
                    logger.info(f"    ✗ No signal (low conviction)")
                
                results_with_signals.append(coin)
                
            except Exception as e:
                logger.warning(f"  Failed to generate signal for {symbol}: {e}")
                coin['signal'] = None
                coin['has_signal'] = False
                results_with_signals.append(coin)
        
        # Sort by those with signals first, then by conviction score
        results_with_signals.sort(
            key=lambda x: (
                x['has_signal'],
                x['signal']['conviction_score'] if x['signal'] else 0
            ),
            reverse=True
        )
        
        signals_found = sum(1 for r in results_with_signals if r['has_signal'])
        logger.info(f"Scan complete: {signals_found} signals generated")
        
        return results_with_signals


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test new listing scanner"""
    
    print("\n" + "="*80)
    print("NEW LISTING SCANNER - TESTING")
    print("="*80 + "\n")
    
    # Auto-select exchange
    exchange_name = "MEXC" if USE_MEXC else "Binance"
    print(f"Using exchange: {exchange_name}\n")
    
    # Initialize scanner
    scanner = NewListingScanner()
    
    # Test 1: Scan all perpetuals
    print("TEST 1: Scan All Perpetuals")
    print("-" * 40)
    
    all_coins = scanner.scan_all_perpetuals()
    
    print(f"Found {len(all_coins)} perpetual contracts\n")
    print("Top 10 by volume:")
    
    top_10 = sorted(all_coins, key=lambda x: x['volume_24h'], reverse=True)[:10]
    for i, coin in enumerate(top_10, 1):
        print(f"  {i}. {coin['symbol']}: ${coin['volume_24h']:,.0f} vol, {coin['change_24h_pct']:+.2f}%")
    
    print()
    
    # Test 2: Filter meme coins
    print("TEST 2: Filter Meme Coins")
    print("-" * 40)
    
    meme_candidates = scanner.filter_meme_coins(
        all_coins,
        min_volume=1_000_000,
        min_change=5.0
    )
    
    print(f"Found {len(meme_candidates)} candidates\n")
    print("Top 5 candidates:")
    
    for i, coin in enumerate(meme_candidates[:5], 1):
        print(f"  {i}. {coin['symbol']}")
        print(f"     Price: ${coin['price']:.4f}")
        print(f"     Volume: ${coin['volume_24h']:,.0f}")
        print(f"     Change: {coin['change_24h_pct']:+.2f}%")
        print()
    
    # Test 3: Scan with signals (optional - takes longer)
    print("TEST 3: Scan With Signals (Top 5)")
    print("-" * 40)
    print("This may take a minute...\n")
    
    # Only scan top 5 for testing
    from strategy.signal_generator import SignalGenerator
    generator = SignalGenerator(min_conviction=30)
    
    for coin in meme_candidates[:5]:
        symbol = coin['symbol']
        try:
            signal = generator.generate_signal(symbol)
            if signal:
                print(f"✓ {symbol}: Signal generated")
                print(f"  Conviction: {signal['conviction_score']}/100")
                print(f"  Entry: ${signal['entry_price']:.4f}")
                print(f"  R:R: {signal['risk_reward_ratio']:.2f}")
            else:
                print(f"✗ {symbol}: No signal (low conviction)")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")
        print()
    
    print("="*80)
    print("TESTING COMPLETE")
    print("="*80)