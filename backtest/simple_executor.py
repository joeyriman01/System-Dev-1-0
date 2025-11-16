<<<<<<< HEAD
"""
SIMPLE BACKTEST EXECUTOR
========================
Validates strategy on historical data with clean baseline performance.

WEEK 4: Minimum viable backtest to validate signal improvements
No complex slippage models - just clean fills at target prices.

Author: Grim (Institutional Standards)
"""

from typing import List, Dict, Optional
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from loguru import logger


@dataclass
class Trade:
    """Single trade result"""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # 'stop', 'target_1', 'target_2', 'target_3', 'runner', 'end_of_data'
    pnl_pct: float
    r_multiple: float
    conviction_score: int
    
    def to_dict(self) -> Dict:
        """Convert to dict for export"""
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pct": round(self.pnl_pct, 2),
            "r_multiple": round(self.r_multiple, 2),
            "conviction_score": self.conviction_score
        }


class SimpleBacktestExecutor:
    """
    Executes strategy signals on historical data.
    
    Tracks performance metrics: win rate, avg R, profit factor.
    
    PHILOSOPHY: Simple fills, no over-optimization. If edge exists, it'll show.
    """
    
    def __init__(self):
        self.trades: List[Trade] = []
        logger.info("SimpleBacktestExecutor initialized")
        
    def execute_signal(self, 
                      symbol: str,
                      df: pd.DataFrame,
                      entry_idx: int,
                      entry_price: float,
                      stop_price: float,
                      targets: List[float],
                      conviction_score: int) -> Optional[Trade]:
        """
        Simulate trade execution from entry to exit.
        
        Args:
            symbol: Trading pair
            df: Historical OHLCV data
            entry_idx: Index in df where entry occurs
            entry_price: Entry price
            stop_price: Stop loss price
            targets: List of profit target prices [T1, T2, T3]
            conviction_score: Signal conviction (0-100)
            
        Returns:
            Trade object if completed, None if not enough data
        """
        # Check if we have enough future data
        if entry_idx >= len(df) - 1:
            logger.warning(f"Not enough data after entry for {symbol}")
            return None
        
        # Calculate stop distance in %
        stop_distance_pct = abs(stop_price - entry_price) / entry_price * 100
        
        # Track position through time
        position_size = 1.0  # Start with 100% position
        remaining_targets = targets.copy()
        partial_pnl = 0.0  # Accumulate P&L from partials
        
        # Scan forward from entry
        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i]
            
            # Check if stop hit first (SHORT trade, so stop is ABOVE entry)
            if candle['high'] >= stop_price:
                # Position stopped out
                pnl_pct = -(stop_distance_pct) * position_size  # Negative P&L
                total_pnl = partial_pnl + pnl_pct
                r_multiple = total_pnl / stop_distance_pct
                
                trade = Trade(
                    symbol=symbol,
                    entry_time=df.iloc[entry_idx]['timestamp'],
                    entry_price=entry_price,
                    exit_time=candle['timestamp'],
                    exit_price=stop_price,
                    exit_reason='stop',
                    pnl_pct=total_pnl,
                    r_multiple=r_multiple,
                    conviction_score=conviction_score
                )
                
                logger.debug(f"{symbol} stopped out: {r_multiple:.2f}R")
                return trade
            
            # Check if any targets hit (SHORT trade, so targets are BELOW entry)
            targets_hit = []
            for target_price in remaining_targets:
                if candle['low'] <= target_price:
                    targets_hit.append(target_price)
            
            if targets_hit:
                # Take partial exits (25% per target for simplicity)
                for target_price in targets_hit:
                    partial_size = 0.25  # 25% per target
                    
                    # Calculate profit for this partial
                    profit_distance = abs(entry_price - target_price)
                    profit_pct = (profit_distance / entry_price) * 100
                    r_for_partial = profit_distance / abs(stop_price - entry_price)
                    
                    # Add to accumulated P&L
                    partial_pnl += profit_pct * partial_size
                    
                    # Reduce position size
                    position_size -= partial_size
                    remaining_targets.remove(target_price)
                    
                    logger.debug(f"{symbol} hit target ${target_price:.4f} - took 25% at {r_for_partial:.2f}R")
                
                # If all targets hit, close remaining as runner
                if position_size <= 0.01:  # Essentially closed
                    # Calculate average exit price (weighted by partials)
                    # For simplicity, use current price for runner exit
                    runner_pnl = (abs(entry_price - candle['close']) / entry_price) * 100 * 0.25
                    total_pnl = partial_pnl + runner_pnl
                    
                    avg_r = total_pnl / stop_distance_pct
                    
                    trade = Trade(
                        symbol=symbol,
                        entry_time=df.iloc[entry_idx]['timestamp'],
                        entry_price=entry_price,
                        exit_time=candle['timestamp'],
                        exit_price=candle['close'],
                        exit_reason='all_targets_hit',
                        pnl_pct=total_pnl,
                        r_multiple=avg_r,
                        conviction_score=conviction_score
                    )
                    
                    logger.debug(f"{symbol} full exit: {avg_r:.2f}R")
                    return trade
        
        # Ran out of data - close at last known price
        last_candle = df.iloc[-1]
        runner_pnl = (abs(entry_price - last_candle['close']) / entry_price) * 100 * position_size
        total_pnl = partial_pnl + runner_pnl
        r_multiple = total_pnl / stop_distance_pct
        
        trade = Trade(
            symbol=symbol,
            entry_time=df.iloc[entry_idx]['timestamp'],
            entry_price=entry_price,
            exit_time=last_candle['timestamp'],
            exit_price=last_candle['close'],
            exit_reason='end_of_data',
            pnl_pct=total_pnl,
            r_multiple=r_multiple,
            conviction_score=conviction_score
        )
        
        logger.debug(f"{symbol} closed at end of data: {r_multiple:.2f}R")
        return trade
    
    def add_trade(self, trade: Trade):
        """Add completed trade to results"""
        self.trades.append(trade)
        logger.debug(f"Trade added: {trade.symbol} - {trade.r_multiple:.2f}R")
    
    def get_performance_stats(self) -> Dict:
        """
        Calculate performance metrics from all trades.
        
        Returns comprehensive statistics including:
        - Win rate
        - Average win/loss in R multiples
        - Profit factor
        - Total R
        - Performance by conviction tier
        """
        if not self.trades:
            return {"error": "No trades to analyze"}
        
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.pnl_pct > 0]
        losers = [t for t in self.trades if t.pnl_pct <= 0]
        
        win_rate = len(winners) / total_trades * 100
        
        avg_win = sum(t.r_multiple for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t.r_multiple for t in losers) / len(losers) if losers else 0
        
        total_r = sum(t.r_multiple for t in self.trades)
        
        gross_profit = sum(t.pnl_pct for t in winners)
        gross_loss = sum(t.pnl_pct for t in losers)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # By conviction tier
        high_conviction = [t for t in self.trades if t.conviction_score >= 70]
        med_conviction = [t for t in self.trades if 50 <= t.conviction_score < 70]
        low_conviction = [t for t in self.trades if t.conviction_score < 50]
        
        def calc_tier_stats(tier_trades):
            if not tier_trades:
                return {"count": 0, "win_rate": 0, "avg_r": 0}
            return {
                "count": len(tier_trades),
                "win_rate": round(len([t for t in tier_trades if t.pnl_pct > 0]) / len(tier_trades) * 100, 2),
                "avg_r": round(sum(t.r_multiple for t in tier_trades) / len(tier_trades), 2)
            }
        
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_win_r": round(avg_win, 2),
            "avg_loss_r": round(avg_loss, 2),
            "total_r": round(total_r, 2),
            "profit_factor": round(profit_factor, 2),
            "winners": len(winners),
            "losers": len(losers),
            "by_conviction": {
                "high": calc_tier_stats(high_conviction),
                "medium": calc_tier_stats(med_conviction),
                "low": calc_tier_stats(low_conviction)
            },
            "validation_status": self._validate_edge(win_rate, avg_win, avg_loss)
        }
    
    def _validate_edge(self, win_rate: float, avg_win_r: float, avg_loss_r: float) -> Dict:
        """
        Validate if strategy has an edge based on target metrics.
        
        Target metrics from Joe's manual trading:
        - Win rate: 40-45%
        - Avg R on winners: 3-5R
        - Avg R on losers: -1R
        """
        validation = {
            "has_edge": False,
            "issues": []
        }
        
        # Check win rate
        if win_rate < 35:
            validation["issues"].append(f"Win rate too low: {win_rate:.1f}% < 35%")
        elif win_rate >= 40:
            validation["has_edge"] = True
        
        # Check avg win
        if avg_win_r < 2.5:
            validation["issues"].append(f"Avg win too small: {avg_win_r:.2f}R < 2.5R")
        elif avg_win_r >= 3.0:
            validation["has_edge"] = True
        
        # Check avg loss (should be close to -1R with tight stops)
        if avg_loss_r < -1.5:
            validation["issues"].append(f"Avg loss too large: {avg_loss_r:.2f}R (stops too wide?)")
        
        # Overall edge check
        expectancy = (win_rate / 100 * avg_win_r) + ((100 - win_rate) / 100 * avg_loss_r)
        validation["expectancy"] = round(expectancy, 2)
        
        if expectancy > 0.5:  # Positive expectancy
            validation["has_edge"] = True
        
        if not validation["issues"]:
            validation["status"] = "[OK] VALIDATED - Edge confirmed"
        else:
            validation["status"] = "[FAIL] FAILED - No edge detected"
        
        return validation
    
    def export_trades(self) -> pd.DataFrame:
        """Export all trades to DataFrame for analysis"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def print_summary(self):
        """Print formatted performance summary"""
        stats = self.get_performance_stats()
        
        if "error" in stats:
            print(stats["error"])
            return
        
        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']}%")
        print(f"  Winners: {stats['winners']} | Losers: {stats['losers']}")
        print(f"  Avg Win: {stats['avg_win_r']:.2f}R")
        print(f"  Avg Loss: {stats['avg_loss_r']:.2f}R")
        print(f"  Total R: {stats['total_r']:.2f}R")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        
        print(f"\nBY CONVICTION TIER:")
        for tier, tier_stats in stats['by_conviction'].items():
            if tier_stats['count'] > 0:
                print(f"  {tier.upper():8s}: {tier_stats['count']} trades, {tier_stats['win_rate']}% WR, {tier_stats['avg_r']:.2f}R avg")
        
        print(f"\nVALIDATION:")
        validation = stats['validation_status']
        print(f"  {validation['status']}")
        print(f"  Expectancy: {validation['expectancy']:.2f}R per trade")
        if validation['issues']:
            print(f"  Issues detected:")
            for issue in validation['issues']:
                print(f"    - {issue}")
        
        print("="*80 + "\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test backtest executor with dummy trades"""
    
    print("\n" + "="*80)
    print("SIMPLE BACKTEST EXECUTOR - TESTING")
    print("="*80 + "\n")
    
    executor = SimpleBacktestExecutor()
    
    # Create dummy historical data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    dummy_df = pd.DataFrame({
        'timestamp': dates,
        'open': 100.0,
        'high': [100 + i * 0.5 for i in range(100)],
        'low': [100 - i * 0.3 for i in range(100)],
        'close': 100.0,
        'volume': 1000.0
    })
    
    print("Simulating 5 trades...")
    print("-" * 40)
    
    # Simulate some trades
    test_trades = [
        {"entry_idx": 10, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 75},
        {"entry_idx": 30, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 65},
        {"entry_idx": 50, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 80},
        {"entry_idx": 70, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 55},
        {"entry_idx": 90, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 70},
    ]
    
    for idx, params in enumerate(test_trades, 1):
        trade = executor.execute_signal(
            symbol=f"TEST{idx}USDT",
            df=dummy_df,
            entry_idx=params["entry_idx"],
            entry_price=params["entry"],
            stop_price=params["stop"],
            targets=params["targets"],
            conviction_score=params["conviction"]
        )
        
        if trade:
            executor.add_trade(trade)
            print(f"  Trade {idx}: {trade.exit_reason:15s} - {trade.r_multiple:.2f}R")
    
    # Print summary
    executor.print_summary()
    
    # Export to DataFrame
    df_trades = executor.export_trades()
    print("Exported to DataFrame:")
    print(df_trades[['symbol', 'exit_reason', 'r_multiple', 'conviction_score']])
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
=======
"""
SIMPLE BACKTEST EXECUTOR
========================
Validates strategy on historical data with clean baseline performance.

WEEK 4: Minimum viable backtest to validate signal improvements
No complex slippage models - just clean fills at target prices.

Author: Grim (Institutional Standards)
"""

from typing import List, Dict, Optional
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from loguru import logger


@dataclass
class Trade:
    """Single trade result"""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # 'stop', 'target_1', 'target_2', 'target_3', 'runner', 'end_of_data'
    pnl_pct: float
    r_multiple: float
    conviction_score: int
    
    def to_dict(self) -> Dict:
        """Convert to dict for export"""
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pct": round(self.pnl_pct, 2),
            "r_multiple": round(self.r_multiple, 2),
            "conviction_score": self.conviction_score
        }


class SimpleBacktestExecutor:
    """
    Executes strategy signals on historical data.
    
    Tracks performance metrics: win rate, avg R, profit factor.
    
    PHILOSOPHY: Simple fills, no over-optimization. If edge exists, it'll show.
    """
    
    def __init__(self):
        self.trades: List[Trade] = []
        logger.info("SimpleBacktestExecutor initialized")
        
    def execute_signal(self, 
                      symbol: str,
                      df: pd.DataFrame,
                      entry_idx: int,
                      entry_price: float,
                      stop_price: float,
                      targets: List[float],
                      conviction_score: int) -> Optional[Trade]:
        """
        Simulate trade execution from entry to exit.
        
        Args:
            symbol: Trading pair
            df: Historical OHLCV data
            entry_idx: Index in df where entry occurs
            entry_price: Entry price
            stop_price: Stop loss price
            targets: List of profit target prices [T1, T2, T3]
            conviction_score: Signal conviction (0-100)
            
        Returns:
            Trade object if completed, None if not enough data
        """
        # Check if we have enough future data
        if entry_idx >= len(df) - 1:
            logger.warning(f"Not enough data after entry for {symbol}")
            return None
        
        # Calculate stop distance in %
        stop_distance_pct = abs(stop_price - entry_price) / entry_price * 100
        
        # Track position through time
        position_size = 1.0  # Start with 100% position
        remaining_targets = targets.copy()
        partial_pnl = 0.0  # Accumulate P&L from partials
        
        # Scan forward from entry
        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i]
            
            # Check if stop hit first (SHORT trade, so stop is ABOVE entry)
            if candle['high'] >= stop_price:
                # Position stopped out
                pnl_pct = -(stop_distance_pct) * position_size  # Negative P&L
                total_pnl = partial_pnl + pnl_pct
                r_multiple = total_pnl / stop_distance_pct
                
                trade = Trade(
                    symbol=symbol,
                    entry_time=df.iloc[entry_idx]['timestamp'],
                    entry_price=entry_price,
                    exit_time=candle['timestamp'],
                    exit_price=stop_price,
                    exit_reason='stop',
                    pnl_pct=total_pnl,
                    r_multiple=r_multiple,
                    conviction_score=conviction_score
                )
                
                logger.debug(f"{symbol} stopped out: {r_multiple:.2f}R")
                return trade
            
            # Check if any targets hit (SHORT trade, so targets are BELOW entry)
            targets_hit = []
            for target_price in remaining_targets:
                if candle['low'] <= target_price:
                    targets_hit.append(target_price)
            
            if targets_hit:
                # Take partial exits (25% per target for simplicity)
                for target_price in targets_hit:
                    partial_size = 0.25  # 25% per target
                    
                    # Calculate profit for this partial
                    profit_distance = abs(entry_price - target_price)
                    profit_pct = (profit_distance / entry_price) * 100
                    r_for_partial = profit_distance / abs(stop_price - entry_price)
                    
                    # Add to accumulated P&L
                    partial_pnl += profit_pct * partial_size
                    
                    # Reduce position size
                    position_size -= partial_size
                    remaining_targets.remove(target_price)
                    
                    logger.debug(f"{symbol} hit target ${target_price:.4f} - took 25% at {r_for_partial:.2f}R")
                
                # If all targets hit, close remaining as runner
                if position_size <= 0.01:  # Essentially closed
                    # Calculate average exit price (weighted by partials)
                    # For simplicity, use current price for runner exit
                    runner_pnl = (abs(entry_price - candle['close']) / entry_price) * 100 * 0.25
                    total_pnl = partial_pnl + runner_pnl
                    
                    avg_r = total_pnl / stop_distance_pct
                    
                    trade = Trade(
                        symbol=symbol,
                        entry_time=df.iloc[entry_idx]['timestamp'],
                        entry_price=entry_price,
                        exit_time=candle['timestamp'],
                        exit_price=candle['close'],
                        exit_reason='all_targets_hit',
                        pnl_pct=total_pnl,
                        r_multiple=avg_r,
                        conviction_score=conviction_score
                    )
                    
                    logger.debug(f"{symbol} full exit: {avg_r:.2f}R")
                    return trade
        
        # Ran out of data - close at last known price
        last_candle = df.iloc[-1]
        runner_pnl = (abs(entry_price - last_candle['close']) / entry_price) * 100 * position_size
        total_pnl = partial_pnl + runner_pnl
        r_multiple = total_pnl / stop_distance_pct
        
        trade = Trade(
            symbol=symbol,
            entry_time=df.iloc[entry_idx]['timestamp'],
            entry_price=entry_price,
            exit_time=last_candle['timestamp'],
            exit_price=last_candle['close'],
            exit_reason='end_of_data',
            pnl_pct=total_pnl,
            r_multiple=r_multiple,
            conviction_score=conviction_score
        )
        
        logger.debug(f"{symbol} closed at end of data: {r_multiple:.2f}R")
        return trade
    
    def add_trade(self, trade: Trade):
        """Add completed trade to results"""
        self.trades.append(trade)
        logger.debug(f"Trade added: {trade.symbol} - {trade.r_multiple:.2f}R")
    
    def get_performance_stats(self) -> Dict:
        """
        Calculate performance metrics from all trades.
        
        Returns comprehensive statistics including:
        - Win rate
        - Average win/loss in R multiples
        - Profit factor
        - Total R
        - Performance by conviction tier
        """
        if not self.trades:
            return {"error": "No trades to analyze"}
        
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.pnl_pct > 0]
        losers = [t for t in self.trades if t.pnl_pct <= 0]
        
        win_rate = len(winners) / total_trades * 100
        
        avg_win = sum(t.r_multiple for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t.r_multiple for t in losers) / len(losers) if losers else 0
        
        total_r = sum(t.r_multiple for t in self.trades)
        
        gross_profit = sum(t.pnl_pct for t in winners)
        gross_loss = sum(t.pnl_pct for t in losers)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # By conviction tier
        high_conviction = [t for t in self.trades if t.conviction_score >= 70]
        med_conviction = [t for t in self.trades if 50 <= t.conviction_score < 70]
        low_conviction = [t for t in self.trades if t.conviction_score < 50]
        
        def calc_tier_stats(tier_trades):
            if not tier_trades:
                return {"count": 0, "win_rate": 0, "avg_r": 0}
            return {
                "count": len(tier_trades),
                "win_rate": round(len([t for t in tier_trades if t.pnl_pct > 0]) / len(tier_trades) * 100, 2),
                "avg_r": round(sum(t.r_multiple for t in tier_trades) / len(tier_trades), 2)
            }
        
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_win_r": round(avg_win, 2),
            "avg_loss_r": round(avg_loss, 2),
            "total_r": round(total_r, 2),
            "profit_factor": round(profit_factor, 2),
            "winners": len(winners),
            "losers": len(losers),
            "by_conviction": {
                "high": calc_tier_stats(high_conviction),
                "medium": calc_tier_stats(med_conviction),
                "low": calc_tier_stats(low_conviction)
            },
            "validation_status": self._validate_edge(win_rate, avg_win, avg_loss)
        }
    
    def _validate_edge(self, win_rate: float, avg_win_r: float, avg_loss_r: float) -> Dict:
        """
        Validate if strategy has an edge based on target metrics.
        
        Target metrics from Joe's manual trading:
        - Win rate: 40-45%
        - Avg R on winners: 3-5R
        - Avg R on losers: -1R
        """
        validation = {
            "has_edge": False,
            "issues": []
        }
        
        # Check win rate
        if win_rate < 35:
            validation["issues"].append(f"Win rate too low: {win_rate:.1f}% < 35%")
        elif win_rate >= 40:
            validation["has_edge"] = True
        
        # Check avg win
        if avg_win_r < 2.5:
            validation["issues"].append(f"Avg win too small: {avg_win_r:.2f}R < 2.5R")
        elif avg_win_r >= 3.0:
            validation["has_edge"] = True
        
        # Check avg loss (should be close to -1R with tight stops)
        if avg_loss_r < -1.5:
            validation["issues"].append(f"Avg loss too large: {avg_loss_r:.2f}R (stops too wide?)")
        
        # Overall edge check
        expectancy = (win_rate / 100 * avg_win_r) + ((100 - win_rate) / 100 * avg_loss_r)
        validation["expectancy"] = round(expectancy, 2)
        
        if expectancy > 0.5:  # Positive expectancy
            validation["has_edge"] = True
        
        if not validation["issues"]:
            validation["status"] = "âœ" VALIDATED - Edge confirmed"
        else:
            validation["status"] = "âœ— FAILED - No edge detected"
        
        return validation
    
    def export_trades(self) -> pd.DataFrame:
        """Export all trades to DataFrame for analysis"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def print_summary(self):
        """Print formatted performance summary"""
        stats = self.get_performance_stats()
        
        if "error" in stats:
            print(stats["error"])
            return
        
        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']}%")
        print(f"  Winners: {stats['winners']} | Losers: {stats['losers']}")
        print(f"  Avg Win: {stats['avg_win_r']:.2f}R")
        print(f"  Avg Loss: {stats['avg_loss_r']:.2f}R")
        print(f"  Total R: {stats['total_r']:.2f}R")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        
        print(f"\nBY CONVICTION TIER:")
        for tier, tier_stats in stats['by_conviction'].items():
            if tier_stats['count'] > 0:
                print(f"  {tier.upper():8s}: {tier_stats['count']} trades, {tier_stats['win_rate']}% WR, {tier_stats['avg_r']:.2f}R avg")
        
        print(f"\nVALIDATION:")
        validation = stats['validation_status']
        print(f"  {validation['status']}")
        print(f"  Expectancy: {validation['expectancy']:.2f}R per trade")
        if validation['issues']:
            print(f"  Issues detected:")
            for issue in validation['issues']:
                print(f"    - {issue}")
        
        print("="*80 + "\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test backtest executor with dummy trades"""
    
    print("\n" + "="*80)
    print("SIMPLE BACKTEST EXECUTOR - TESTING")
    print("="*80 + "\n")
    
    executor = SimpleBacktestExecutor()
    
    # Create dummy historical data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    dummy_df = pd.DataFrame({
        'timestamp': dates,
        'open': 100.0,
        'high': [100 + i * 0.5 for i in range(100)],
        'low': [100 - i * 0.3 for i in range(100)],
        'close': 100.0,
        'volume': 1000.0
    })
    
    print("Simulating 5 trades...")
    print("-" * 40)
    
    # Simulate some trades
    test_trades = [
        {"entry_idx": 10, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 75},
        {"entry_idx": 30, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 65},
        {"entry_idx": 50, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 80},
        {"entry_idx": 70, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 55},
        {"entry_idx": 90, "entry": 100, "stop": 105, "targets": [98, 96, 94], "conviction": 70},
    ]
    
    for idx, params in enumerate(test_trades, 1):
        trade = executor.execute_signal(
            symbol=f"TEST{idx}USDT",
            df=dummy_df,
            entry_idx=params["entry_idx"],
            entry_price=params["entry"],
            stop_price=params["stop"],
            targets=params["targets"],
            conviction_score=params["conviction"]
        )
        
        if trade:
            executor.add_trade(trade)
            print(f"  Trade {idx}: {trade.exit_reason:15s} - {trade.r_multiple:.2f}R")
    
    # Print summary
    executor.print_summary()
    
    # Export to DataFrame
    df_trades = executor.export_trades()
    print("Exported to DataFrame:")
    print(df_trades[['symbol', 'exit_reason', 'r_multiple', 'conviction_score']])
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
    print("="*80)