# NullSpectre v2.0 - Updated Development Roadmap
**Version:** 2.1 (Research-Enhanced)  
**Last Updated:** November 15, 2025  
**Status:** Week 4 - Pattern Validation & Enhancement Phase  

---

## Project Overview

**Mission:** Build a systematic trading system that automates Joe's proven manual edge for shorting meme coin exhaustion patterns on cryptocurrency perpetual futures.

**Proven Manual Performance:**
- Win Rate: 42%
- Risk:Reward: 3-5:1 average
- Account Growth: 3X over several months
- Trade Frequency: 2-4 setups per week

**Core Edge:** Fading retail exhaustion after 3+ rejection attempts at resistance, confirmed by order flow divergence and funding rate positioning.

**Why Automation:** The "speed problem" - manual execution misses quick runner exits and loses opportunities during analysis delays.

---

## Development Timeline Overview

| Phase | Weeks | Status | Completion |
|-------|-------|--------|------------|
| **Phase 1: Foundation** | 1-2 | ✅ COMPLETE | 100% |
| **Phase 2: Strategy Logic** | 3 | ✅ COMPLETE | 100% |
| **Phase 3: Enhancement & Validation** | 4-5 | 🟡 IN PROGRESS | 60% |
| **Phase 4: Risk & Execution** | 6-8 | ⚪ PENDING | 0% |
| **Phase 5: Testing & Deployment** | 9-12 | ⚪ PENDING | 0% |

---

## Phase 1: Foundation (Weeks 1-2) ✅ COMPLETE

### Week 1: Infrastructure Setup ✅
- ✅ Project structure created
- ✅ Python environment (venv + requirements.txt)
- ✅ MEXC API integration (REST + WebSocket)
- ✅ SQLite database setup
- ✅ Configuration management (Pydantic v2)
- ✅ Logging system (loguru)

**Deliverable:** ✅ Core infrastructure operational with 5/5 tests passing

---

### Week 2: Data Collection Modules ✅
- ✅ New Listing Scanner (Module 1)
  - Scans 512 perpetual contracts on MEXC
  - Prioritizes by volume, volatility, setup quality
  
- ✅ Market Data Aggregator (Module 2)
  - Multi-timeframe candle analysis (15M, 1H, 4H, Daily)
  - ATR calculations for volatility
  - Support/resistance detection
  
- ✅ Funding Rate Tracker (Module 3)
  - Detects overleveraged positioning
  - Funding rate flip detection
  - Historical funding analysis
  
- ✅ Order Book Monitor (Module 4)
  - Real-time bid/ask imbalance tracking
  - Supply/demand pressure analysis
  - Depth of market monitoring

**Deliverable:** ✅ System can scan and collect comprehensive market data

---

## Phase 2: Strategy Logic (Week 3) ✅ COMPLETE

### Week 3: Core Strategy Modules ✅
- ✅ Rejection Pattern Detector
  - Identifies resistance levels
  - Counts valid rejection attempts
  - Qualifies setups at 3+ rejections
  
- ✅ CVD Analyzer
  - Calculates Cumulative Volume Delta
  - Detects bearish divergence patterns
  - Order flow confirmation
  
- ✅ Conviction Scorer
  - Combines all signals into 0-100 score
  - Maps scores to conviction tiers (MAX/HIGH/MEDIUM/SKIP)
  - Determines leverage and risk allocation
  
- ✅ Signal Generator
  - Creates complete trade instructions
  - Calculates entry prices, stops, targets
  - Generates tiered profit-taking levels

**Deliverable:** ✅ End-to-end signal generation working (BTC test signal: 51.5/100 conviction)

---

## Phase 3: Enhancement & Validation (Weeks 4-5) 🟡 IN PROGRESS

### Week 4: Research-Backed Enhancements 🟡 60% COMPLETE

**Current Status:**
- ✅ Historical data collection framework exists
- ✅ Basic backtesting engine structure in place
- ✅ Pattern validation tools created
- 🟡 Research analysis complete (P&D detection paper reviewed)
- ⚪ Strategy enhancements pending implementation
- ⚪ Validation against historical pumps pending

---

#### 🎯 PRIORITY 1: Market Cap Filter (NEW)
**Status:** ⚪ TO BUILD  
**Timeline:** 1 hour  
**Research Basis:** Paper 2412.18848 - 95.7% of pumped coins have market cap <$60M

**Implementation:**
```
File: strategy/market_cap_filter.py (NEW)
Purpose: Reduce scanner noise from 512 → ~50 pump-vulnerable coins
API: CoinMarketCap or CoinGecko
Thresholds:
  - HIGH RISK: Market cap <$2.7M (median of pumped coins)
  - MEDIUM RISK: Market cap $2.7M-$60M
  - FILTER OUT: Market cap >$60M
```

**Expected Impact:**
- 90% reduction in scanner noise
- Focus on truly vulnerable meme coins
- Better signal-to-noise ratio

---

#### 🎯 PRIORITY 2: Rejection Quality Scoring (ENHANCE EXISTING)
**Status:** ⚪ TO BUILD  
**Timeline:** 2 hours  
**Research Basis:** Not all 3-pump patterns are equal - quality matters

**Implementation:**
```
File: strategy/rejection_pattern.py (ENHANCE)
Current: count_rejections() - simple counter
Add: score_rejection_quality() - quality assessment

Quality Factors:
1. Wick Length (40 points)
   - Long wick >50% of candle range = bullish rejection
   - Score: (wick_length / total_range) * 40

2. Close Position (30 points)
   - Close in lower 30% of range = sellers controlled
   - Score: (1 - close_position) * 30

3. Volume Spike (30 points)
   - Volume >120% of average = real selling
   - Score: min(volume_ratio / 1.2, 1.0) * 30

Total Score: 0-100 per rejection candle
Pattern Score: Average of all rejection candles
```

**Additional Enhancements:**
```
Rejection Spacing Analysis:
- Tight spacing (<2 hours) = EXHAUSTION (score +30)
- Wide spacing (>6 hours) = CONSOLIDATION (score -20)

Rejection Acceleration:
- Descending peaks (lower highs each attempt) = TRUE EXHAUSTION (score +40)
- Ascending peaks = FALSE SIGNAL (score -30)
```

**Expected Impact:**
- Filter out low-quality 3-pump patterns
- Increase win rate by 10-15 percentage points
- Better conviction scoring accuracy

---

#### 🎯 PRIORITY 3: Support Level Detection (ENHANCE EXISTING)
**Status:** ⚪ TO BUILD  
**Timeline:** 3 hours  
**Research Basis:** Target actual support levels instead of fixed % gains

**Implementation:**
```
File: strategy/signal_generator.py (ENHANCE)
Current: Fixed % targets (50%, 100%, 200%)
Add: identify_support_levels() - volume profile analysis

Volume Profile Calculation:
1. Get 30 days of historical candles
2. Create 100 price buckets across range
3. Sum volume at each price level
4. Identify top 5 High Volume Nodes (HVN)
5. HVN = support levels where price spent time + volume

Target Calculation:
- Replace fixed % with nearest support levels below entry
- Target 1 (33%): First support (quick scalp)
- Target 2 (33%): Second support (main target)
- Target 3 (34%): Third support (runner)

Calculate R-Multiple per target:
R = (entry - target) / stop_distance
```

**Expected Impact:**
- Higher target hit rate (support levels have buyers)
- Better R-multiples on average
- More logical profit-taking structure

---

#### 🔬 PRIORITY 4: Microstructure Analysis (ENHANCE EXISTING)
**Status:** ⚪ TO BUILD  
**Timeline:** 4 hours  
**Research Basis:** Paper 2412.18848 - Order book patterns at true exhaustion

**Implementation:**
```
File: strategy/cvd_analyzer.py (ENHANCE)
Current: Basic CVD calculation
Add: analyze_rejection_microstructure() - order book deep dive

Microstructure Signals at Rejection Peak:
1. Bid/Ask Depth Ratio (30 points)
   - Ask depth / Bid depth
   - >1.5 = sellers dominating (bullish for short)
   - Score: min(ratio / 1.5, 1.0) * 30

2. Recent Trade Side (40 points)
   - Last 100 trades: sell volume / total volume
   - >60% sells = heavy distribution
   - Score: (sell_ratio - 0.5) * 80

3. Large Order Detection (30 points)
   - Count trades >2x average size
   - 3+ large sells = whales exiting
   - Score: min(large_sell_count / 3, 1.0) * 30

Combined Microstructure Score: 0-100
Threshold: >70 = Exhaustion Confirmed
```

**Expected Impact:**
- Distinguish coordinated pumps from organic exhaustion
- Reduce false signals by 20-30%
- Higher conviction on confirmed setups

---

### Week 5: Validation & Calibration ⚪ PENDING

#### Historical Pattern Validation
**Status:** ⚪ TO BUILD  
**Timeline:** 2 days

**Tasks:**
1. Build Simple Backtest Executor
   - File: backtest/simple_executor.py (NEW)
   - Simulate short entries and exits
   - Track P&L, max adverse excursion, target hit rates
   
2. Collect 20-50 Historical Pump Examples
   - Use existing data/historical/ CSV files
   - Add more if needed from recent MEXC pumps
   - Tag each with outcome (worked/failed)
   
3. Run Enhanced Strategy Against Historical Data
   - Test WITH enhancements vs WITHOUT
   - Compare:
     - Win rate (target: 45%+)
     - Average R-multiple (target: 3:1+)
     - Signal quality (fewer false positives)
     - Target hit rates (support vs fixed %)

**Success Criteria:**
- ✅ Win rate ≥40% (maintain manual performance)
- ✅ Avg R-multiple ≥3:1 (maintain manual performance)
- ✅ Rejection quality score correlates with wins (t-test p<0.05)
- ✅ Support-based targets hit 15%+ more than fixed %
- ✅ Microstructure score >70 improves win rate by 10%+

**If Validation Fails (<35% win rate):**
- Debug signal generation logic
- Review rejected patterns for commonalities
- Adjust thresholds before proceeding to execution

---

## Phase 4: Risk Management & Execution (Weeks 6-8) ⚪ PENDING

### Week 6: Position Sizing & Risk Controls ⚪

#### Kelly Criterion Position Sizing
**Status:** ⚪ TO BUILD  
**File:** risk/position_sizer.py

**Implementation:**
```python
Kelly Formula: f* = (p*b - q) / b
Where:
  p = win probability (from live data)
  b = avg_win / avg_loss ratio
  q = 1 - p

Conservative Approach: Use 1/4 Kelly or 1/2 Kelly
Adjustments:
  - Multiply by conviction score (0-100 → 0-2x multiplier)
  - Reduce by 50% after 3+ consecutive losses
  - Cap at 12% account risk (max conviction)
```

**Risk Controls:**
```python
Account-Level:
- Max daily loss: 20% of account
- Max consecutive losses: 6 (then pause/reduce size)
- Max concurrent positions: 3
- Max exposure per symbol: 15%

Position-Level:
- Max leverage: 100X (only on MAX conviction)
- Stop loss: ALWAYS required, never removed
- Max stop distance: 20% (prevents runaway risk)
```

---

#### ATR-Adjusted Stop Placement
**Status:** ⚪ TO BUILD  
**File:** strategy/signal_generator.py (ENHANCE)

**Implementation:**
```python
Current: Fixed buffer above rejection high
Enhanced: ATR-based volatility adjustment

Stop Calculation:
1. Base stop = rejection_high * 1.005 (0.5% buffer)
2. ATR stop = rejection_high + (1.5 * ATR)
3. Final stop = max(base_stop, atr_stop)  # Use wider

Exchange Adjustment:
- MEXC: 1.2x multiplier (low liquidity, more gaps)
- Binance: 1.0x (baseline)
- KuCoin: 0.9x (high liquidity, tighter)

Validates: Stop distance <20% of entry
```

---

### Week 7: Order Execution Engine ⚪

**Status:** ⚪ TO BUILD  
**Files:** execution/ directory (all empty currently)

**Components to Build:**
1. order_executor.py
   - Place limit orders via MEXC API
   - Monitor fill status
   - Handle partial fills
   - Cancel unfilled orders after timeout
   
2. position_manager.py
   - Track open positions
   - Calculate real-time P&L
   - Monitor max adverse excursion
   - Update position state
   
3. stop_manager.py
   - Place stop-loss orders
   - Move stops to breakeven (after 1st partial)
   - Trail stops (optional, based on testing)
   - Emergency stop updates
   
4. profit_manager.py
   - Execute partial exits at targets
   - Track which partials filled
   - Manage runner position
   - Monitor support bounce signals for runner exit

**Testing Approach:**
- Use MEXC testnet first (if available)
- Paper trading mode with live prices but simulated fills
- Start with minimum position sizes
- Verify order routing and fill logic

---

### Week 8: Position Monitoring & Runner Management ⚪

**Status:** ⚪ TO BUILD

**Advanced Features:**
1. Runner Exit Detection
   - Monitor support level touches (2-3+ bounces)
   - Check CVD flip (from negative to positive)
   - Volume spike on bounce (>1.3x average)
   - Alert user when runner exit conditions met
   
2. Support Bounce Analysis
   - Track price at support levels
   - Calculate bounce magnitude
   - Determine if bounce is real or fake-out
   - Generate runner exit signal

**Implementation:**
```python
File: execution/runner_monitor.py (NEW)

Runner Exit Logic:
1. Track support touches:
   - Count touches within 1% of support level
   - Space touches by 5+ minutes (avoid noise)

2. Confirm bounce:
   - Price moves >1% off support (adaptive by ATR)
   - Volume >1.3x recent average
   - CVD flips positive (buyers entering)

3. Exit signal:
   - 2 touches + volume + CVD = ALERT user
   - 3+ touches + 1 other signal = AUTO close
```

---

## Phase 5: Testing & Deployment (Weeks 9-12) ⚪ PENDING

### Week 9-10: Paper Trading & Validation ⚪

**Paper Trading Mode:**
- Run system 24/7 on live market data
- Simulate all orders (no real money)
- Log every signal, decision, hypothetical fill
- Track as if real: P&L, drawdown, Sharpe ratio

**Daily Review Process:**
1. Compare system signals to manual analysis
2. Check if paper trades match expected behavior
3. Verify risk management working correctly
4. Log any bugs or edge cases

**Success Criteria (50 Paper Trades):**
- ✅ Win rate ≥35% (minimum acceptable)
- ✅ Avg R-multiple ≥2.5:1
- ✅ Max drawdown ≤25%
- ✅ Sharpe ratio ≥1.0
- ✅ No critical bugs in 50 trades
- ✅ Fill simulation matches expected slippage

**If Paper Trading Fails:**
- Debug execution logic
- Review signal quality vs actual market behavior
- Adjust thresholds and parameters
- Repeat paper trading until stable

---

### Week 11: Live Trading (Micro Capital) ⚪

**Live Trading Phase 1: £100 Test Capital**
- Enable real order execution on MEXC
- Trade ONLY MAX conviction signals (score ≥80)
- Manual confirmation required for EVERY trade
- Maximum 1 position at a time

**Metrics to Track:**
1. Execution Quality
   - Fill rate (orders filled vs cancelled)
   - Average slippage (filled price vs expected)
   - Latency (signal → order placement time)
   - Order rejection rate
   
2. Performance Comparison
   - Live P&L vs paper trading expectation
   - Win rate: live vs backtest vs paper
   - R-multiple: live vs expected
   - Any surprises or anomalies

**Daily Risk Checks:**
- Verify stop-loss orders placed correctly
- Check position sizing calculations
- Monitor for any execution errors
- Log every trade with full context

**Success Criteria (10 Live Trades):**
- ✅ No execution errors or bugs
- ✅ Fill slippage <2% on average
- ✅ Stops work correctly (hit when expected)
- ✅ P&L matches paper trading expectations (±20%)

---

### Week 12: Scale to Full Capital ⚪

**Live Trading Phase 2: £3-4K Capital**
- Increase capital allocation
- Enable HIGH conviction trades (score ≥60)
- Keep manual confirmation initially
- Multiple concurrent positions allowed (max 3)

**Optional: Semi-Auto Mode**
- System generates signal
- Shows full breakdown + confidence
- 5-second countdown
- Auto-executes if user doesn't cancel
- Allows override but defaults to execution

**Full Monitoring Dashboard:**
- Real-time position status
- P&L tracking
- Risk metrics (daily loss, exposure, leverage)
- Performance analytics
- Alert system for emergencies

**End of Week 12 Deliverable:**
- ✅ System trading live profitably
- ✅ Performance report: win rate, Sharpe, drawdown
- ✅ Ready for full capital deployment

---

## Research-Backed Enhancements Summary

### Implemented (Based on Paper 2412.18848)
| Enhancement | Priority | Status | Expected Impact |
|-------------|----------|--------|-----------------|
| Market cap filter (<$60M) | P0 | ⚪ Week 4 | 90% noise reduction |
| Rejection quality scoring | P0 | ⚪ Week 4 | +10-15% win rate |
| Support level detection | P0 | ⚪ Week 4 | +15% target hit rate |
| Microstructure analysis | P1 | ⚪ Week 5 | -20% false signals |
| ATR-adjusted stops | P2 | ⚪ Week 6 | Better risk-adjusted returns |
| Kelly position sizing | P2 | ⚪ Week 6 | Optimal capital allocation |

### Deferred (Requires Live Data First)
| Enhancement | Reason to Defer | Decision Criteria |
|-------------|-----------------|-------------------|
| Order size pattern detection | Need data to confirm if coordinated pumps are a problem | After 50 trades: if >50% losses are Telegram pumps → build detector |
| Exchange-specific stop multipliers | Need slippage data to calibrate | After 50 trades: measure actual MEXC slippage, adjust if >3% |
| Z-score anomaly detection | Solves different problem (pre-pump prediction vs post-exhaustion fade) | Skip - not relevant to our strategy |
| Regime detection | Don't know if regimes affect win rate | After 50 trades: if win rate variance by BTC trend >20 points → build filter |
| Telegram monitoring | Legal/complexity concerns | v3.0 feature if coordinated pumps become major issue |

---

## Success Metrics & Validation Gates

### Strategy Validation Gate (End of Week 5)
**Proceed to Execution IF:**
- ✅ Win rate ≥40% on historical data
- ✅ Avg R-multiple ≥3:1
- ✅ Signal quality improvements proven statistically

**Do NOT Proceed IF:**
- ❌ Win rate <35%
- ❌ No statistical improvement from enhancements
- ❌ Too many false signals

### Paper Trading Gate (End of Week 10)
**Proceed to Live Trading IF:**
- ✅ 50 paper trades completed
- ✅ Win rate ≥35%
- ✅ No critical bugs
- ✅ Risk management working correctly

### Live Trading Gate (End of Week 11)
**Scale to Full Capital IF:**
- ✅ 10 live trades at £100 level successful
- ✅ Execution quality acceptable (<2% slippage)
- ✅ P&L matches expectations (±20%)
- ✅ No major surprises or edge cases

---

## Future Enhancements (v3.0+)

**After 6+ Months of Live Trading:**

### Advanced Signal Processing
- Machine learning for pattern recognition
- Ensemble methods combining multiple models
- Adaptive thresholds based on market conditions
- Real-time regime detection

### Execution Optimization
- Smart order routing across exchanges
- Liquidity-seeking algorithms
- Dynamic slippage modeling
- Optimal execution timing

### Portfolio Management
- Multi-coin correlation analysis
- Portfolio-level risk management
- Sector exposure limits
- Capital allocation optimization

### Market Intelligence
- Telegram pump monitoring (if needed)
- Social sentiment integration
- On-chain analysis for early positioning detection
- Cross-exchange arbitrage detection

---

## Critical Risks & Mitigation

### Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| MEXC API downtime | Can't execute trades | MEDIUM | Manual backup execution, exchange failover plan |
| Database corruption | Lose trade history | LOW | Daily backups, multiple redundancy |
| WebSocket disconnection | Miss signals | MEDIUM | Automatic reconnection, heartbeat monitoring |
| Order execution failure | Lose opportunity | MEDIUM | Retry logic, fallback to market orders |

### Market Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Low liquidity gaps | Stop slippage >20% | HIGH on MEXC | ATR-adjusted stops, position sizing limits |
| Coordinated dumps | Violent squeeze through stops | MEDIUM | Order size detection (future), conviction filtering |
| Market regime change | Strategy stops working | MEDIUM | Track win rate by regime, pause if needed |
| Exchange manipulation | Front-running or wash trading | LOW | Choose reputable pairs, monitor for spoofing |

### Strategy Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Edge decay | Strategy stops working over time | HIGH (18-36 month lifespan) | Continuous monitoring, adaptation, kill switch |
| Overfitting | Backtest looks good, live fails | MEDIUM | Out-of-sample validation, live paper trading |
| Psychological pressure | Override system in losing streak | MEDIUM | Automated execution, strict rules, journaling |
| Black swan event | Massive unexpected loss | LOW | Position sizing limits, max daily loss circuit breaker |

---

## Key Learnings & Principles

### What We Know Works (From Manual Trading)
1. ✅ **3-pump exhaustion rule** - Multiple rejections signal true exhaustion
2. ✅ **CVD divergence** - Price up + selling down = distribution
3. ✅ **Funding rate positioning** - Crowded longs fuel the dump
4. ✅ **Asymmetric risk/reward** - 42% win rate profitable with 3-5:1 R
5. ✅ **Diamond hands on runners** - Holding 25% to exhaustion captures big moves

### What We're Testing (From Research)
1. 🔬 **Market cap filtering** - 95.7% of pumps are <$60M coins
2. 🔬 **Rejection quality** - Not all 3-pump patterns are equal
3. 🔬 **Support-based targets** - Volume profile levels likely hold better
4. 🔬 **Microstructure confirmation** - Order book patterns at true exhaustion
5. 🔬 **Kelly sizing** - Optimal position sizing for long-term growth

### What We're Avoiding (Lessons Learned)
1. ❌ **Over-complexity** - v1.0 failed due to feature bloat
2. ❌ **Premature optimization** - Build regime detection ONLY if data shows need
3. ❌ **Wrong problem solving** - Z-score prediction for pre-pump (not our edge)
4. ❌ **Curve-fitting** - Validate out-of-sample before live trading
5. ❌ **Ignoring execution** - v1.0 had no real execution, just infrastructure

---

## Development Standards

### Code Quality
- Comprehensive docstrings for all functions
- Type hints for parameters and returns
- Unit tests for critical logic
- Error handling with specific exceptions
- Logging at appropriate levels (DEBUG/INFO/WARNING/ERROR)

### Testing Approach
- Test each module independently before integration
- Use real market data for validation
- Compare system output to manual analysis
- Document any discrepancies or surprises

### Version Control
- Commit after each working feature
- Tag major milestones (v2.0, v2.1, etc.)
- Keep development log updated
- Document all changes and rationale

---

## Contact & Resources

**Development Team:**
- Joe (Strategy Owner) - Proven manual trader, 3X returns
- Grim (System Architect) - Institutional quant systems background

**Key Research Papers:**
- [2412.18848] Machine Learning-Based Detection of Pump-and-Dump Schemes in Real-Time
- [2405.14767] FinRobot: An Open-Source AI Agent Platform for Financial Applications using Large Language Models

**Exchange:**
- Primary: MEXC Global (UK-accessible)
- Backup: None currently (Binance/Bybit UK-restricted)

**Capital Allocation:**
- Phase 1: £100 (proof of concept)
- Phase 2: £3-4K bi-monthly
- Target: Scale to full allocation after proven success

---

## Next Actions

### Immediate (This Week - Week 4 Completion)
1. ⚪ Build market_cap_filter.py (1 hour)
2. ⚪ Enhance rejection_pattern.py with quality scoring (2 hours)
3. ⚪ Enhance signal_generator.py with support detection (3 hours)
4. ⚪ Build simple_executor.py for validation (2 hours)

### Short-Term (Next Week - Week 5)
5. ⚪ Enhance cvd_analyzer.py with microstructure analysis (4 hours)
6. ⚪ Collect 20-50 historical pump examples (1 day)
7. ⚪ Run validation against historical data (1 day)
8. ⚪ Analyze results and adjust thresholds

### Medium-Term (Weeks 6-8)
9. ⚪ Build risk management modules (Kelly sizing, stops)
10. ⚪ Build execution engine (order executor, position manager)
11. ⚪ Build monitoring and alerting systems
12. ⚪ Integration testing of full system

### Long-Term (Weeks 9-12)
13. ⚪ Paper trading (50 trades)
14. ⚪ Live trading at £100 level (10 trades)
15. ⚪ Scale to £3-4K if successful
16. ⚪ Performance analysis and optimization

---

**Last Updated:** November 15, 2025  
**Next Milestone:** Complete Week 4 enhancements by November 22, 2025  
**Target Live Date:** February 1, 2026 (Week 12)  
**First Live Trade Target:** January 18, 2026 (Week 11)  

---

*"Build institutional-grade systems, but start with proven edges. Automate what works manually, validate ruthlessly, and never risk capital on unproven strategies."* - Grim

---
