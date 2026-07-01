# Quantitative Trading System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a complete Python quantitative trading system with data fetching, dual strategy engine, backtesting, risk management, and CLI — targeting >=20% annualized return.

**Architecture:** Modular Python package: data layer (fetch + process), strategy layer (6 strategies across 2 families), backtest layer (engine + analysis + reporting), risk layer (position sizing + stops), CLI entry point.

**Tech Stack:** Python 3.9+, pandas, numpy, matplotlib, yfinance, akshare, pytest, argparse

## Global Constraints

- Python 3.9+ with full type annotations
- PEP 8, logging (not print), complete docstrings
- No silent failures — every exception must be logged
- All strategy params configurable in config.py
- Default: US equities via yfinance, $1M capital, 2023-01-01 to 2025-01-01
- Commission 0.03%, slippage 0.01%, risk-free rate 2%

---

### Task 1: Project Foundation — requirements.txt, config.py, __init__.py files

**Files to create:**
- `/work/quant-trading-system/requirements.txt`
- `/work/quant-trading-system/config.py`
- `/work/quant-trading-system/data/__init__.py`
- `/work/quant-trading-system/strategy/__init__.py`
- `/work/quant-trading-system/backtest/__init__.py`
- `/work/quant-trading-system/risk/__init__.py`
- `/work/quant-trading-system/tests/__init__.py`

**Produced interfaces:**
- `config.py` exports: SYMBOLS, MARKET, START_DATE, END_DATE, INITIAL_CAPITAL, COMMISSION, SLIPPAGE, RISK_FREE_RATE, MAX_POSITION_SIZE, MAX_DRAWDOWN_LIMIT, RISK_PER_TRADE, STRATEGY_PARAMS dict, BENCHMARK_SYMBOL
- All `__init__.py` files empty or with minimal imports

**Steps:**

- [ ] Step 1: Create requirements.txt with exact pinned versions:
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
yfinance>=0.2.30
akshare>=1.12.0
pytest>=7.4.0
scipy>=1.10.0
```

- [ ] Step 2: Create config.py with all configuration constants:
```python
"""Global configuration for the quantitative trading system."""
from typing import List, Dict, Any
from datetime import date

# Market selection: "us" for yfinance US equities, "a" for akshare A-shares
MARKET: str = "us"

# Default stock universe (US equities)
SYMBOLS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]

# A-share symbols (used when MARKET="a")
A_SHARE_SYMBOLS: List[str] = ["000001", "600519", "000858", "600036", "601318"]

# Date range for backtesting
START_DATE: date = date(2023, 1, 1)
END_DATE: date = date(2025, 1, 1)

# Data settings
DEFAULT_INTERVAL: str = "1d"  # 1d, 1wk, 1mo
CACHE_DIR: str = "data/cache"

# Backtest settings
INITIAL_CAPITAL: float = 1_000_000.0
COMMISSION: float = 0.0003  # 0.03%
SLIPPAGE: float = 0.0001    # 0.01%

# Performance metrics
RISK_FREE_RATE: float = 0.02  # 2%
BENCHMARK_SYMBOL: str = "^GSPC"  # S&P 500

# Risk management
MAX_POSITION_SIZE: float = 0.20   # 20% max in single stock
MAX_DRAWDOWN_LIMIT: float = 0.15  # 15% portfolio drawdown limit
RISK_PER_TRADE: float = 0.02      # 2% risk per trade
ATR_PERIOD: int = 14
ATR_STOP_MULTIPLIER: float = 2.0

# Strategy parameters
STRATEGY_PARAMS: Dict[str, Dict[str, Any]] = {
    "DualMACrossover": {"fast_period": 20, "slow_period": 50},
    "MACDStrategy": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "TurtleTrading": {"donchian_entry": 20, "donchian_exit": 10, "atr_period": 20},
    "BollingerBands": {"period": 20, "num_std": 2.0},
    "RSIStrategy": {"period": 14, "oversold": 30, "overbought": 70},
    "PairTrading": {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5},
}

# Logging
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

- [ ] Step 3: Create all empty __init__.py files
- [ ] Step 4: Verify Python can import the package:
```bash
cd /work/quant-trading-system && python3 -c "import config; print(f'Market: {config.MARKET}, Capital: ${config.INITIAL_CAPITAL:,.0f}')"
```

---

### Task 2: Data Fetcher — data/fetcher.py

**Files:**
- Create: `/work/quant-trading-system/data/fetcher.py`
- Test: `/work/quant-trading-system/tests/test_data.py` (first tests)

**Interfaces produced:**
- `class DataFetcher`: `__init__(market: str, cache_dir: str)`, `get_data(symbols, start_date, end_date, interval) -> Dict[str, pd.DataFrame]`, `_fetch_yfinance(symbol, start, end, interval) -> pd.DataFrame`, `_fetch_akshare(symbol, start, end, interval) -> pd.DataFrame`, `_cache_path(symbol, interval) -> str`, `_load_cache(symbol, interval) -> Optional[pd.DataFrame]`, `_save_cache(symbol, df)`, `_fetch_with_retry(fetch_fn, symbol, max_retries=3) -> pd.DataFrame`

**Steps:**

- [ ] Step 1: Write failing test in tests/test_data.py:
```python
import pytest
import pandas as pd
from datetime import date
from data.fetcher import DataFetcher

def test_fetcher_initialization():
    fetcher = DataFetcher(market="us", cache_dir="data/cache")
    assert fetcher.market == "us"
    assert fetcher.cache_dir == "data/cache"

def test_get_data_returns_dict():
    fetcher = DataFetcher(market="us", cache_dir="data/cache")
    # This will attempt real download; mock in integration
    assert hasattr(fetcher, 'get_data')
```

- [ ] Step 2: Run test to verify failure: `pytest tests/test_data.py -v`

- [ ] Step 3: Implement data/fetcher.py with full fetcher class:
  - Use yfinance.download() for US stocks, akshare for A-shares
  - Implement CSV caching with symbol-interval-date keyed filenames
  - Retry decorator with exponential backoff (1s, 2s, 4s)
  - Standardize column names to: open, high, low, close, volume
  - Log all fetch operations

- [ ] Step 4: Run tests: `pytest tests/test_data.py -v` (expected: pass or skip on network)

- [ ] Step 5: Commit

---

### Task 3: Data Processor — data/processor.py

**Files:**
- Create: `/work/quant-trading-system/data/processor.py`
- Test: append to `/work/quant-trading-system/tests/test_data.py`

**Interfaces produced:**
- `class DataProcessor`: `add_all_indicators(df: pd.DataFrame) -> pd.DataFrame`, `clean_data(df: pd.DataFrame) -> pd.DataFrame`
- Individual indicator methods: `_add_ma(df, periods)`, `_add_ema(df, fast, slow)`, `_add_macd(df, fast, slow, signal)`, `_add_rsi(df, period)`, `_add_bollinger(df, period, num_std)`, `_add_atr(df, period)`, `_add_donchian(df, period)`, `_add_vwap(df)`, `_add_returns(df)`

**Steps:**

- [ ] Step 1: Write test for indicator calculations:
```python
def test_processor_ma():
    from data.processor import DataProcessor
    df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0] * 2
    })
    # Pad to 22 rows for MA20
    processor = DataProcessor()
    result = processor.add_all_indicators(df)
    assert 'ma_20' in result.columns
    assert not result['ma_20'].iloc[-1] is None  # or check value
```

- [ ] Step 2: Run test to verify failure
- [ ] Step 3: Implement data/processor.py with all indicators:
  - MA(5,10,20,60,120), EMA(12,26)
  - MACD with histogram, RSI(14)
  - Bollinger Bands (upper, middle, lower)
  - ATR(14) using Wilder's method
  - Donchian Channel (20-day high/low)
  - VWAP
  - Daily returns and log returns
  - Forward-fill NaN, flag outliers (>5 sigma)

- [ ] Step 4: Run tests: `pytest tests/test_data.py -v`
- [ ] Step 5: Commit

---

### Task 4: Strategy Base + Trend Following — strategy/base.py + strategy/trend_following.py

**Files:**
- Create: `/work/quant-trading-system/strategy/base.py`
- Create: `/work/quant-trading-system/strategy/trend_following.py`
- Test: `/work/quant-trading-system/tests/test_strategy.py`

**Interfaces produced:**
- `class BaseStrategy(ABC)`: `name: str`, `params: Dict`, `generate_signals(data: pd.DataFrame) -> pd.DataFrame`
- `class DualMACrossover(BaseStrategy)`: signal column: 1=buy, -1=sell, 0=hold
- `class MACDStrategy(BaseStrategy)`: same signal format
- `class TurtleTrading(BaseStrategy)`: same signal format with ATR stops

**Steps:**

- [ ] Step 1: Write strategy tests with synthetic data:
```python
import pandas as pd
import numpy as np
from strategy.base import BaseStrategy
from strategy.trend_following import DualMACrossover, MACDStrategy, TurtleTrading

def test_dual_ma_crossover_signal_shape():
    # Create synthetic price data with a clear crossover
    n = 200
    close = np.concatenate([np.linspace(100, 150, 100), np.linspace(150, 100, 100)])
    df = pd.DataFrame({'close': close})
    strategy = DualMACrossover()
    signals = strategy.generate_signals(df)
    assert 'signal' in signals.columns
    assert len(signals) == n
    assert set(signals['signal'].unique()).issubset({-1, 0, 1})

def test_macd_strategy_signal_values():
    n = 200
    close = np.concatenate([np.linspace(100, 150, 100), np.linspace(150, 100, 100)])
    df = pd.DataFrame({'close': close})
    strategy = MACDStrategy()
    signals = strategy.generate_signals(df)
    assert 'signal' in signals.columns

def test_turtle_trading_signal_values():
    n = 200
    close = np.concatenate([np.linspace(100, 180, 100), np.linspace(180, 80, 100)])
    df = pd.DataFrame({'close': close, 'high': close * 1.02, 'low': close * 0.98})
    strategy = TurtleTrading()
    signals = strategy.generate_signals(df)
    assert 'signal' in signals.columns
```

- [ ] Step 2: Run tests to verify failure
- [ ] Step 3: Implement strategy/base.py:
  - ABC with abstract `generate_signals(data) -> pd.DataFrame`
  - `__init__` takes optional params override dict
  - `_validate_data(data)` helper checking required columns

- [ ] Step 4: Implement strategy/trend_following.py:
  - DualMACrossover: compute fast/slow MA, signal = 1 when fast > slow (and was <=), -1 when fast < slow (and was >=)
  - MACDStrategy: compute MACD line and signal line, cross above = buy, cross below = sell
  - TurtleTrading: entry on 20-day Donchian breakout, exit on 10-day Donchian low, ATR trailing stop

- [ ] Step 5: Run tests: `pytest tests/test_strategy.py -v`
- [ ] Step 6: Commit

---

### Task 5: Mean Reversion Strategies — strategy/mean_reversion.py

**Files:**
- Create: `/work/quant-trading-system/strategy/mean_reversion.py`
- Test: append to `/work/quant-trading-system/tests/test_strategy.py`

**Interfaces produced:**
- `class BollingerBands(BaseStrategy)`: buy at lower band, sell at middle
- `class RSIStrategy(BaseStrategy)`: buy RSI<30, sell RSI>70
- `class PairTrading(BaseStrategy)`: z-score based pair trading

**Steps:**

- [ ] Step 1: Write tests for mean reversion strategies:
```python
from strategy.mean_reversion import BollingerBands, RSIStrategy, PairTrading

def test_bollinger_strategy():
    n = 200
    close = np.concatenate([np.full(50, 100.0), np.linspace(100, 80, 50), np.linspace(80, 120, 50), np.full(50, 100.0)])
    df = pd.DataFrame({'close': close})
    strategy = BollingerBands()
    signals = strategy.generate_signals(df)
    assert 'signal' in signals.columns
    # Should generate buy signal when price touches lower band
    assert 1 in signals['signal'].values

def test_rsi_strategy():
    n = 200
    close = np.concatenate([np.full(50, 100.0), np.linspace(100, 80, 50), np.linspace(80, 120, 50), np.full(50, 100.0)])
    df = pd.DataFrame({'close': close})
    strategy = RSIStrategy()
    signals = strategy.generate_signals(df)
    assert 'signal' in signals.columns

def test_pair_trading_strategy():
    n = 200
    s1 = np.cumsum(np.random.randn(n) * 0.5) + 100
    s2 = s1 + np.random.randn(n) * 0.3  # correlated
    df = pd.DataFrame({'close': s1})
    df2 = pd.DataFrame({'close': s2})
    strategy = PairTrading()
    signals = strategy.generate_signals(df, df2)
    assert 'signal' in signals.columns
```

- [ ] Step 2: Run tests to verify failure
- [ ] Step 3: Implement strategy/mean_reversion.py:
  - BollingerBands: compute BB(period, num_std), buy when close <= lower_band, sell when close >= middle_band
  - RSIStrategy: compute RSI(14), buy at oversold (<30), sell at overbought (>70)
  - PairTrading: compute spread z-score, enter long-short at |z|>2, exit at z->0. The second stock is the hedge pair

- [ ] Step 4: Run tests: `pytest tests/test_strategy.py -v`
- [ ] Step 5: Commit

---

### Task 6: Risk Manager — risk/manager.py

**Files:**
- Create: `/work/quant-trading-system/risk/manager.py`
- Create: `/work/quant-trading-system/risk/__init__.py`
- Test: `/work/quant-trading-system/tests/test_risk.py` (new file)

**Interfaces produced:**
- `class RiskManager`: `calculate_position_size(capital, price, atr, strategy_signal) -> int`, `calculate_atr_stop(entry_price, atr, direction) -> float`, `calculate_kelly_fraction(win_rate, avg_win, avg_loss) -> float`, `check_portfolio_limits(positions, total_capital) -> bool`, `apply_risk_limits(trades: List[Dict], capital: float) -> List[Dict]`

**Steps:**

- [ ] Step 1: Write risk tests:
```python
import pytest
from risk.manager import RiskManager

def test_position_size_calculation():
    rm = RiskManager()
    shares = rm.calculate_position_size(capital=1_000_000, price=100.0, atr=2.0)
    # Max 2% risk: $20,000 risk. ATR stop at 2*ATR = $4. So shares = 20000/4 = 5000
    # But max 20% position: $200,000 / $100 = 2000 shares. So 2000.
    assert shares == 2000

def test_atr_stop_long():
    rm = RiskManager()
    stop = rm.calculate_atr_stop(entry_price=100.0, atr=2.0, direction=1)
    assert stop == 96.0  # 100 - 2*2.0

def test_kelly_fraction():
    rm = RiskManager()
    f = rm.calculate_kelly_fraction(win_rate=0.55, avg_win=0.05, avg_loss=0.03)
    # Kelly: (p*b - q) / b = (0.55*1.667 - 0.45)/1.667 = 0.28
    assert 0.2 < f < 0.35

def test_portfolio_limits():
    rm = RiskManager()
    # Max 20% per position, 15% drawdown limit
    assert rm.check_portfolio_limits(
        positions={"AAPL": 150_000}, total_capital=1_000_000
    ) is True
    assert rm.check_portfolio_limits(
        positions={"AAPL": 250_000}, total_capital=1_000_000
    ) is False  # exceeds 20%
```

- [ ] Step 2: Run tests to verify failure
- [ ] Step 3: Implement risk/manager.py:
  - Fixed-fraction: shares = (capital * risk_per_trade) / (atr * atr_multiplier)
  - Cap at MAX_POSITION_SIZE of capital
  - ATR stop: entry - atr*multiplier for long, entry + atr*multiplier for short
  - Kelly: f = (p*b - q) / b where b = avg_win/avg_loss
  - Portfolio drawdown: track peak equity, liquidate if drawdown > 15%
  - Single position max 20%

- [ ] Step 4: Run tests: `pytest tests/test_risk.py -v`
- [ ] Step 5: Commit

---

### Task 7: Backtest Engine — backtest/engine.py

**Files:**
- Create: `/work/quant-trading-system/backtest/engine.py`
- Test: `/work/quant-trading-system/tests/test_backtest.py`

**Interfaces produced:**
- `class BacktestResult`: dataclass with `equity_curve`, `trades`, `final_value`, `total_return`, `metrics`
- `class BacktestEngine`: `run(signals, prices, initial_capital) -> BacktestResult`, `_execute_trade(...)`, `_calculate_costs(price, shares) -> float`

**Steps:**

- [ ] Step 1: Write backtest test:
```python
import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine, BacktestResult

def test_backtest_buy_and_hold():
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    prices = pd.DataFrame({
        'close': np.linspace(100, 150, n)
    }, index=dates)
    signals = pd.DataFrame({
        'signal': [0] * 50 + [1] * 50  # Buy at day 51
    }, index=dates)
    
    engine = BacktestEngine(initial_capital=1_000_000)
    result = engine.run(signals=signals, prices=prices)
    
    assert isinstance(result, BacktestResult)
    assert result.final_value > 1_000_000  # Made money
    assert len(result.trades) > 0

def test_empty_signals():
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    prices = pd.DataFrame({'close': np.ones(n) * 100}, index=dates)
    signals = pd.DataFrame({'signal': np.zeros(n)}, index=dates)
    
    engine = BacktestEngine(initial_capital=1_000_000)
    result = engine.run(signals=signals, prices=prices)
    assert result.final_value == 1_000_000  # No trades, no change
```

- [ ] Step 2: Run test to verify failure
- [ ] Step 3: Implement backtest/engine.py:
  - Vectorized loop through time steps
  - Track: cash, holdings (shares per symbol), equity curve
  - On buy signal (1): allocate capital using RiskManager, deduct costs
  - On sell signal (-1): liquidate position, add proceeds minus costs
  - Commission 0.03%, slippage 0.01% applied to each trade
  - Record each trade: timestamp, symbol, direction, price, shares, cost, pnl
  - Return BacktestResult with full equity curve and trade log

- [ ] Step 4: Run tests: `pytest tests/test_backtest.py -v`
- [ ] Step 5: Commit

---

### Task 8: Performance Analyzer — backtest/analyzer.py

**Files:**
- Create: `/work/quant-trading-system/backtest/analyzer.py`
- Test: append to `/work/quant-trading-system/tests/test_backtest.py`

**Interfaces produced:**
- `class PerformanceAnalyzer`: `analyze(result: BacktestResult, benchmark_returns: pd.Series) -> Dict[str, float]`
- Metrics: total_return, annualized_return, sharpe_ratio, max_drawdown, drawdown_duration, win_rate, profit_loss_ratio, calmar_ratio, alpha, beta, monthly_returns, yearly_returns

**Steps:**

- [ ] Step 1: Write analyzer test:
```python
from backtest.analyzer import PerformanceAnalyzer

def test_analyzer_metrics():
    # Create a simple equity curve: steady growth
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    equity = 1_000_000 * (1 + np.linspace(0, 0.20, 252))  # 20% growth
    equity_curve = pd.Series(equity, index=dates)
    
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(equity_curve=equity_curve, benchmark_returns=None)
    
    assert 'total_return' in metrics
    assert 'annualized_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert metrics['total_return'] == pytest.approx(0.20, rel=0.05)

def test_max_drawdown():
    analyzer = PerformanceAnalyzer()
    # Equity with a 10% drawdown in middle
    equity = pd.Series([100, 110, 120, 115, 110, 105, 99, 105, 115, 125])
    dd = analyzer._calculate_max_drawdown(equity)
    assert dd == pytest.approx(0.175, rel=0.01)  # (120-99)/120 = 0.175
```

- [ ] Step 2: Run test to verify failure
- [ ] Step 3: Implement backtest/analyzer.py:
  - Daily returns from equity curve
  - Annualized: (1+total_return)^(252/trading_days) - 1
  - Sharpe: (mean_daily_return - rf_daily) / std_daily * sqrt(252)
  - Max drawdown: peak-to-trough percentage
  - Drawdown duration: days from peak to recovery
  - Win rate: winning trades / total trades
  - Profit factor: gross profit / gross loss
  - Calmar: annualized_return / max_drawdown
  - Alpha/Beta: CAPM regression vs benchmark
  - Monthly/yearly return aggregation

- [ ] Step 4: Run tests: `pytest tests/test_backtest.py -v`
- [ ] Step 5: Commit

---

### Task 9: Report Generator — backtest/reporter.py

**Files:**
- Create: `/work/quant-trading-system/backtest/reporter.py`

**Interfaces produced:**
- `class ReportGenerator`: `generate(result, analyzer_metrics, output_dir) -> str` (path to report), `_plot_equity_curve(...)`, `_plot_drawdown(...)`, `_plot_monthly_heatmap(...)`, `_plot_trade_signals(...)`, `_generate_markdown_report(...)`

**Steps:**

- [ ] Step 1: Implement backtest/reporter.py:
  - Use matplotlib with 'Agg' backend
  - Four charts: equity curve (with benchmark overlay), drawdown curve, monthly returns heatmap, trade signal markers on price chart
  - Save charts as PNG to `reports/` directory
  - Generate markdown report with all metrics table, strategy summary, risk metrics
  - Return path to markdown report file

- [ ] Step 2: Verify with quick manual test:
```bash
cd /work/quant-trading-system && python3 -c "
from backtest.reporter import ReportGenerator
rg = ReportGenerator()
print('ReportGenerator initialized successfully')
"
```
- [ ] Step 3: Commit

---

### Task 10: Main CLI — main.py

**Files:**
- Create: `/work/quant-trading-system/main.py`

**Interfaces produced:**
- CLI with argparse: `--mode {fetch, backtest, optimize, report}`, `--strategy`, `--symbols`, `--start`, `--end`, `--capital`

**Steps:**

- [ ] Step 1: Implement main.py with full CLI:
  - fetch mode: calls DataFetcher.get_data() for all config symbols
  - backtest mode: runs specified strategy through backtest engine and generates report
  - optimize mode: grid search over strategy parameters, finds best by Sharpe
  - report mode: regenerates report from saved results
  - All modes use logging, handle exceptions gracefully

- [ ] Step 2: Verify CLI help:
```bash
cd /work/quant-trading-system && python3 main.py --help
```
- [ ] Step 3: Commit

---

### Task 11: Integration — Full System Test + README

**Files:**
- Create: `/work/quant-trading-system/README.md`
- Modify: `/work/quant-trading-system/tests/test_backtest.py` (add integration test)

**Steps:**

- [ ] Step 1: Write integration test that runs data fetch → process → strategy → backtest → analyze pipeline
- [ ] Step 2: Write comprehensive README.md:
  - Project introduction and architecture diagram
  - Installation: `pip install -r requirements.txt`
  - Quick start: fetch data, run backtest, view report
  - Module descriptions and strategy principles
  - Configuration guide
  - Risk warning
- [ ] Step 3: Run full test suite: `cd /work/quant-trading-system && python3 -m pytest tests/ -v`
- [ ] Step 4: Run a complete backtest: `python3 main.py --mode backtest --strategy all`
- [ ] Step 5: Commit final changes

---

### Task 12: Git Init + GitHub Push

**Steps:**

- [ ] Step 1: Initialize git repo:
```bash
cd /work/quant-trading-system && git init && git add -A && git commit -m "Initial commit: 量化交易系统 - 趋势跟踪+均值回归+完整回测引擎"
```
- [ ] Step 2: Create GitHub repo and push:
```bash
cd /work/quant-trading-system && gh repo create quant-trading-system --public --source=. --remote=origin --push --description "Python股票量化交易系统 | 年化收益20% | 趋势跟踪+均值回归策略 | 完整回测引擎"
```
- [ ] Step 3: Output the repository URL

