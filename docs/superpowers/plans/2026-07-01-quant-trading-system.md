# Quantitative Trading System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Python quantitative trading system with data fetching, dual strategy engine, backtesting with metrics/visualization, simulated trade execution, and CLI interface — targeting 20% annualized return.

**Architecture:** Modular Python package with 4 core modules (data, strategy, backtest, executor) plus CLI entry point. Strategies inherit from BaseStrategy. Backtest engine is strategy-agnostic via signal interface. Executor simulates broker operations independently.

**Tech Stack:** Python 3.13, pandas, numpy, matplotlib, yfinance, pytest, argparse

## Global Constraints

- Python 3.13 compatibility required
- Default test symbol: AAPL
- Data source: yfinance primary, fallback to simulated data
- Default initial capital: $100,000
- Transaction costs: 0.1% commission + 0.05% slippage
- Risk-free rate for Sharpe ratio: 2%
- Target annualized return: >=20%
- All modules must have pytest unit tests
- CLI uses argparse with subcommands `backtest` and `simulate`
- Data columns: Open, High, Low, Close, Volume (capitalized)
- Signal values: 1=buy, -1=sell, 0=hold
- Strategies encode signals in DataFrame columns: Signal, Price
- OHLCV data is indexed by Date (DatetimeIndex)

## File Structure

```
/work/quant-trading-system/
├── config.py                  # Global constants
├── requirements.txt           # Python dependencies
├── setup.py                   # Package install config
├── main.py                    # CLI entry point (argparse)
├── README.md                  # Usage documentation
├── data/
│   ├── __init__.py
│   ├── fetcher.py             # DataFetcher: yfinance + mock fallback
│   └── storage.py             # DataStorage: CSV cache
├── strategy/
│   ├── __init__.py
│   ├── base.py                # BaseStrategy ABC
│   ├── ma_crossover.py        # MACrossoverStrategy
│   └── momentum_breakout.py   # MomentumBreakoutStrategy
├── backtest/
│   ├── __init__.py
│   ├── engine.py              # BacktestEngine
│   ├── analyzer.py            # PerformanceAnalyzer
│   └── reporter.py            # ReportGenerator (matplotlib + HTML)
├── executor/
│   ├── __init__.py
│   ├── broker.py              # Broker: order management
│   ├── portfolio.py           # Portfolio: positions, cash, P&L
│   └── logger.py              # TradeLogger: file-based logging
└── tests/
    ├── __init__.py
    ├── test_strategy.py
    ├── test_backtest.py
    └── test_executor.py
```

## Data Contracts Between Subagents

### Contract 1: OHLCV DataFrame
- Index: `pd.DatetimeIndex` with name `Date`
- Columns: `['Open', 'High', 'Low', 'Close', 'Volume']`
- All float except Volume (int)

### Contract 2: Signal DataFrame
- Index: same as input OHLCV DataFrame
- Columns: `['Signal', 'Price']`
  - `Signal`: int, values {1=buy, -1=sell, 0=hold}
  - `Price`: float, the close price at signal time

### Contract 3: Backtest Result dict
```python
{
    'equity_curve': pd.Series,     # Portfolio value over time
    'trades': list[dict],          # Completed trades
    'final_equity': float,         # Final portfolio value
}
# Each trade dict:
{
    'entry_date': Timestamp, 'exit_date': Timestamp,
    'entry_price': float, 'exit_price': float,
    'shares': int, 'profit_loss': float,
    'profit_loss_pct': float, 'type': str,
}
```

### Contract 4: Performance Metrics dict
```python
{
    'total_return': float (%), 'annualized_return': float (%),
    'max_drawdown': float (%), 'sharpe_ratio': float,
    'win_rate': float (%), 'profit_loss_ratio': float,
    'total_trades': int, 'final_equity': float,
}
```

---

## Subagent 1: Infrastructure — config.py + requirements.txt + setup.py + data/ module

### Task 1.1: Create requirements.txt and setup.py

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`

**Interfaces:**
- Produces: dependency declarations and package metadata

- [ ] **Step 1: Write requirements.txt**

```text
yfinance>=0.2.38
pandas>=2.2.0
numpy>=2.0.0
matplotlib>=3.9.0
pytest>=9.0.0
```

- [ ] **Step 2: Write setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="quant-trading-system",
    version="0.1.0",
    description="A quantitative stock trading system with backtesting and simulation",
    author="Quant Trader",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.38",
        "pandas>=2.2.0",
        "numpy>=2.0.0",
        "matplotlib>=3.9.0",
    ],
    entry_points={
        "console_scripts": [
            "quant-trading=main:main",
        ],
    },
)
```

- [ ] **Step 3: Install dependencies**

```bash
pip install yfinance pandas matplotlib
```

### Task 1.2: Create config.py

**Files:**
- Create: `config.py`

**Interfaces:**
- Produces: all global constants used by every module

- [ ] **Step 1: Write config.py**

```python
"""Global configuration constants for the quant trading system."""

# Default settings
DEFAULT_SYMBOL = "AAPL"
DEFAULT_CAPITAL = 100_000.0
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Transaction costs
COMMISSION_RATE = 0.001   # 0.1%
SLIPPAGE_RATE = 0.0005    # 0.05%

# Strategy defaults
MA_SHORT_WINDOW = 10
MA_LONG_WINDOW = 30
MOMENTUM_LOOKBACK = 20
MOMENTUM_EXIT_LOOKBACK = 10
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOP_LOSS_PCT = 0.05       # 5%
TAKE_PROFIT_PCT = 0.15     # 15%
POSITION_SIZE_PCT = 0.80   # 80% of capital per trade

# Risk-free rate for Sharpe ratio
RISK_FREE_RATE = 0.02      # 2%

# Output directories
REPORT_DIR = "reports"
LOG_DIR = "logs"
DATA_DIR = "data_cache"

# Logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
```

### Task 1.3: Create data/__init__.py

**Files:**
- Create: `data/__init__.py`

- [ ] **Step 1: Write data/__init__.py**

```python
from .fetcher import DataFetcher
from .storage import DataStorage

__all__ = ["DataFetcher", "DataStorage"]
```

### Task 1.4: Create data/fetcher.py

**Files:**
- Create: `data/fetcher.py`

**Interfaces:**
- Produces: `DataFetcher.fetch(symbol, start, end) -> pd.DataFrame` with columns Open/High/Low/Close/Volume, index Date

- [ ] **Step 1: Write data/fetcher.py**

```python
"""Data fetcher using yfinance with simulated data fallback."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches stock historical data from yfinance, falls back to simulated data."""

    def __init__(self):
        self._yfinance_available = None

    def _check_yfinance(self) -> bool:
        """Check if yfinance is available and working."""
        if self._yfinance_available is not None:
            return self._yfinance_available
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="5d")
            if data is not None and not data.empty:
                self._yfinance_available = True
                return True
        except Exception:
            pass
        self._yfinance_available = False
        return False

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical daily data for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g. 'AAPL')
            start: Start date string 'YYYY-MM-DD'
            end: End date string 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, indexed by Date
        """
        if self._check_yfinance():
            try:
                return self._fetch_yfinance(symbol, start, end)
            except Exception as e:
                logger.warning(f"yfinance fetch failed: {e}, falling back to simulated data")

        logger.info(f"Using simulated data for {symbol}")
        return self._generate_mock_data(symbol, start, end)

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch data from yfinance."""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data.index.name = 'Date'
        return data

    def _generate_mock_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Generate realistic simulated stock data using geometric Brownian motion."""
        dates = pd.date_range(start=start, end=end, freq='B')
        n = len(dates)
        if n == 0:
            raise ValueError(f"No business days between {start} and {end}")

        seed = sum(ord(c) for c in symbol)
        rng = np.random.default_rng(seed)

        initial_price = 150.0
        mu = 0.0008    # ~20% annual drift
        sigma = 0.015  # daily volatility

        daily_returns = rng.normal(mu, sigma, n)
        prices = initial_price * np.exp(np.cumsum(daily_returns))

        noise = rng.normal(0, 0.005, n)
        df = pd.DataFrame({
            'Open': prices * (1 + noise),
            'High': prices * (1 + np.abs(rng.normal(0, 0.01, n))),
            'Low': prices * (1 - np.abs(rng.normal(0, 0.01, n))),
            'Close': prices,
            'Volume': rng.integers(50_000_000, 100_000_000, n),
        }, index=dates)

        # Ensure OHLC consistency: High >= max(Open,Close), Low <= min(Open,Close)
        for i in range(n):
            o, c = df.iloc[i]['Open'], df.iloc[i]['Close']
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i]['High'], o, c)
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i]['Low'], o, c)

        df.index.name = 'Date'
        return df
```

- [ ] **Step 2: Verify DataFetcher works**

```bash
cd /work/quant-trading-system && python -c "
from data.fetcher import DataFetcher
f = DataFetcher()
df = f.fetch('AAPL', '2024-01-01', '2024-06-30')
print(f'Shape: {df.shape}, Columns: {list(df.columns)}')
print(df.head())
"
```

### Task 1.5: Create data/storage.py

**Files:**
- Create: `data/storage.py`

**Interfaces:**
- Produces: `DataStorage.save/load/exists` for CSV caching

- [ ] **Step 1: Write data/storage.py**

```python
"""Data storage for caching fetched stock data as CSV files."""

import os
import logging
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)


class DataStorage:
    """Caches stock data to local CSV files."""

    def __init__(self, cache_dir: str = DATA_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol.upper()}.csv")

    def save(self, df: pd.DataFrame, symbol: str) -> str:
        path = self.get_path(symbol)
        df.to_csv(path)
        logger.info(f"Saved {symbol} data to {path} ({len(df)} rows)")
        return path

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self.get_path(symbol)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {symbol} data from cache ({len(df)} rows)")
        return df

    def exists(self, symbol: str) -> bool:
        return os.path.exists(self.get_path(symbol))
```

- [ ] **Step 2: Verify DataStorage works**

```bash
cd /work/quant-trading-system && python -c "
from data.fetcher import DataFetcher
from data.storage import DataStorage
f = DataFetcher()
df = f.fetch('AAPL', '2024-01-01', '2024-06-30')
s = DataStorage()
s.save(df, 'AAPL')
loaded = s.load('AAPL')
print(f'Loaded shape: {loaded.shape}, Exists: {s.exists(\"AAPL\")}')
"
```

---

## Subagent 2: Strategy Module — strategy/ + tests/test_strategy.py

### Task 2.1: Create strategy/__init__.py

**Files:**
- Create: `strategy/__init__.py`

```python
from .base import BaseStrategy
from .ma_crossover import MACrossoverStrategy
from .momentum_breakout import MomentumBreakoutStrategy

__all__ = ["BaseStrategy", "MACrossoverStrategy", "MomentumBreakoutStrategy"]
```

### Task 2.2: Create strategy/base.py

**Files:**
- Create: `strategy/base.py`

**Interfaces:**
- Produces: `BaseStrategy` ABC with abstract `generate_signals(data) -> pd.DataFrame`
  - Returns DataFrame with columns `Signal` (1/-1/0) and `Price`

```python
"""Base strategy abstract class."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, **params):
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from historical data.

        Args:
            data: DataFrame with columns Open, High, Low, Close, Volume,
                  indexed by Date.

        Returns:
            DataFrame with columns:
                - Signal: 1 (buy), -1 (sell), 0 (hold)
                - Price: The price at which to execute (Close price)
            Indexed by Date (same as input).
        """
        pass
```

### Task 2.3: Create strategy/ma_crossover.py + tests/test_strategy.py

**Files:**
- Create: `strategy/ma_crossover.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_strategy.py`

**Interfaces:**
- Produces: `MACrossoverStrategy(short_window=10, long_window=30).generate_signals(data) -> pd.DataFrame`

- [ ] **Step 1: Write tests/test_strategy.py**

```python
"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np

from strategy.ma_crossover import MACrossoverStrategy
from strategy.momentum_breakout import MomentumBreakoutStrategy


def create_test_data(n_days: int = 200, trend: float = 0.001) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
    rng = np.random.default_rng(42)
    returns = rng.normal(trend, 0.015, n_days)
    close = 100 * np.exp(np.cumsum(returns))
    data = pd.DataFrame({
        'Open': close * (1 + rng.normal(0, 0.003, n_days)),
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': rng.integers(50_000_000, 100_000_000, n_days),
    }, index=dates)
    for i in range(n_days):
        o, c = data.iloc[i]['Open'], data.iloc[i]['Close']
        data.iloc[i, data.columns.get_loc('High')] = max(data.iloc[i]['High'], o, c)
        data.iloc[i, data.columns.get_loc('Low')] = min(data.iloc[i]['Low'], o, c)
    return data


class TestMACrossoverStrategy:

    def test_initialization(self):
        strategy = MACrossoverStrategy(short_window=10, long_window=30)
        assert strategy.short_window == 10
        assert strategy.long_window == 30

    def test_default_parameters(self):
        strategy = MACrossoverStrategy()
        assert strategy.short_window == 10
        assert strategy.long_window == 30

    def test_generate_signals_shape(self):
        data = create_test_data(200)
        strategy = MACrossoverStrategy()
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)
        assert 'Signal' in signals.columns
        assert 'Price' in signals.columns

    def test_signal_values_are_valid(self):
        data = create_test_data(200)
        strategy = MACrossoverStrategy()
        signals = strategy.generate_signals(data)
        assert set(signals['Signal'].unique()).issubset({-1, 0, 1})

    def test_golden_cross_generates_buy(self):
        """When short MA crosses above long MA, should generate buy signal."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        prices = np.concatenate([
            np.linspace(100, 80, 50),
            np.linspace(80, 120, 50),
        ])
        data = pd.DataFrame({
            'Open': prices * 1.001, 'High': prices * 1.01,
            'Low': prices * 0.99, 'Close': prices,
            'Volume': np.full(100, 50_000_000),
        }, index=dates)
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        buy_signals = signals[signals['Signal'] == 1]
        assert len(buy_signals) > 0

    def test_death_cross_generates_sell(self):
        """When short MA crosses below long MA, should generate sell signal."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        prices = np.concatenate([
            np.linspace(100, 120, 50),
            np.linspace(120, 80, 50),
        ])
        data = pd.DataFrame({
            'Open': prices * 1.001, 'High': prices * 1.01,
            'Low': prices * 0.99, 'Close': prices,
            'Volume': np.full(100, 50_000_000),
        }, index=dates)
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        sell_signals = signals[signals['Signal'] == -1]
        assert len(sell_signals) > 0

    def test_no_signals_initially_during_warmup(self):
        data = create_test_data(100)
        strategy = MACrossoverStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert signals['Signal'].iloc[:29].sum() == 0


class TestMomentumBreakoutStrategy:

    def test_initialization(self):
        strategy = MomentumBreakoutStrategy(lookback=20, exit_lookback=10)
        assert strategy.lookback == 20
        assert strategy.exit_lookback == 10

    def test_default_parameters(self):
        strategy = MomentumBreakoutStrategy()
        assert strategy.lookback == 20
        assert strategy.exit_lookback == 10

    def test_generate_signals_shape(self):
        data = create_test_data(200)
        strategy = MomentumBreakoutStrategy()
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)
        assert 'Signal' in signals.columns
        assert 'Price' in signals.columns

    def test_signal_values_are_valid(self):
        data = create_test_data(200)
        strategy = MomentumBreakoutStrategy()
        signals = strategy.generate_signals(data)
        assert set(signals['Signal'].unique()).issubset({-1, 0, 1})

    def test_breakout_generates_buy(self):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='B')
        rng = np.random.default_rng(42)
        prices = np.concatenate([
            rng.uniform(90, 110, 30),
            np.linspace(110, 130, 30),
        ])
        data = pd.DataFrame({
            'Open': prices * 0.999, 'High': prices * 1.015,
            'Low': prices * 0.985, 'Close': prices,
            'Volume': np.full(60, 50_000_000),
        }, index=dates)
        strategy = MomentumBreakoutStrategy(lookback=20, exit_lookback=10)
        signals = strategy.generate_signals(data)
        buy_signals = signals[signals['Signal'] == 1]
        assert len(buy_signals) > 0

    def test_breakdown_generates_sell(self):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='B')
        rng = np.random.default_rng(42)
        prices = np.concatenate([
            rng.uniform(90, 110, 30),
            np.linspace(110, 80, 30),
        ])
        data = pd.DataFrame({
            'Open': prices * 0.999, 'High': prices * 1.015,
            'Low': prices * 0.985, 'Close': prices,
            'Volume': np.full(60, 50_000_000),
        }, index=dates)
        strategy = MomentumBreakoutStrategy(lookback=20, exit_lookback=10)
        signals = strategy.generate_signals(data)
        sell_signals = signals[signals['Signal'] == -1]
        assert len(sell_signals) > 0
```

- [ ] **Step 2: Write strategy/ma_crossover.py**

```python
"""Dual Moving Average Crossover strategy."""

import pandas as pd

from config import MA_SHORT_WINDOW, MA_LONG_WINDOW
from .base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """Dual moving average crossover strategy.

    Buy signal: short MA crosses above long MA (golden cross).
    Sell signal: short MA crosses below long MA (death cross).
    """

    def __init__(self, short_window: int = MA_SHORT_WINDOW,
                 long_window: int = MA_LONG_WINDOW):
        super().__init__(short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0
        signals['Price'] = df['Close']

        df['MA_Short'] = df['Close'].rolling(window=self.short_window).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_window).mean()

        df['Short_Above'] = df['MA_Short'] > df['MA_Long']
        df['Cross_Above'] = df['Short_Above'] & (~df['Short_Above'].shift(1).fillna(False))
        df['Cross_Below'] = (~df['Short_Above']) & (df['Short_Above'].shift(1).fillna(False))

        signals.loc[df['Cross_Above'], 'Signal'] = 1
        signals.loc[df['Cross_Below'], 'Signal'] = -1

        return signals
```

- [ ] **Step 3: Write strategy/momentum_breakout.py**

```python
"""Momentum Breakout strategy with RSI filter."""

import pandas as pd
import numpy as np

from config import MOMENTUM_LOOKBACK, MOMENTUM_EXIT_LOOKBACK, RSI_OVERBOUGHT, RSI_OVERSOLD
from .base import BaseStrategy


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum breakout strategy.

    Buy: Price breaks above N-day high (and RSI not overbought).
    Sell: Price breaks below M-day low (and RSI not oversold).
    """

    def __init__(self, lookback: int = MOMENTUM_LOOKBACK,
                 exit_lookback: int = MOMENTUM_EXIT_LOOKBACK):
        super().__init__(lookback=lookback, exit_lookback=exit_lookback)
        self.lookback = lookback
        self.exit_lookback = exit_lookback

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0
        signals['Price'] = df['Close']

        df['Rolling_High'] = df['High'].rolling(window=self.lookback).max()
        df['Rolling_Low'] = df['Low'].rolling(window=self.exit_lookback).min()
        df['RSI'] = self._compute_rsi(df['Close'])

        df['Prev_High'] = df['Rolling_High'].shift(1)
        df['Prev_Low'] = df['Rolling_Low'].shift(1)

        buy_condition = (
            (df['Close'] > df['Prev_High']) &
            (df['Close'].shift(1) <= df['Prev_High']) &
            (df['RSI'] < RSI_OVERBOUGHT)
        )

        sell_condition = (
            (df['Close'] < df['Prev_Low']) &
            (df['Close'].shift(1) >= df['Prev_Low']) &
            (df['RSI'] > RSI_OVERSOLD)
        )

        signals.loc[buy_condition, 'Signal'] = 1
        signals.loc[sell_condition, 'Signal'] = -1

        return signals
```

- [ ] **Step 4: Run strategy tests**

```bash
cd /work/quant-trading-system && python -m pytest tests/test_strategy.py -v
```

---

## Subagent 3: Backtest Module — backtest/ + tests/test_backtest.py

### Task 3.1: Create backtest/__init__.py

**Files:**
- Create: `backtest/__init__.py`

```python
from .engine import BacktestEngine
from .analyzer import PerformanceAnalyzer
from .reporter import ReportGenerator

__all__ = ["BacktestEngine", "PerformanceAnalyzer", "ReportGenerator"]
```

### Task 3.2: Create backtest/engine.py + tests/test_backtest.py

**Files:**
- Create: `backtest/engine.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: OHLCV DataFrame + Signal DataFrame
- Produces: `BacktestEngine.run(data, signals, capital) -> dict` with equity_curve, trades, final_equity

- [ ] **Step 1: Write tests/test_backtest.py**

```python
"""Tests for backtest engine and analyzer."""

import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.analyzer import PerformanceAnalyzer


def create_test_data_and_signals(n_days: int = 252):
    """Create test OHLCV data and simple alternating buy/sell signals."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_days)))

    data = pd.DataFrame({
        'Open': prices * 1.001,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': rng.integers(50_000_000, 100_000_000, n_days),
    }, index=dates)

    signals = pd.DataFrame({'Signal': 0, 'Price': prices}, index=dates)
    signals.loc[dates[50], 'Signal'] = 1
    signals.loc[dates[100], 'Signal'] = -1
    signals.loc[dates[150], 'Signal'] = 1
    signals.loc[dates[200], 'Signal'] = -1

    return data, signals


class TestBacktestEngine:

    def test_initialization(self):
        engine = BacktestEngine(initial_capital=100000)
        assert engine.initial_capital == 100000
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005

    def test_run_returns_expected_keys(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert 'equity_curve' in result
        assert 'trades' in result
        assert 'final_equity' in result

    def test_equity_curve_has_correct_length(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert len(result['equity_curve']) == len(data)

    def test_no_signals_preserves_capital(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        data = pd.DataFrame({
            'Open': np.full(100, 100.0), 'High': np.full(100, 101.0),
            'Low': np.full(100, 99.0), 'Close': np.full(100, 100.0),
            'Volume': np.full(100, 50_000_000),
        }, index=dates)
        signals = pd.DataFrame({'Signal': 0, 'Price': np.full(100, 100.0)}, index=dates)
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert result['final_equity'] == 100000

    def test_buy_then_sell_creates_trades(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert len(result['trades']) == 2

    def test_trades_have_required_fields(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        if result['trades']:
            trade = result['trades'][0]
            required = ['entry_date', 'exit_date', 'entry_price', 'exit_price',
                        'shares', 'profit_loss', 'profit_loss_pct', 'type']
            for field in required:
                assert field in trade, f"Missing field: {field}"

    def test_profitable_trend_produces_gain(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        prices = np.linspace(100, 150, 100)
        data = pd.DataFrame({
            'Open': prices, 'High': prices * 1.01, 'Low': prices * 0.99,
            'Close': prices, 'Volume': np.full(100, 50_000_000),
        }, index=dates)
        signals = pd.DataFrame({'Signal': 0, 'Price': prices}, index=dates)
        signals.loc[dates[10], 'Signal'] = 1
        signals.loc[dates[90], 'Signal'] = -1
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert result['final_equity'] > 100000

    def test_commission_and_slippage_applied(self):
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        prices = np.linspace(100, 110, 50)
        data = pd.DataFrame({
            'Open': prices, 'High': prices * 1.01, 'Low': prices * 0.99,
            'Close': prices, 'Volume': np.full(50, 50_000_000),
        }, index=dates)
        signals = pd.DataFrame({'Signal': 0, 'Price': prices}, index=dates)
        signals.loc[dates[5], 'Signal'] = 1
        signals.loc[dates[45], 'Signal'] = -1

        engine_with = BacktestEngine(commission_rate=0.001, slippage_rate=0.0005)
        result_with = engine_with.run(data, signals, 100000)

        engine_no = BacktestEngine(commission_rate=0.0, slippage_rate=0.0)
        result_no = engine_no.run(data, signals, 100000)

        assert result_with['final_equity'] < result_no['final_equity']


class TestPerformanceAnalyzer:

    def _make_result(self, equity_series, trades=None):
        return {
            'equity_curve': equity_series,
            'trades': trades or [],
            'final_equity': equity_series.iloc[-1],
        }

    def test_annualized_return(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert 18 < metrics['annualized_return'] < 22

    def test_max_drawdown(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        equity = pd.Series(
            [100000] * 50 + list(np.linspace(100000, 90000, 50)),
            index=dates
        )
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['max_drawdown'] == pytest.approx(10.0, abs=0.5)

    def test_no_drawdown_when_always_up(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        equity = pd.Series(np.linspace(100000, 150000, 100), index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['max_drawdown'] == 0.0

    def test_sharpe_ratio_positive(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['sharpe_ratio'] > 0

    def test_win_rate(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.cumsum(np.random.randn(252) * 100), index=dates)
        trades = [
            {'profit_loss': 1000, 'type': 'signal', 'entry_date': dates[0], 'exit_date': dates[10],
             'entry_price': 100, 'exit_price': 110, 'shares': 100, 'profit_loss_pct': 10.0},
            {'profit_loss': -500, 'type': 'signal', 'entry_date': dates[20], 'exit_date': dates[30],
             'entry_price': 100, 'exit_price': 95, 'shares': 100, 'profit_loss_pct': -5.0},
            {'profit_loss': 300, 'type': 'signal', 'entry_date': dates[40], 'exit_date': dates[50],
             'entry_price': 100, 'exit_price': 103, 'shares': 100, 'profit_loss_pct': 3.0},
        ]
        result = self._make_result(equity, trades)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['win_rate'] == pytest.approx(66.67, abs=0.1)

    def test_profit_loss_ratio(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        trades = [
            {'profit_loss': 1000, 'type': 'signal', 'entry_date': dates[0], 'exit_date': dates[10],
             'entry_price': 100, 'exit_price': 110, 'shares': 100, 'profit_loss_pct': 10.0},
            {'profit_loss': -200, 'type': 'signal', 'entry_date': dates[20], 'exit_date': dates[30],
             'entry_price': 100, 'exit_price': 98, 'shares': 100, 'profit_loss_pct': -2.0},
        ]
        result = self._make_result(equity, trades)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['profit_loss_ratio'] == pytest.approx(5.0, abs=0.1)

    def test_all_metrics_present(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        required = ['annualized_return', 'max_drawdown', 'sharpe_ratio',
                    'win_rate', 'profit_loss_ratio', 'total_trades',
                    'final_equity', 'total_return']
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
```

- [ ] **Step 2: Write backtest/engine.py**

```python
"""Backtest engine for simulating trades based on strategy signals."""

import logging
import pandas as pd
import numpy as np

from config import COMMISSION_RATE, SLIPPAGE_RATE, POSITION_SIZE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulates trade execution based on strategy signals."""

    def __init__(self, initial_capital: float = 100_000.0,
                 commission_rate: float = COMMISSION_RATE,
                 slippage_rate: float = SLIPPAGE_RATE):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def run(self, data: pd.DataFrame, signals: pd.DataFrame,
            capital: float | None = None) -> dict:
        if capital is None:
            capital = self.initial_capital

        cash = capital
        position = 0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity_curve = pd.Series(index=data.index, dtype=float)

        stop_loss_level = None
        take_profit_level = None
        half_sold = False

        for date, row in data.iterrows():
            price = row['Close']
            signal = signals.loc[date, 'Signal'] if date in signals.index else 0

            # Check stop-loss
            if position > 0 and stop_loss_level is not None:
                if row['Low'] <= stop_loss_level:
                    exit_price = stop_loss_level * (1 - self.slippage_rate)
                    proceeds = position * exit_price * (1 - self.commission_rate)
                    cost_basis = position * entry_price * (1 + self.commission_rate)
                    pnl = proceeds - cost_basis
                    trades.append({
                        'entry_date': entry_date, 'exit_date': date,
                        'entry_price': entry_price, 'exit_price': exit_price,
                        'shares': position, 'profit_loss': pnl,
                        'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                        'type': 'stop_loss',
                    })
                    cash += proceeds
                    position = 0
                    entry_price = 0.0
                    stop_loss_level = None
                    take_profit_level = None
                    half_sold = False

            # Check take-profit partial sell
            if position > 0 and take_profit_level is not None and not half_sold:
                if row['High'] >= take_profit_level:
                    sell_shares = position // 2
                    if sell_shares > 0:
                        exit_price = take_profit_level * (1 - self.slippage_rate)
                        proceeds = sell_shares * exit_price * (1 - self.commission_rate)
                        cost_basis = sell_shares * entry_price * (1 + self.commission_rate)
                        pnl = proceeds - cost_basis
                        trades.append({
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_price': entry_price, 'exit_price': exit_price,
                            'shares': sell_shares, 'profit_loss': pnl,
                            'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                            'type': 'take_profit_partial',
                        })
                        cash += proceeds
                        position -= sell_shares
                        half_sold = True

            # Process buy signal
            if signal == 1 and position == 0:
                buy_price = price * (1 + self.slippage_rate)
                max_shares = int((cash * POSITION_SIZE_PCT) / buy_price)
                if max_shares > 0:
                    cost = max_shares * buy_price * (1 + self.commission_rate)
                    if cost <= cash:
                        cash -= cost
                        position = max_shares
                        entry_price = buy_price
                        entry_date = date
                        stop_loss_level = entry_price * (1 - STOP_LOSS_PCT)
                        take_profit_level = entry_price * (1 + TAKE_PROFIT_PCT)
                        half_sold = False

            # Process sell signal
            elif signal == -1 and position > 0:
                sell_price = price * (1 - self.slippage_rate)
                proceeds = position * sell_price * (1 - self.commission_rate)
                cost_basis = position * entry_price * (1 + self.commission_rate)
                pnl = proceeds - cost_basis
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'entry_price': entry_price, 'exit_price': sell_price,
                    'shares': position, 'profit_loss': pnl,
                    'profit_loss_pct': (sell_price / entry_price - 1) * 100,
                    'type': 'signal',
                })
                cash += proceeds
                position = 0
                entry_price = 0.0
                stop_loss_level = None
                take_profit_level = None
                half_sold = False

            equity_curve.loc[date] = cash + position * price

        # Close remaining position at last price
        if position > 0:
            last_price = data['Close'].iloc[-1]
            proceeds = position * last_price * (1 - self.commission_rate)
            cost_basis = position * entry_price * (1 + self.commission_rate)
            pnl = proceeds - cost_basis
            trades.append({
                'entry_date': entry_date, 'exit_date': data.index[-1],
                'entry_price': entry_price, 'exit_price': last_price,
                'shares': position, 'profit_loss': pnl,
                'profit_loss_pct': (last_price / entry_price - 1) * 100,
                'type': 'close_out',
            })

        equity_curve = equity_curve.ffill().fillna(capital)

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': equity_curve.iloc[-1],
        }
```

- [ ] **Step 3: Write backtest/analyzer.py**

```python
"""Performance analyzer for backtest results."""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE


class PerformanceAnalyzer:
    """Calculates performance metrics from backtest results."""

    @staticmethod
    def analyze(result: dict) -> dict:
        equity = result['equity_curve']
        trades = result['trades']

        initial_equity = equity.iloc[0]
        final_equity = equity.iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100

        trading_days = len(equity)
        years = trading_days / 252
        if years > 0 and initial_equity > 0:
            annualized_return = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0

        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0

        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_returns = daily_returns - RISK_FREE_RATE / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0.0

        winning_trades = sum(1 for t in trades if t['profit_loss'] > 0)
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0)
        gross_loss = abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0))
        if gross_loss != 0:
            profit_loss_ratio = gross_profit / gross_loss
        else:
            profit_loss_ratio = float('inf') if gross_profit > 0 else 0.0

        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate': round(win_rate, 2),
            'profit_loss_ratio': round(profit_loss_ratio, 2),
            'total_trades': total_trades,
            'final_equity': round(final_equity, 2),
        }
```

- [ ] **Step 4: Write backtest/reporter.py** (see Subagent 3 step below)

The reporter generates console text, PNG charts (equity curve, drawdown, trade signals), and an HTML report with embedded charts. Implementation: uses matplotlib with Agg backend, generates 3 charts, HTML with styled metrics cards and trade table.

- [ ] **Step 5: Run backtest tests**

```bash
cd /work/quant-trading-system && python -m pytest tests/test_backtest.py -v
```

---

## Subagent 4: Executor Module — executor/ + tests/test_executor.py

### Task 4.1: Create executor/__init__.py, portfolio.py, broker.py, logger.py

**Files:**
- Create: `executor/__init__.py`
- Create: `executor/portfolio.py`
- Create: `executor/broker.py`
- Create: `executor/logger.py`
- Create: `tests/test_executor.py`

**Interfaces:**
- `Portfolio(initial_cash, commission_rate)` — buy/sell/update_prices/get_equity/get_position
- `Broker(slippage_rate)` — place_market_order/place_limit_order/execute_pending_orders/cancel_order
- `TradeLogger(log_dir)` — log_trade/log_order/log_fill/log_portfolio_snapshot
- Order dataclass with OrderType(MARKET/LIMIT), OrderSide(BUY/SELL), OrderStatus(PENDING/FILLED/CANCELLED/REJECTED)

- [ ] **Step 1: Write executor/__init__.py**

```python
from .portfolio import Portfolio, Position
from .broker import Broker, Order, OrderType, OrderSide, OrderStatus
from .logger import TradeLogger

__all__ = ["Portfolio", "Position", "Broker", "Order", "OrderType",
           "OrderSide", "OrderStatus", "TradeLogger"]
```

- [ ] **Step 2: Write executor/portfolio.py** — `Position` dataclass (symbol, shares, avg_cost, current_price) with market_value, cost_basis, unrealized_pnl properties. `Portfolio` class with cash tracking, buy(symbol, shares, price, date), sell(symbol, shares, price, date), update_prices(prices dict), get_equity(prices), get_position(symbol), trade_history.

- [ ] **Step 3: Write executor/broker.py** — `OrderType(MARKET, LIMIT)`, `OrderSide(BUY, SELL)`, `OrderStatus(PENDING, FILLED, CANCELLED, REJECTED)` enums. `Order` dataclass. `Broker` with place_market_order, place_limit_order, execute_pending_orders (market orders fill immediately with slippage; limit orders fill when price condition met), cancel_order.

- [ ] **Step 4: Write executor/logger.py** — `TradeLogger` writes JSON-line trade logs to `logs/` directory. Methods: log_trade(dict), log_order(order), log_fill(order), log_portfolio_snapshot(portfolio, date).

- [ ] **Step 5: Write tests/test_executor.py** — Tests for Portfolio (buy/sell/equity/insufficient cash/position tracking), Broker (market/limit orders, fills, cancellations, slippage), TradeLogger (file creation, content verification).

- [ ] **Step 6: Run executor tests**

```bash
cd /work/quant-trading-system && python -m pytest tests/test_executor.py -v
```

---

## Subagent 5: Integration — main.py + README.md + End-to-End Validation

### Task 5.1: Create tests/__init__.py

**Files:**
- Create: `tests/__init__.py` (empty file)

### Task 5.2: Create main.py

**Files:**
- Create: `main.py`

**Interfaces:**
- CLI with argparse: `backtest` and `simulate` subcommands
- Arguments: --symbol, --strategy, --start, --end, --capital, --days

Detailed implementation:
- `main()` sets up argparse with subparsers
- `--help` shows full help with examples
- `cmd_backtest(args)`: fetches data (cache-first), generates signals, runs backtest, analyzes, generates report
- `cmd_simulate(args)`: fetches recent data, runs day-by-day simulation with Broker + Portfolio + TradeLogger

### Task 5.3: Create README.md

**Files:**
- Create: `README.md`

Comprehensive README with:
- Project description
- Installation instructions
- Quick start examples
- CLI command reference tables
- Project structure
- Performance metrics explanation
- Testing instructions

### Task 5.4: End-to-end validation

- [ ] **Step 1: Install all dependencies**

```bash
pip install yfinance pandas matplotlib
```

- [ ] **Step 2: Run all tests**

```bash
cd /work/quant-trading-system && python -m pytest tests/ -v
```

- [ ] **Step 3: Test CLI help**

```bash
python main.py --help
```

- [ ] **Step 4: Run backtest**

```bash
python main.py backtest --symbol AAPL --strategy ma_crossover --start 2023-01-01 --end 2024-12-31 --capital 100000
```

- [ ] **Step 5: Run simulate**

```bash
python main.py simulate --symbol AAPL --strategy momentum_breakout --days 60 --capital 100000
```

- [ ] **Step 6: Cross-validate with second strategy**

```bash
python main.py backtest --symbol AAPL --strategy momentum_breakout --start 2023-01-01 --end 2024-12-31 --capital 100000
```
