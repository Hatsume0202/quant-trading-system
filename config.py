"""Global configuration for the quantitative trading system.

All strategy parameters, risk limits, and operational settings are centralized here.
Modify these values to adjust system behavior without touching module code.
"""

from typing import Dict, List, Any
from datetime import date

# =============================================================================
# Market & Data Configuration
# =============================================================================

# Market selection: "us" for yfinance US equities, "a" for akshare A-shares
MARKET: str = "us"

# Default stock universe (US equities — liquid large-cap stocks)
SYMBOLS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "JNJ",
]

# A-share symbols (used when MARKET = "a")
A_SHARE_SYMBOLS: List[str] = [
    "000001", "600519", "000858", "600036", "601318",
]

# Date range for backtesting and data fetching
START_DATE: date = date(2023, 1, 1)
END_DATE: date = date(2025, 1, 1)

# Data interval: "1d", "1wk", "1mo"
DEFAULT_INTERVAL: str = "1d"

# Local cache directory for downloaded data
CACHE_DIR: str = "data/cache"

# =============================================================================
# Backtest Configuration
# =============================================================================

# Initial portfolio capital (USD)
INITIAL_CAPITAL: float = 1_000_000.0

# Transaction cost: commission rate per trade
COMMISSION: float = 0.0003  # 0.03%

# Slippage: price impact per trade
SLIPPAGE: float = 0.0001  # 0.01%

# =============================================================================
# Performance Metrics
# =============================================================================

# Risk-free rate for Sharpe ratio calculation
RISK_FREE_RATE: float = 0.02  # 2% annual

# Benchmark symbol for alpha/beta calculation
BENCHMARK_SYMBOL: str = "^GSPC"  # S&P 500 index

# =============================================================================
# Risk Management
# =============================================================================

# Maximum allocation to a single stock (as fraction of portfolio)
MAX_POSITION_SIZE: float = 0.20  # 20%

# Portfolio-level maximum drawdown before liquidation
MAX_DRAWDOWN_LIMIT: float = 0.15  # 15%

# Maximum risk per trade as fraction of total capital
RISK_PER_TRADE: float = 0.02  # 2%

# ATR parameters for stop-loss calculation
ATR_PERIOD: int = 14
ATR_STOP_MULTIPLIER: float = 2.0

# =============================================================================
# Strategy Parameters
# =============================================================================

STRATEGY_PARAMS: Dict[str, Dict[str, Any]] = {
    # Trend Following Strategies
    "DualMACrossover": {
        "fast_period": 20,
        "slow_period": 50,
    },
    "MACDStrategy": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    },
    "TurtleTrading": {
        "donchian_entry": 20,
        "donchian_exit": 10,
        "atr_period": 20,
    },
    # Mean Reversion Strategies
    "BollingerBands": {
        "period": 20,
        "num_std": 2.0,
    },
    "RSIStrategy": {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
    },
    "PairTrading": {
        "lookback": 60,
        "entry_z": 2.0,
        "exit_z": 0.5,
    },
}

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
