"""Extended settings for the quantitative trading system.

This module supplements config.py with additional constants needed
by the strategy, backtest, and risk modules.
"""

from typing import Dict, List, Any
from datetime import date

# =============================================================================
# Market & Data
# =============================================================================

MARKET: str = "us"

SYMBOLS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "JNJ",
]

A_SHARE_SYMBOLS: List[str] = [
    "000001", "600519", "000858", "600036", "601318",
]

START_DATE: date = date(2023, 1, 1)
END_DATE: date = date(2025, 1, 1)

DEFAULT_INTERVAL: str = "1d"
CACHE_DIR: str = "data/cache"

# =============================================================================
# Backtest
# =============================================================================

INITIAL_CAPITAL: float = 1_000_000.0
COMMISSION: float = 0.0003
SLIPPAGE: float = 0.0001

# =============================================================================
# Performance
# =============================================================================

RISK_FREE_RATE: float = 0.02
BENCHMARK_SYMBOL: str = "^GSPC"

# =============================================================================
# Risk Management
# =============================================================================

MAX_POSITION_SIZE: float = 0.20
MAX_DRAWDOWN_LIMIT: float = 0.15
RISK_PER_TRADE: float = 0.02
ATR_PERIOD: int = 14
ATR_STOP_MULTIPLIER: float = 2.0

# =============================================================================
# Strategy Parameters
# =============================================================================

STRATEGY_PARAMS: Dict[str, Dict[str, Any]] = {
    "DualMACrossover": {"fast_period": 20, "slow_period": 50},
    "MACDStrategy": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "TurtleTrading": {"donchian_entry": 20, "donchian_exit": 10, "atr_period": 20},
    "BollingerBands": {"period": 20, "num_std": 2.0},
    "RSIStrategy": {"period": 14, "oversold": 30, "overbought": 70},
    "PairTrading": {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5},
}

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
