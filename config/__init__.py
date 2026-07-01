"""Configuration module for the quant trading system.

Re-exports the Config class from .settings and provides top-level constants.
"""

from .settings import Config, config

# Global configuration constants
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

# Risk management
RISK_PER_TRADE = 0.02      # 2% of capital risked per trade
MAX_POSITION_SIZE = 0.20   # 20% max portfolio in one position
MAX_DRAWDOWN_LIMIT = 0.15  # 15% drawdown triggers liquidation
ATR_PERIOD = 14            # ATR lookback period
ATR_STOP_MULTIPLIER = 2.0  # ATR multiplier for stop-loss distance

# Strategy parameter presets (used by mean_reversion module)
STRATEGY_PARAMS = {
    "BollingerBands": {"period": 20, "num_std": 2.0},
    "RSIStrategy": {"period": 14, "oversold": 30, "overbought": 70},
    "PairTrading": {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5},
}

# Output directories
REPORT_DIR = "reports"
LOG_DIR = "logs"
DATA_DIR = "data_cache"

# Logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Backward-compatible aliases for backtest engine
INITIAL_CAPITAL = DEFAULT_CAPITAL     # 100,000
COMMISSION = COMMISSION_RATE          # 0.001
SLIPPAGE = SLIPPAGE_RATE              # 0.0005
BENCHMARK_SYMBOL = "SPY"              # ETF benchmark

__all__ = [
    "Config",
    "config",
    "DEFAULT_SYMBOL",
    "DEFAULT_CAPITAL",
    "DEFAULT_START_DATE",
    "DEFAULT_END_DATE",
    "COMMISSION_RATE",
    "SLIPPAGE_RATE",
    "MA_SHORT_WINDOW",
    "MA_LONG_WINDOW",
    "MOMENTUM_LOOKBACK",
    "MOMENTUM_EXIT_LOOKBACK",
    "RSI_OVERBOUGHT",
    "RSI_OVERSOLD",
    "STOP_LOSS_PCT",
    "TAKE_PROFIT_PCT",
    "POSITION_SIZE_PCT",
    "RISK_FREE_RATE",
    "RISK_PER_TRADE",
    "MAX_POSITION_SIZE",
    "MAX_DRAWDOWN_LIMIT",
    "ATR_PERIOD",
    "ATR_STOP_MULTIPLIER",
    "REPORT_DIR",
    "LOG_DIR",
    "DATA_DIR",
    "LOG_FORMAT",
    "LOG_DATE_FORMAT",
    "INITIAL_CAPITAL",
    "COMMISSION",
    "SLIPPAGE",
    "BENCHMARK_SYMBOL",
    "STRATEGY_PARAMS",
]
