"""System configuration parameters for the quant trading system."""

from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for backtesting and trading parameters.

    Attributes:
        INITIAL_CAPITAL: Starting portfolio cash in USD.
        COMMISSION_RATE: Broker commission as fraction (0.0003 = 0.03%).
        SLIPPAGE: Price slippage as fraction (0.0001 = 0.01%).
        STAMP_DUTY: Tax on sell side (0.001 = 0.1%, A-share).
        RISK_FREE_RATE: Annual risk-free rate for Sharpe ratio.
        MAX_POSITION_SIZE: Max fraction of portfolio in one position.
        STOP_LOSS: Stop-loss threshold as negative return fraction.
        TAKE_PROFIT: Take-profit threshold as positive return fraction.
        MAX_DRAWDOWN_LIMIT: Circuit breaker - max allowed drawdown.
    """
    INITIAL_CAPITAL: float = 100_000.0
    COMMISSION_RATE: float = 0.0003
    SLIPPAGE: float = 0.0001
    STAMP_DUTY: float = 0.001
    RISK_FREE_RATE: float = 0.03
    MAX_POSITION_SIZE: float = 0.20
    STOP_LOSS: float = -0.08
    TAKE_PROFIT: float = 0.15
    MAX_DRAWDOWN_LIMIT: float = 0.25


# Singleton instance for easy import
config = Config()
