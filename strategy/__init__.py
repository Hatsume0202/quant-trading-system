"""Strategy module with multiple trading strategies."""

from .base import BaseStrategy, Strategy
from .ma_cross import MACrossStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

__all__ = [
    "BaseStrategy",
    "Strategy",
    "MACrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
]
