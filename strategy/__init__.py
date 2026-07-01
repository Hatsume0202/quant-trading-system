"""Strategy module for trading signal generation."""

from .base import BaseStrategy, Strategy
from .trend_following import DualMACrossover, MACDStrategy, TurtleTrading
from .mean_reversion import BollingerBands, RSIStrategy, PairTrading

__all__ = [
    "BaseStrategy",
    "Strategy",
    "DualMACrossover",
    "MACDStrategy",
    "TurtleTrading",
    "BollingerBands",
    "RSIStrategy",
    "PairTrading",
]
