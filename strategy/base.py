"""Base strategy class defining the interface for all trading strategies."""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement generate_signals which returns
    a Series of trading signals aligned with the input data index.

    Signal values:
        1  -> Long entry (buy)
        -1 -> Exit position (sell)
        0  -> Hold / no action
    """

    def __init__(self, **params):
        """Initialize strategy with configurable parameters.

        Args:
            **params: Strategy-specific parameters.
        """
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data.

        Args:
            data: DataFrame with price and indicator columns.

        Returns:
            pd.Series of signals with same index as data.
            Values: 1 (buy), -1 (sell), 0 (hold).
        """
        pass

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
