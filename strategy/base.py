"""Base strategy abstract class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    All strategies must implement generate_signals() which returns a DataFrame
    with a 'signal' column: 1 = buy/long, -1 = sell/short, 0 = hold/no action.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize strategy with optional parameter overrides.

        Args:
            params: Dictionary of parameter overrides. If None, uses config defaults.
            **kwargs: Additional parameters stored in self.params.
        """
        self.name: str = self.__class__.__name__
        if params is not None:
            self.params: Dict[str, Any] = params
        else:
            self.params: Dict[str, Any] = dict(kwargs)
        logger.info(f"Initialized {self.name} with params: {self.params}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate trading signals from market data.

        Args:
            data: DataFrame with OHLCV and indicator columns.
            **kwargs: Additional data (e.g., second stock for pair trading).

        Returns:
            DataFrame with 'signal' column (1, -1, or 0) and same index as input.
        """
        pass

    def _validate_data(self, data: pd.DataFrame, required_cols: list) -> None:
        """Validate that required columns exist in the data.

        Args:
            data: Input DataFrame.
            required_cols: List of required column names.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: Missing required columns: {missing}")

    def get_params(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.params


# Backward compatibility alias
Strategy = BaseStrategy
