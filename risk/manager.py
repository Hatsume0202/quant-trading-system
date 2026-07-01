"""Risk management: stop-loss, take-profit, position sizing, drawdown limits."""

import logging
import numpy as np
from typing import Dict, Optional
from config.settings import Config

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio risk management for backtesting.

    Provides:
    - Stop-loss / take-profit checks per position
    - Position concentration limits
    - Drawdown circuit breaker
    - Kelly criterion position sizing
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize with config.

        Args:
            config: System Config. Uses defaults if None.
        """
        self.config = config or Config()
        self._entry_prices: Dict[str, float] = {}
        self._peak_equity: float = 0.0

    def check_stop_loss(self, symbol: str, current_price: float, shares: int) -> bool:
        """Check if stop-loss should trigger.

        Returns True if current price has dropped below stop-loss threshold
        relative to entry price.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            shares: Current position size.

        Returns:
            True if stop-loss triggered.
        """
        entry = self._entry_prices.get(symbol)
        if entry is None or entry <= 0:
            return False
        pnl_pct = (current_price - entry) / entry
        return pnl_pct <= self.config.STOP_LOSS

    def check_take_profit(self, symbol: str, current_price: float, shares: int) -> bool:
        """Check if take-profit should trigger.

        Returns True if current price has risen above take-profit threshold
        relative to entry price.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            shares: Current position size.

        Returns:
            True if take-profit triggered.
        """
        entry = self._entry_prices.get(symbol)
        if entry is None or entry <= 0:
            return False
        pnl_pct = (current_price - entry) / entry
        return pnl_pct >= self.config.TAKE_PROFIT

    def record_entry(self, symbol: str, price: float):
        """Record entry price for a position.

        Args:
            symbol: Ticker symbol.
            price: Entry price.
        """
        self._entry_prices[symbol] = price

    def record_exit(self, symbol: str):
        """Clear entry price record on position exit."""
        self._entry_prices.pop(symbol, None)

    def check_drawdown_limit(self, current_equity: float) -> bool:
        """Check if max drawdown circuit breaker triggers.

        Args:
            current_equity: Current portfolio value.

        Returns:
            True if drawdown exceeds limit (should halt trading).
        """
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            return False
        if self._peak_equity <= 0:
            return False
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        return drawdown >= self.config.MAX_DRAWDOWN_LIMIT

    def calculate_kelly_fraction(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Kelly criterion for optimal position sizing.

        f* = (p*b - q)/b, where p=win_rate, q=1-p, b=avg_win/avg_loss.
        Uses half-Kelly for safety.

        Returns:
            Recommended fraction of capital to risk (0 to MAX_POSITION_SIZE).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        return max(0.0, min(kelly * 0.5, self.config.MAX_POSITION_SIZE))

    def reset(self):
        """Reset state for a new backtest."""
        self._entry_prices.clear()
        self._peak_equity = 0.0
