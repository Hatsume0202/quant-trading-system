"""Risk management module - position sizing, stop-loss, circuit breakers."""

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Config


class RiskManager:
    """Central risk management for the trading system.

    Responsibilities:
    - Position sizing (max allocation per trade)
    - Stop-loss / take-profit enforcement
    - Maximum drawdown circuit breaker
    - Kelly criterion position optimization

    Tracks peak equity and current drawdown across the backtest.
    Records entry prices per symbol for stop-loss/take-profit checks.
    """

    def __init__(self, config: Config):
        """Initialize risk manager.

        Args:
            config: System configuration.
        """
        self.config = config
        self.peak_equity = config.INITIAL_CAPITAL
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False
        self.position_entry_prices: dict = {}  # symbol -> entry price

    def can_open_position(
        self,
        symbol: str,
        trade_cost: float,
        current_equity: float,
        current_positions: dict,
    ) -> bool:
        """Check if a new position can be opened.

        Args:
            symbol: Ticker symbol.
            trade_cost: Total cost of the trade.
            current_equity: Current portfolio equity.
            current_positions: Dict of symbol -> shares held.

        Returns:
            True if position is allowed.
        """
        # Circuit breaker check
        if self.circuit_breaker_triggered:
            return False

        # Drawdown check
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if self.current_drawdown > self.config.MAX_DRAWDOWN_LIMIT:
                self.circuit_breaker_triggered = True
                return False

        # Position size check: trade cost <= max_position_size * equity
        max_allowed = self.config.MAX_POSITION_SIZE * current_equity
        if trade_cost > max_allowed:
            return False

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        return True

    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        position_shares: int,
    ) -> bool:
        """Check if stop-loss is triggered.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            position_shares: Number of shares held.

        Returns:
            True if stop-loss triggered (should sell).
        """
        if symbol not in self.position_entry_prices or position_shares <= 0:
            return False

        entry_price = self.position_entry_prices[symbol]
        pnl_pct = (current_price - entry_price) / entry_price

        return pnl_pct <= self.config.STOP_LOSS

    def check_take_profit(
        self,
        symbol: str,
        current_price: float,
        position_shares: int,
    ) -> bool:
        """Check if take-profit is triggered.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            position_shares: Number of shares held.

        Returns:
            True if take-profit triggered (should sell).
        """
        if symbol not in self.position_entry_prices or position_shares <= 0:
            return False

        entry_price = self.position_entry_prices[symbol]
        pnl_pct = (current_price - entry_price) / entry_price

        return pnl_pct >= self.config.TAKE_PROFIT

    def record_entry(self, symbol: str, entry_price: float):
        """Record position entry price for stop-loss/take-profit tracking.

        Args:
            symbol: Ticker symbol.
            entry_price: Entry price per share.
        """
        self.position_entry_prices[symbol] = entry_price

    def record_exit(self, symbol: str):
        """Clear entry price record when position is closed.

        Args:
            symbol: Ticker symbol.
        """
        self.position_entry_prices.pop(symbol, None)

    def kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
    ) -> float:
        """Calculate optimal position size using Kelly criterion.

        f* = (p * b - q) / b
        where:
            p = win rate, q = 1 - p, b = avg_win / avg_loss

        Uses half-Kelly for conservative sizing.

        Args:
            win_rate: Historical win rate (0 to 1).
            avg_win: Average winning trade profit.
            avg_loss: Average losing trade loss (positive number).
            equity: Current portfolio equity.

        Returns:
            Recommended dollar amount for next position.
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0

        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly_fraction = (win_rate * b - q) / b
        kelly_fraction = max(0.0, min(kelly_fraction, self.config.MAX_POSITION_SIZE))

        # Half-Kelly for safety
        half_kelly = kelly_fraction * 0.5

        return half_kelly * equity

    def reset(self):
        """Reset risk manager state for a new run."""
        self.peak_equity = self.config.INITIAL_CAPITAL
        self.current_drawdown = 0.0
        self.circuit_breaker_triggered = False
        self.position_entry_prices = {}
