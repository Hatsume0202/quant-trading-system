"""Risk management module: position sizing, stop-loss, and portfolio limits."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config import (
    RISK_PER_TRADE, MAX_POSITION_SIZE, MAX_DRAWDOWN_LIMIT,
    ATR_PERIOD, ATR_STOP_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio risk management and position sizing.

    Implements:
    - Fixed-fractional position sizing (max risk per trade)
    - ATR-based dynamic stop-loss
    - Kelly criterion position optimization
    - Single-stock concentration limits
    - Portfolio-level drawdown circuit breaker
    """

    def __init__(
        self,
        risk_per_trade: float = RISK_PER_TRADE,
        max_position_size: float = MAX_POSITION_SIZE,
        max_drawdown_limit: float = MAX_DRAWDOWN_LIMIT,
        atr_stop_multiplier: float = ATR_STOP_MULTIPLIER,
    ):
        """Initialize risk manager with configurable limits.

        Args:
            risk_per_trade: Max fraction of capital to risk per trade (default 2%).
            max_position_size: Max fraction of portfolio in one stock (default 20%).
            max_drawdown_limit: Portfolio drawdown that triggers liquidation (default 15%).
            atr_stop_multiplier: ATR multiplier for stop-loss distance.
        """
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit
        self.atr_stop_multiplier = atr_stop_multiplier
        self.peak_equity: float = 0.0
        logger.info(
            f"RiskManager initialized: risk_per_trade={risk_per_trade:.1%}, "
            f"max_position={max_position_size:.0%}, max_drawdown={max_drawdown_limit:.0%}"
        )

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> int:
        """Calculate the number of shares to trade based on risk limits.

        Uses fixed-fractional method: shares = (capital * risk_per_trade) / (atr * multiplier).
        Capped by max_position_size of total portfolio.

        Args:
            capital: Available cash.
            price: Current stock price.
            atr: Average True Range for stop distance. If None, uses 2% of price.
            volatility: Alternative risk measure (not used if ATR provided).

        Returns:
            Number of shares (integer).
        """
        if atr is None or atr <= 0 or np.isnan(atr):
            atr = price * 0.02  # Default: 2% of price

        # Risk amount: how much money we're willing to lose
        risk_amount = capital * self.risk_per_trade

        # Stop distance: how far below entry the stop sits
        stop_distance = atr * self.atr_stop_multiplier

        if stop_distance <= 0:
            stop_distance = price * 0.01

        # Shares based on risk
        risk_based_shares = int(risk_amount / stop_distance)

        # Shares based on position size limit
        max_position_value = capital * self.max_position_size
        position_limit_shares = int(max_position_value / price)

        shares = min(risk_based_shares, position_limit_shares)
        shares = max(0, shares)  # No negative shares

        logger.debug(
            f"Position sizing: capital=${capital:,.0f}, price=${price:.2f}, "
            f"atr=${atr:.2f}, shares={shares}"
        )
        return shares

    def calculate_atr_stop(
        self, entry_price: float, atr: float, direction: int = 1
    ) -> float:
        """Calculate ATR-based stop-loss price.

        Args:
            entry_price: Trade entry price.
            atr: Current ATR value.
            direction: 1 for long (stop below), -1 for short (stop above).

        Returns:
            Stop-loss price level.
        """
        stop_distance = atr * self.atr_stop_multiplier
        if direction == 1:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_trailing_stop(
        self, current_price: float, atr: float, direction: int = 1
    ) -> float:
        """Calculate trailing stop based on current price.

        Args:
            current_price: Current market price.
            atr: Current ATR value.
            direction: 1 for long, -1 for short.

        Returns:
            Trailing stop price level.
        """
        stop_distance = atr * self.atr_stop_multiplier
        if direction == 1:
            return current_price - stop_distance
        else:
            return current_price + stop_distance

    def calculate_kelly_fraction(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate optimal position size using Kelly criterion.

        Kelly formula: f* = (p * b - q) / b
        where p = win_rate, q = 1-p, b = avg_win/avg_loss

        Args:
            win_rate: Fraction of winning trades (0 to 1).
            avg_win: Average gain per winning trade (positive).
            avg_loss: Average loss per losing trade (positive number).

        Returns:
            Kelly fraction (clamped to [0, max_position_size]).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0

        b_ratio = avg_win / avg_loss
        q = 1.0 - win_rate

        kelly = (win_rate * b_ratio - q) / b_ratio

        # Use half-Kelly for safety, cap at max position size
        kelly = max(0.0, min(kelly * 0.5, self.max_position_size))

        logger.debug(
            f"Kelly: win_rate={win_rate:.2f}, avg_win={avg_win:.4f}, "
            f"avg_loss={avg_loss:.4f}, fraction={kelly:.4f}"
        )
        return kelly

    def check_portfolio_limits(
        self,
        positions: Dict[str, float],
        total_capital: float,
        current_equity: float,
    ) -> Tuple[bool, str]:
        """Check if current portfolio state violates any risk limits.

        Args:
            positions: Dict mapping symbol to current position value.
            total_capital: Total portfolio starting capital.
            current_equity: Current portfolio equity.

        Returns:
            Tuple of (is_valid, reason).
        """
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check single position concentration
        for symbol, value in positions.items():
            fraction = value / current_equity if current_equity > 0 else 1.0
            if fraction > self.max_position_size:
                return False, f"{symbol} position {fraction:.1%} exceeds max {self.max_position_size:.0%}"

        # Check portfolio drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown_limit:
                return False, f"Portfolio drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_limit:.0%}"

        return True, "OK"

    def apply_risk_limits(
        self, trades: List[Dict], capital: float
    ) -> List[Dict]:
        """Filter and adjust trades to comply with risk limits.

        Args:
            trades: List of trade dicts with keys: symbol, price, shares, direction.
            capital: Available capital.

        Returns:
            Filtered/adjusted list of trades.
        """
        validated_trades = []
        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            price = trade.get("price", 0)
            direction = trade.get("direction", 1)

            if price <= 0:
                logger.warning(f"Skipping {symbol}: invalid price {price}")
                continue

            shares = trade.get("shares", 0)
            max_shares = self.calculate_position_size(capital, price)

            if shares > max_shares:
                logger.info(
                    f"Reducing {symbol} from {shares} to {max_shares} shares "
                    f"(position limit)"
                )
                trade["shares"] = max_shares
                trade["original_shares"] = shares

            validated_trades.append(trade)

        return validated_trades

    def should_liquidate(self, current_equity: float) -> bool:
        """Check if portfolio drawdown triggers liquidation.

        Args:
            current_equity: Current portfolio value.

        Returns:
            True if drawdown exceeds limit.
        """
        if self.peak_equity <= 0:
            self.peak_equity = current_equity
            return False

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown_limit:
            logger.warning(
                f"LIQUIDATION TRIGGERED: drawdown {drawdown:.1%} > limit {self.max_drawdown_limit:.0%}"
            )
            return True

        return False

    def reset(self) -> None:
        """Reset peak equity tracker (e.g., between backtests)."""
        self.peak_equity = 0.0
