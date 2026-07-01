"""Risk management module: position sizing, stop-loss, and portfolio limits."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from settings import (
        RISK_PER_TRADE, MAX_POSITION_SIZE, MAX_DRAWDOWN_LIMIT,
        ATR_STOP_MULTIPLIER,
    )
except ImportError:
    RISK_PER_TRADE = 0.02
    MAX_POSITION_SIZE = 0.20
    MAX_DRAWDOWN_LIMIT = 0.15
    ATR_STOP_MULTIPLIER = 2.0

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
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit
        self.atr_stop_multiplier = atr_stop_multiplier
        self.peak_equity: float = 0.0
        logger.info(
            f"RiskManager: risk_per_trade={risk_per_trade:.1%}, "
            f"max_position={max_position_size:.0%}, max_drawdown={max_drawdown_limit:.0%}"
        )

    def calculate_position_size(
        self, capital: float, price: float, atr: Optional[float] = None
    ) -> int:
        if atr is None or atr <= 0 or np.isnan(atr):
            atr = price * 0.02
        risk_amount = capital * self.risk_per_trade
        stop_distance = atr * self.atr_stop_multiplier
        if stop_distance <= 0:
            stop_distance = price * 0.01
        risk_based_shares = int(risk_amount / stop_distance)
        max_position_value = capital * self.max_position_size
        position_limit_shares = int(max_position_value / price)
        shares = min(risk_based_shares, position_limit_shares)
        return max(0, shares)

    def calculate_atr_stop(self, entry_price: float, atr: float, direction: int = 1) -> float:
        stop_distance = atr * self.atr_stop_multiplier
        return entry_price - stop_distance if direction == 1 else entry_price + stop_distance

    def calculate_kelly_fraction(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        b_ratio = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly = (win_rate * b_ratio - q) / b_ratio
        return max(0.0, min(kelly * 0.5, self.max_position_size))

    def check_portfolio_limits(
        self, positions: Dict[str, float], total_capital: float, current_equity: float
    ) -> Tuple[bool, str]:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        for symbol, value in positions.items():
            fraction = value / current_equity if current_equity > 0 else 1.0
            if fraction > self.max_position_size:
                return False, f"{symbol} position {fraction:.1%} exceeds max {self.max_position_size:.0%}"
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown_limit:
                return False, f"Portfolio drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_limit:.0%}"
        return True, "OK"

    def should_liquidate(self, current_equity: float) -> bool:
        if self.peak_equity <= 0:
            self.peak_equity = current_equity
            return False
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return False
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return drawdown > self.max_drawdown_limit

    def reset(self) -> None:
        self.peak_equity = 0.0
