"""Simulated broker handling order execution, portfolio, and cash management."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import pandas as pd

from config.settings import Config


@dataclass
class TradeRecord:
    """Single trade execution record."""
    date: pd.Timestamp
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    shares: int
    commission: float
    slippage_cost: float
    stamp_duty: float
    cash_flow: float
    position_after: int


class Broker:
    """Simulated broker for backtesting.

    Manages:
    - Cash balance and position tracking
    - Commission, slippage, stamp duty calculations
    - Trade recording
    - Risk manager integration (optional)

    Buy price:  price * (1 + slippage)
    Sell price: price * (1 - slippage)
    Both incur commission; sells also incur stamp duty.
    """

    def __init__(self, config: Config, risk_manager=None):
        """Initialize broker with config and optional risk manager.

        Args:
            config: System configuration.
            risk_manager: Optional RiskManager instance for position checks.
        """
        self.config = config
        self.risk_manager = risk_manager
        self.cash = config.INITIAL_CAPITAL
        self.initial_capital = config.INITIAL_CAPITAL
        self.positions: Dict[str, int] = {}  # symbol -> shares held
        self.trades: List[TradeRecord] = []
        self.equity_history: list = []  # list of (date, equity)

    def execute_buy(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        shares: Optional[int] = None,
        amount: Optional[float] = None,
    ) -> Optional[TradeRecord]:
        """Execute a buy order.

        Args:
            symbol: Ticker symbol.
            date: Trade date.
            price: Reference price (adjusted for slippage internally).
            shares: Number of shares to buy (mutually exclusive with amount).
            amount: Dollar amount to invest (mutually exclusive with shares).

        Returns:
            TradeRecord if executed, None if rejected.
        """
        buy_price = price * (1 + self.config.SLIPPAGE)

        if shares is not None:
            pass  # use provided shares
        elif amount is not None:
            shares = int(amount / (buy_price * (1 + self.config.COMMISSION_RATE)))
        else:
            raise ValueError("Either shares or amount must be specified")

        if shares <= 0:
            return None

        trade_value = buy_price * shares
        commission_total = trade_value * self.config.COMMISSION_RATE
        total_cost = trade_value + commission_total

        # Risk check
        if self.risk_manager is not None:
            current_equity = self.get_equity({symbol: price})
            if not self.risk_manager.can_open_position(symbol, total_cost, current_equity, self.positions):
                return None

        # Cash check - buy as many as we can afford
        if total_cost > self.cash:
            max_shares = int(self.cash / (buy_price * (1 + self.config.COMMISSION_RATE)))
            if max_shares <= 0:
                return None
            shares = max_shares
            trade_value = buy_price * shares
            commission_total = trade_value * self.config.COMMISSION_RATE
            total_cost = trade_value + commission_total

        # Execute
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares

        # Record entry price with risk manager
        if self.risk_manager is not None:
            self.risk_manager.record_entry(symbol, buy_price)

        trade = TradeRecord(
            date=date,
            symbol=symbol,
            action='buy',
            price=buy_price,
            shares=shares,
            commission=commission_total,
            slippage_cost=price * self.config.SLIPPAGE * shares,
            stamp_duty=0.0,
            cash_flow=-total_cost,
            position_after=self.positions[symbol],
        )
        self.trades.append(trade)
        return trade

    def execute_sell(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        shares: Optional[int] = None,
    ) -> Optional[TradeRecord]:
        """Execute a sell order.

        Args:
            symbol: Ticker symbol.
            date: Trade date.
            price: Reference price.
            shares: Shares to sell. If None, sell entire position.

        Returns:
            TradeRecord if executed, None if no position.
        """
        position = self.positions.get(symbol, 0)
        if position <= 0:
            return None

        if shares is None:
            shares = position
        shares = min(shares, position)

        sell_price = price * (1 - self.config.SLIPPAGE)
        trade_value = sell_price * shares
        commission_total = trade_value * self.config.COMMISSION_RATE
        stamp_duty_total = trade_value * self.config.STAMP_DUTY
        net_proceeds = trade_value - commission_total - stamp_duty_total

        self.cash += net_proceeds
        self.positions[symbol] -= shares
        if self.positions[symbol] == 0:
            del self.positions[symbol]

        # Clear entry price with risk manager
        if self.risk_manager is not None:
            self.risk_manager.record_exit(symbol)

        trade = TradeRecord(
            date=date,
            symbol=symbol,
            action='sell',
            price=sell_price,
            shares=shares,
            commission=commission_total,
            slippage_cost=price * self.config.SLIPPAGE * shares,
            stamp_duty=stamp_duty_total,
            cash_flow=net_proceeds,
            position_after=self.positions.get(symbol, 0),
        )
        self.trades.append(trade)
        return trade

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity.

        Args:
            current_prices: Dict mapping symbol -> current price.

        Returns:
            Total equity = cash + market value of positions.
        """
        position_value = sum(
            shares * current_prices.get(sym, 0)
            for sym, shares in self.positions.items()
        )
        return self.cash + position_value

    def record_equity(self, date: pd.Timestamp, current_prices: Dict[str, float]):
        """Record daily equity snapshot.

        Args:
            date: Current date.
            current_prices: Dict mapping symbol -> current price.
        """
        self.equity_history.append((date, self.get_equity(current_prices)))

    def reset(self):
        """Reset broker state for a new backtest run."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
