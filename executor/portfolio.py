"""Portfolio management — tracks positions, cash, and equity."""

from dataclasses import dataclass
import logging

from config import COMMISSION_RATE

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a holding in a single stock."""
    symbol: str
    shares: int
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1) * 100


class Portfolio:
    """Manages account portfolio including cash and positions."""

    def __init__(self, initial_cash: float = 100_000.0,
                 commission_rate: float = COMMISSION_RATE):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.positions: dict[str, Position] = {}
        self.trade_history: list[dict] = []

    def buy(self, symbol: str, shares: int, price: float, date=None) -> float:
        """Execute a buy order. Returns the total cost."""
        if shares <= 0:
            raise ValueError("Shares must be positive for buy")
        cost = shares * price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        if total_cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")

        self.cash -= total_cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            total_cost_basis = pos.cost_basis + cost
            pos.shares = total_shares
            pos.avg_cost = total_cost_basis / total_shares if total_shares > 0 else 0
        else:
            self.positions[symbol] = Position(
                symbol=symbol, shares=shares,
                avg_cost=price, current_price=price,
            )

        trade = {
            'action': 'BUY', 'symbol': symbol, 'shares': shares,
            'price': price, 'cost': cost, 'commission': commission,
            'total_cost': total_cost, 'date': date,
        }
        self.trade_history.append(trade)
        logger.info(f"BUY {shares} {symbol} @ ${price:.2f}, cost=${total_cost:.2f}")
        return total_cost

    def sell(self, symbol: str, shares: int, price: float, date=None) -> float:
        """Execute a sell order. Returns the net proceeds."""
        if shares <= 0:
            raise ValueError("Shares must be positive for sell")
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")
        if shares > self.positions[symbol].shares:
            raise ValueError(f"Insufficient shares: have {self.positions[symbol].shares}, trying to sell {shares}")

        proceeds = shares * price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission

        self.cash += net_proceeds
        pos = self.positions[symbol]
        pos.shares -= shares
        pos.current_price = price

        realized_pnl = (price - pos.avg_cost) * shares - commission

        if pos.shares == 0:
            del self.positions[symbol]

        trade = {
            'action': 'SELL', 'symbol': symbol, 'shares': shares,
            'price': price, 'proceeds': proceeds, 'commission': commission,
            'net_proceeds': net_proceeds, 'realized_pnl': realized_pnl, 'date': date,
        }
        self.trade_history.append(trade)
        logger.info(f"SELL {shares} {symbol} @ ${price:.2f}, proceeds=${net_proceeds:.2f}, P&L=${realized_pnl:.2f}")
        return net_proceeds

    def update_prices(self, prices: dict[str, float]):
        """Update current prices for all positions."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

    def get_equity(self, prices: dict[str, float] | None = None) -> float:
        """Calculate total portfolio equity."""
        if prices:
            self.update_prices(prices)
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    @property
    def total_return(self) -> float:
        return (self.get_equity() / self.initial_cash - 1) * 100

    @property
    def position_count(self) -> int:
        return len(self.positions)
