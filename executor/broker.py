"""Simulated broker interface for order management."""

from dataclasses import dataclass
from enum import Enum
import logging

from config import SLIPPAGE_RATE

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    fill_date = None
    commission: float = 0.0

    @property
    def order_id(self) -> str:
        return f"{self.symbol}_{self.side.value}_{id(self)}"


class Broker:
    """Simulated brokerage that manages orders and executes against market data."""

    def __init__(self, slippage_rate: float = SLIPPAGE_RATE):
        self.slippage_rate = slippage_rate
        self.orders: list[Order] = []
        self.filled_orders: list[Order] = []

    def place_market_order(self, symbol: str, side: OrderSide,
                           quantity: int) -> Order:
        order = Order(symbol=symbol, side=side, quantity=quantity,
                      order_type=OrderType.MARKET)
        self.orders.append(order)
        logger.info(f"Placed {side.value.upper()} market order: {quantity} {symbol}")
        return order

    def place_limit_order(self, symbol: str, side: OrderSide,
                          quantity: int, limit_price: float) -> Order:
        order = Order(symbol=symbol, side=side, quantity=quantity,
                      order_type=OrderType.LIMIT, limit_price=limit_price)
        self.orders.append(order)
        logger.info(f"Placed {side.value.upper()} limit order: {quantity} {symbol} @ ${limit_price:.2f}")
        return order

    def execute_pending_orders(self, current_prices: dict[str, float],
                                date=None) -> list[Order]:
        filled = []
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            if order.symbol not in current_prices:
                continue

            market_price = current_prices[order.symbol]

            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    order.fill_price = market_price * (1 + self.slippage_rate)
                else:
                    order.fill_price = market_price * (1 - self.slippage_rate)
                order.status = OrderStatus.FILLED
                order.fill_date = date

            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and market_price <= order.limit_price:
                    order.fill_price = order.limit_price
                    order.status = OrderStatus.FILLED
                    order.fill_date = date
                elif order.side == OrderSide.SELL and market_price >= order.limit_price:
                    order.fill_price = order.limit_price
                    order.status = OrderStatus.FILLED
                    order.fill_date = date

            if order.status == OrderStatus.FILLED:
                filled.append(order)
                self.filled_orders.append(order)
                logger.info(f"Filled {order.side.value.upper()} {order.quantity} {order.symbol} @ ${order.fill_price:.2f}")

        self.orders = [o for o in self.orders if o.status == OrderStatus.PENDING]
        return filled

    def cancel_order(self, order: Order) -> bool:
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            self.orders = [o for o in self.orders if o.order_id != order.order_id]
            logger.info(f"Cancelled order: {order.order_id}")
            return True
        return False

    @property
    def pending_count(self) -> int:
        return len(self.orders)
