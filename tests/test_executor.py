"""Tests for executor module — Portfolio, Broker, TradeLogger."""

import pytest
import os
import tempfile

from executor.portfolio import Portfolio, Position
from executor.broker import Broker, OrderType, OrderSide, OrderStatus, Order
from executor.logger import TradeLogger


class TestPortfolio:

    def test_initialization(self):
        p = Portfolio(initial_cash=100000)
        assert p.cash == 100000
        assert p.position_count == 0
        assert p.get_equity() == 100000

    def test_buy_reduces_cash(self):
        p = Portfolio(initial_cash=100000)
        cost = p.buy('AAPL', 100, 150.0)
        assert cost > 0
        assert p.cash < 100000
        assert p.position_count == 1

    def test_buy_creates_position(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 100, 150.0)
        pos = p.get_position('AAPL')
        assert pos is not None
        assert pos.symbol == 'AAPL'
        assert pos.shares == 100

    def test_buy_insufficient_cash_raises(self):
        p = Portfolio(initial_cash=1000)
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.buy('AAPL', 10000, 150.0)

    def test_sell_reduces_position(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 200, 150.0)
        proceeds = p.sell('AAPL', 100, 160.0)
        assert proceeds > 0
        pos = p.get_position('AAPL')
        assert pos.shares == 100

    def test_sell_all_removes_position(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 100, 150.0)
        p.sell('AAPL', 100, 160.0)
        assert p.position_count == 0

    def test_sell_not_held_raises(self):
        p = Portfolio(initial_cash=100000)
        with pytest.raises(ValueError, match="No position"):
            p.sell('AAPL', 100, 150.0)

    def test_sell_too_many_shares_raises(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 100, 150.0)
        with pytest.raises(ValueError, match="Insufficient shares"):
            p.sell('AAPL', 200, 160.0)

    def test_equity_reflects_price_changes(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 100, 150.0)
        equity1 = p.get_equity({'AAPL': 160.0})
        assert equity1 > 100000
        equity2 = p.get_equity({'AAPL': 140.0})
        assert equity2 < 100000

    def test_total_return_calculation(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 100, 150.0)
        p.update_prices({'AAPL': 165.0})
        assert p.total_return > 0

    def test_trade_history_recorded(self):
        p = Portfolio(initial_cash=100000)
        p.buy('AAPL', 100, 150.0)
        p.sell('AAPL', 100, 160.0)
        assert len(p.trade_history) == 2
        assert p.trade_history[0]['action'] == 'BUY'
        assert p.trade_history[1]['action'] == 'SELL'


class TestBroker:

    def test_place_market_order(self):
        b = Broker()
        order = b.place_market_order('AAPL', OrderSide.BUY, 100)
        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_place_limit_order(self):
        b = Broker()
        order = b.place_limit_order('AAPL', OrderSide.BUY, 100, 148.0)
        assert order.limit_price == 148.0
        assert order.order_type == OrderType.LIMIT

    def test_market_order_fills_immediately(self):
        b = Broker()
        order = b.place_market_order('AAPL', OrderSide.BUY, 100)
        filled = b.execute_pending_orders({'AAPL': 150.0})
        assert len(filled) == 1
        assert filled[0].status == OrderStatus.FILLED
        assert filled[0].fill_price > 150.0

    def test_limit_buy_fills_at_or_below_limit(self):
        b = Broker()
        b.place_limit_order('AAPL', OrderSide.BUY, 100, 148.0)
        filled = b.execute_pending_orders({'AAPL': 145.0})
        assert len(filled) == 1
        assert filled[0].fill_price == 148.0

    def test_limit_buy_does_not_fill_above_limit(self):
        b = Broker()
        order = b.place_limit_order('AAPL', OrderSide.BUY, 100, 148.0)
        filled = b.execute_pending_orders({'AAPL': 150.0})
        assert len(filled) == 0
        assert order.status == OrderStatus.PENDING

    def test_limit_sell_fills_at_or_above_limit(self):
        b = Broker()
        b.place_limit_order('AAPL', OrderSide.SELL, 100, 152.0)
        filled = b.execute_pending_orders({'AAPL': 155.0})
        assert len(filled) == 1

    def test_cancel_order(self):
        b = Broker()
        order = b.place_market_order('AAPL', OrderSide.BUY, 100)
        assert b.cancel_order(order) is True
        assert order.status == OrderStatus.CANCELLED
        assert b.pending_count == 0

    def test_multiple_orders(self):
        b = Broker()
        b.place_market_order('AAPL', OrderSide.BUY, 100)
        b.place_market_order('GOOGL', OrderSide.SELL, 50)
        assert b.pending_count == 2
        filled = b.execute_pending_orders({'AAPL': 150.0, 'GOOGL': 140.0})
        assert len(filled) == 2
        assert b.pending_count == 0


class TestTradeLogger:

    def test_log_trade_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tl = TradeLogger(log_dir=tmpdir)
            tl.log_trade({'action': 'BUY', 'symbol': 'AAPL', 'shares': 100, 'price': 150.0})
            assert os.path.exists(tl.log_file)
            with open(tl.log_file) as f:
                content = f.read()
            assert 'AAPL' in content
            assert 'BUY' in content

    def test_log_order_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tl = TradeLogger(log_dir=tmpdir)
            b = Broker()
            order = b.place_market_order('MSFT', OrderSide.BUY, 50)
            tl.log_order(order)
            assert os.path.exists(tl.log_file)

    def test_log_portfolio_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tl = TradeLogger(log_dir=tmpdir)
            p = Portfolio()
            tl.log_portfolio_snapshot(p)
            assert os.path.exists(tl.log_file)
