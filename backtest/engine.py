"""Backtest engine - event-driven loop over historical data."""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from config.settings import Config, config as default_config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtesting engine.

    Iterates through historical bars, generates strategy signals,
    executes orders through a simulated broker, and tracks equity.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config

    def run(
        self,
        data: pd.DataFrame,
        strategy,
        symbol: str = "STOCK",
        risk_manager=None,
    ) -> dict:
        """Run backtest for a single stock.

        Args:
            data: OHLCV DataFrame with indicator columns.
            strategy: Strategy instance with generate_signals().
            symbol: Ticker symbol.
            risk_manager: Optional RiskManager instance.

        Returns:
            dict with keys: trades, equity_curve, benchmark_curve,
                            returns, config, symbol, signals.
        """
        from .broker import Broker

        broker = Broker(self.config, risk_manager=risk_manager)
        signals = strategy.generate_signals(data)

        initial_price = data["close"].iloc[0]
        benchmark_shares = int(self.config.INITIAL_CAPITAL / initial_price)

        equity_curve = []
        benchmark_curve = []
        dates = []

        for i, (idx, row) in enumerate(data.iterrows()):
            price = row["close"]
            signal = int(signals.iloc[i]) if i < len(signals) else 0

            # Risk manager checks
            position = broker.positions.get(symbol, 0)
            if risk_manager is not None and position > 0:
                if risk_manager.check_stop_loss(symbol, price, position):
                    broker.execute_sell(symbol, idx, price)
                elif risk_manager.check_take_profit(symbol, price, position):
                    broker.execute_sell(symbol, idx, price)

            # Execute signals
            if signal == 1:
                if broker.positions.get(symbol, 0) > 0:
                    broker.execute_sell(symbol, idx, price)
                current_equity = broker.get_equity({symbol: price})
                max_position_val = self.config.MAX_POSITION_SIZE * current_equity
                invest_amount = min(broker.cash, max_position_val)
                if invest_amount > 0:
                    broker.execute_buy(symbol, idx, price, amount=invest_amount)
            elif signal == -1:
                broker.execute_sell(symbol, idx, price)

            equity = broker.get_equity({symbol: price})
            benchmark = benchmark_shares * price

            equity_curve.append(equity)
            benchmark_curve.append(benchmark)
            dates.append(idx)

        equity_s = pd.Series(equity_curve, index=dates, name="equity")
        benchmark_s = pd.Series(benchmark_curve, index=dates, name="benchmark")
        returns_s = equity_s.pct_change().fillna(0)

        # Convert TradeRecord objects to dicts
        trades = []
        for t in broker.trades:
            shares = t.shares
            entry_price = t.price
            # Calculate P&L: for sells, compare to avg cost
            trades.append({
                "date": t.date,
                "symbol": t.symbol,
                "action": t.action,
                "price": t.price,
                "shares": t.shares,
                "commission": t.commission,
                "slippage": t.slippage_cost,
                "stamp_duty": t.stamp_duty,
                "cash_flow": t.cash_flow,
            })

        # Enrich trades with profit_loss by matching buy/sell pairs
        trades = self._calculate_trade_pnl(trades)

        return {
            "trades": trades,
            "equity_curve": equity_s,
            "benchmark_curve": benchmark_s,
            "returns": returns_s,
            "config": self.config,
            "symbol": symbol,
            "signals": signals,
        }

    def _calculate_trade_pnl(self, trades: list) -> list:
        """Calculate profit/loss by matching buy/sell pairs."""
        result = []
        buys = []
        for t in trades:
            t_copy = dict(t)
            t_copy["profit_loss"] = 0.0
            if t["action"] == "buy":
                buys.append(t_copy)
            elif t["action"] == "sell":
                if buys:
                    buy = buys.pop(0)
                    buy_cost = buy["price"] * buy["shares"]
                    sell_proceeds = t["price"] * t["shares"]
                    pnl = sell_proceeds - buy_cost - t["commission"] - t["stamp_duty"] - buy["commission"]
                    # Assign P&L proportionally
                    t_copy["profit_loss"] = pnl
                    t_copy["entry_price"] = buy["price"]
                    t_copy["entry_date"] = buy["date"]
                result.append(t_copy)
        # Remaining buys (unclosed positions)
        for b in buys:
            result.append(b)
        return result
