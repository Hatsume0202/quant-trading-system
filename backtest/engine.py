"""Event-driven backtesting engine."""

from typing import Optional

import pandas as pd
import numpy as np

from config.settings import Config, config as default_config
from strategy.base import Strategy


class BacktestEngine:
    """Main backtesting engine.

    Runs a strategy over historical data, tracking portfolio value
    and recording all trades through the broker.

    Architecture: Loop through each bar (day), check for signals,
    execute orders via broker, record daily equity.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize backtest engine.

        Args:
            config: System configuration. Uses default if None.
        """
        self.config = config or default_config
        self.results = None

    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "STOCK",
        risk_manager=None,
    ) -> dict:
        """Run backtest for a single stock.

        Args:
            data: OHLCV DataFrame (with indicators).
            strategy: Strategy instance with generate_signals method.
            symbol: Ticker symbol for labeling.
            risk_manager: Optional RiskManager instance.

        Returns:
            Dictionary with keys:
                'trades': List of trade record dicts
                'equity_curve': pd.Series of daily portfolio value
                'benchmark_curve': pd.Series of buy-and-hold value
                'returns': pd.Series of daily returns
                'config': Config used
                'symbol': str
                'signals': pd.Series of trading signals
        """
        from .broker import Broker

        broker = Broker(self.config, risk_manager=risk_manager)

        # Generate signals
        signals = strategy.generate_signals(data)

        # Benchmark: buy and hold
        initial_price = data['close'].iloc[0]
        benchmark_shares = int(self.config.INITIAL_CAPITAL / initial_price)

        equity_curve = []
        benchmark_curve = []
        dates = []

        for i, (idx, row) in enumerate(data.iterrows()):
            price = row['close']
            signal = signals.iloc[i]

            # Check stop-loss / take-profit if we have a position
            position = broker.positions.get(symbol, 0)
            if risk_manager is not None and position > 0:
                if risk_manager.check_stop_loss(symbol, price, position):
                    broker.execute_sell(symbol, idx, price)
                elif risk_manager.check_take_profit(symbol, price, position):
                    broker.execute_sell(symbol, idx, price)

            # Execute signal
            if signal == 1:  # Buy
                # Close existing position first
                if broker.positions.get(symbol, 0) > 0:
                    broker.execute_sell(symbol, idx, price)
                # Open new long position respecting max position size
                current_equity = broker.get_equity({symbol: price})
                max_position = self.config.MAX_POSITION_SIZE * current_equity
                invest_amount = min(broker.cash, max_position)
                if invest_amount > 0:
                    broker.execute_buy(symbol, idx, price, amount=invest_amount)

            elif signal == -1:  # Sell
                broker.execute_sell(symbol, idx, price)

            # Record equity
            equity = broker.get_equity({symbol: price})
            benchmark = benchmark_shares * price

            equity_curve.append(equity)
            benchmark_curve.append(benchmark)
            dates.append(idx)

        # Build result
        equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(dates), name='equity')
        benchmark_series = pd.Series(benchmark_curve, index=pd.DatetimeIndex(dates), name='benchmark')

        # Daily returns
        returns = equity_series.pct_change().dropna()

        # Convert trades to dicts
        trades_dict = [
            {
                'date': t.date,
                'symbol': t.symbol,
                'action': t.action,
                'price': t.price,
                'shares': t.shares,
                'commission': t.commission,
                'slippage_cost': t.slippage_cost,
                'stamp_duty': t.stamp_duty,
                'cash_flow': t.cash_flow,
                'position_after': t.position_after,
            }
            for t in broker.trades
        ]

        self.results = {
            'trades': trades_dict,
            'equity_curve': equity_series,
            'benchmark_curve': benchmark_series,
            'returns': returns,
            'config': self.config,
            'symbol': symbol,
            'signals': signals,
        }

        return self.results
