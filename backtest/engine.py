"""Backtest engine for simulating trading strategies."""

import logging
from typing import Optional
import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL, COMMISSION, SLIPPAGE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, POSITION_SIZE_PCT,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest engine that simulates trading with transaction costs.

    Simulates trading with:
    - Commission: percentage of trade value
    - Slippage: adverse price movement on execution
    - Stop-loss and take-profit exits
    - Position sizing based on POSITION_SIZE_PCT of capital
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = COMMISSION,
        slippage_rate: float = SLIPPAGE,
    ):
        """Initialize backtest engine.

        Args:
            initial_capital: Starting portfolio value.
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%).
            slippage_rate: Slippage rate per trade (e.g., 0.0005 = 0.05%).
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stop_loss_pct = STOP_LOSS_PCT
        self.take_profit_pct = TAKE_PROFIT_PCT
        self.position_size_pct = POSITION_SIZE_PCT
        logger.info(
            f"BacktestEngine: capital=${initial_capital:,.0f}, "
            f"commission={commission_rate:.4f}, slippage={slippage_rate:.4f}"
        )

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        capital: Optional[float] = None,
    ) -> dict:
        """Run backtest with given data and signals.

        Args:
            data: OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
            signals: Signal DataFrame with columns ['Signal', 'Price'].
                     Signal: 1=buy, -1=sell, 0=hold.
            capital: Optional override for initial capital.

        Returns:
            dict with keys:
                - equity_curve: pd.Series of portfolio value over time
                - trades: list of trade dicts, each with:
                    entry_date, exit_date, entry_price, exit_price,
                    shares, profit_loss, profit_loss_pct, type
                - final_equity: float, final portfolio value
        """
        init_capital = capital if capital is not None else self.initial_capital

        # Align data and signals on common index
        common_idx = data.index.intersection(signals.index).sort_values()
        if len(common_idx) == 0:
            return {
                'equity_curve': pd.Series(dtype=float),
                'trades': [],
                'final_equity': init_capital,
            }

        prices = data.loc[common_idx, 'Close']
        sig_series = signals.loc[common_idx, 'Signal']

        n = len(common_idx)
        cash = init_capital
        shares = 0
        entry_price = 0.0
        entry_date = None
        equity = np.zeros(n)
        trades: list[dict] = []

        for i in range(n):
            date = common_idx[i]
            price = float(prices.iloc[i])
            signal = int(sig_series.iloc[i])

            # Check stop-loss and take-profit on existing position
            if shares > 0:
                sl_price = entry_price * (1 - self.stop_loss_pct)
                tp_price = entry_price * (1 + self.take_profit_pct)
                if price <= sl_price:
                    # Stop-loss triggered
                    signal = -1
                elif price >= tp_price:
                    # Take-profit triggered
                    signal = -1

            # Execute buy
            if signal == 1 and shares == 0:
                exec_price = price * (1 + self.slippage_rate)
                # Position sizing: use POSITION_SIZE_PCT of capital
                max_capital = cash * self.position_size_pct
                shares = int(max_capital / exec_price)
                if shares < 1:
                    shares = 0
                else:
                    cost = exec_price * shares * (1 + self.commission_rate)
                    if cost <= cash:
                        cash -= cost
                        entry_price = exec_price
                        entry_date = date
                    else:
                        shares = 0

            # Execute sell
            elif signal == -1 and shares > 0:
                exec_price = price * (1 - self.slippage_rate)
                proceeds = exec_price * shares * (1 - self.commission_rate)
                cash += proceeds

                profit_loss = proceeds - (entry_price * shares * (1 + self.commission_rate))
                profit_loss_pct = (exec_price / entry_price - 1) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exec_price,
                    'shares': shares,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'type': 'long',
                })
                shares = 0
                entry_price = 0.0
                entry_date = None

            # Calculate equity
            position_value = shares * price
            equity[i] = cash + position_value

        # Close any remaining position at last price
        if shares > 0:
            final_price = float(prices.iloc[-1]) * (1 - self.slippage_rate)
            proceeds = final_price * shares * (1 - self.commission_rate)
            cash += proceeds
            equity[-1] = cash

            profit_loss = proceeds - (entry_price * shares * (1 + self.commission_rate))
            profit_loss_pct = (final_price / entry_price - 1) * 100

            trades.append({
                'entry_date': entry_date,
                'exit_date': common_idx[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': shares,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'type': 'long',
            })

        equity_curve = pd.Series(equity, index=common_idx, name='equity')

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': float(equity[-1]),
            'symbol': 'STOCK',
        }
