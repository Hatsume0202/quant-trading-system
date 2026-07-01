"""Backtest engine for simulating trades based on strategy signals."""

import logging
import pandas as pd
import numpy as np

from config import COMMISSION_RATE, SLIPPAGE_RATE, POSITION_SIZE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulates trade execution based on strategy signals."""

    def __init__(self, initial_capital: float = 100_000.0,
                 commission_rate: float = COMMISSION_RATE,
                 slippage_rate: float = SLIPPAGE_RATE):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def run(self, data: pd.DataFrame, signals: pd.DataFrame,
            capital: float | None = None) -> dict:
        if capital is None:
            capital = self.initial_capital

        cash = capital
        position = 0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity_curve = pd.Series(index=data.index, dtype=float)

        stop_loss_level = None
        take_profit_level = None
        half_sold = False

        for date, row in data.iterrows():
            price = row['Close']
            signal = signals.loc[date, 'Signal'] if date in signals.index else 0

            # Check stop-loss
            if position > 0 and stop_loss_level is not None:
                if row['Low'] <= stop_loss_level:
                    exit_price = stop_loss_level * (1 - self.slippage_rate)
                    proceeds = position * exit_price * (1 - self.commission_rate)
                    cost_basis = position * entry_price * (1 + self.commission_rate)
                    pnl = proceeds - cost_basis
                    trades.append({
                        'entry_date': entry_date, 'exit_date': date,
                        'entry_price': entry_price, 'exit_price': exit_price,
                        'shares': position, 'profit_loss': pnl,
                        'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                        'type': 'stop_loss',
                    })
                    cash += proceeds
                    position = 0
                    entry_price = 0.0
                    stop_loss_level = None
                    take_profit_level = None
                    half_sold = False

            # Check take-profit partial sell
            if position > 0 and take_profit_level is not None and not half_sold:
                if row['High'] >= take_profit_level:
                    sell_shares = position // 2
                    if sell_shares > 0:
                        exit_price = take_profit_level * (1 - self.slippage_rate)
                        proceeds = sell_shares * exit_price * (1 - self.commission_rate)
                        cost_basis = sell_shares * entry_price * (1 + self.commission_rate)
                        pnl = proceeds - cost_basis
                        trades.append({
                            'entry_date': entry_date, 'exit_date': date,
                            'entry_price': entry_price, 'exit_price': exit_price,
                            'shares': sell_shares, 'profit_loss': pnl,
                            'profit_loss_pct': (exit_price / entry_price - 1) * 100,
                            'type': 'take_profit_partial',
                        })
                        cash += proceeds
                        position -= sell_shares
                        half_sold = True

            # Process buy signal
            if signal == 1 and position == 0:
                buy_price = price * (1 + self.slippage_rate)
                max_shares = int((cash * POSITION_SIZE_PCT) / buy_price)
                if max_shares > 0:
                    cost = max_shares * buy_price * (1 + self.commission_rate)
                    if cost <= cash:
                        cash -= cost
                        position = max_shares
                        entry_price = buy_price
                        entry_date = date
                        stop_loss_level = entry_price * (1 - STOP_LOSS_PCT)
                        take_profit_level = entry_price * (1 + TAKE_PROFIT_PCT)
                        half_sold = False

            # Process sell signal
            elif signal == -1 and position > 0:
                sell_price = price * (1 - self.slippage_rate)
                proceeds = position * sell_price * (1 - self.commission_rate)
                cost_basis = position * entry_price * (1 + self.commission_rate)
                pnl = proceeds - cost_basis
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'entry_price': entry_price, 'exit_price': sell_price,
                    'shares': position, 'profit_loss': pnl,
                    'profit_loss_pct': (sell_price / entry_price - 1) * 100,
                    'type': 'signal',
                })
                cash += proceeds
                position = 0
                entry_price = 0.0
                stop_loss_level = None
                take_profit_level = None
                half_sold = False

            equity_curve.loc[date] = cash + position * price

        # Close remaining position at last price
        if position > 0:
            last_price = data['Close'].iloc[-1]
            proceeds = position * last_price * (1 - self.commission_rate)
            cost_basis = position * entry_price * (1 + self.commission_rate)
            pnl = proceeds - cost_basis
            trades.append({
                'entry_date': entry_date, 'exit_date': data.index[-1],
                'entry_price': entry_price, 'exit_price': last_price,
                'shares': position, 'profit_loss': pnl,
                'profit_loss_pct': (last_price / entry_price - 1) * 100,
                'type': 'close_out',
            })

        equity_curve = equity_curve.ffill().fillna(capital)

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': equity_curve.iloc[-1],
        }
