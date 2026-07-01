"""Performance analyzer for backtest results."""

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE


class PerformanceAnalyzer:
    """Calculates performance metrics from backtest results."""

    @staticmethod
    def analyze(result: dict) -> dict:
        equity = result['equity_curve']
        trades = result['trades']

        initial_equity = equity.iloc[0]
        final_equity = equity.iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100

        trading_days = len(equity)
        years = trading_days / 252
        if years > 0 and initial_equity > 0:
            annualized_return = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0

        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0

        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_returns = daily_returns - RISK_FREE_RATE / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0.0

        winning_trades = sum(1 for t in trades if t['profit_loss'] > 0)
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0)
        gross_loss = abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0))
        if gross_loss != 0:
            profit_loss_ratio = gross_profit / gross_loss
        else:
            profit_loss_ratio = float('inf') if gross_profit > 0 else 0.0

        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate': round(win_rate, 2),
            'profit_loss_ratio': round(profit_loss_ratio, 2),
            'total_trades': total_trades,
            'final_equity': round(final_equity, 2),
        }
