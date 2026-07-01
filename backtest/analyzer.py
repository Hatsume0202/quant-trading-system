"""Performance analyzer for backtest results: metrics, reports, and charts."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import RISK_FREE_RATE


class PerformanceAnalyzer:
    """Calculates performance metrics and generates reports/charts."""

    def __init__(self, result: dict = None):
        self.result = result or {}
        self.equity = result.get("equity_curve", pd.Series()) if result else pd.Series()
        self.trades = result.get("trades", []) if result else []
        self.symbol = result.get("symbol", "STOCK") if result else "STOCK"

    def analyze(self) -> dict:
        equity = self.equity
        trades = self.trades

        if len(equity) == 0:
            return {
                "total_return": 0.0, "annualized_return": 0.0,
                "max_drawdown": 0.0, "sharpe_ratio": 0.0,
                "win_rate": 0.0, "profit_loss_ratio": 0.0,
                "total_trades": 0, "final_equity": 0.0,
            }

        initial = equity.iloc[0]
        final = equity.iloc[-1]
        total_return_pct = (final / initial - 1) * 100

        trading_days = len(equity)
        years = trading_days / 252
        if years > 0 and initial > 0:
            annual_return_pct = ((final / initial) ** (1 / years) - 1) * 100
        else:
            annual_return_pct = 0.0

        daily_returns = equity.pct_change().dropna()

        rf_daily = RISK_FREE_RATE / 252
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess = daily_returns - rf_daily
            sharpe = np.sqrt(252) * excess.mean() / daily_returns.std()
        else:
            sharpe = 0.0

        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        max_dd_pct = abs(drawdowns.min()) if not drawdowns.empty else 0.0

        winning = [t for t in trades if t.get("profit_loss", 0) > 0]
        losing = [t for t in trades if t.get("profit_loss", 0) < 0]
        total_trades = len(trades)
        win_rate_pct = (len(winning) / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(t.get("profit_loss", 0) for t in winning)
        gross_loss = abs(sum(t.get("profit_loss", 0) for t in losing))
        if gross_loss != 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        return {
            "total_return": round(total_return_pct, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return": round(annual_return_pct, 2),
            "annual_return_pct": round(annual_return_pct, 2),
            "max_drawdown": round(max_dd_pct, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate": round(win_rate_pct, 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            "profit_loss_ratio": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            "total_trades": total_trades,
            "final_equity": round(final, 2),
        }

    def report(self) -> str:
        m = self.analyze()
        return (
            f"{'='*60}\n"
            f"  BACKTEST REPORT — {self.symbol}\n"
            f"{'='*60}\n"
            f"  Final Equity:         ${m['final_equity']:>12,.2f}\n"
            f"  Total Return:         {m['total_return']:>11.2f}%\n"
            f"  Annualized Return:    {m['annualized_return']:>11.2f}%\n"
            f"  Max Drawdown:         {m['max_drawdown']:>11.2f}%\n"
            f"  Sharpe Ratio:         {m['sharpe_ratio']:>11.2f}\n"
            f"  Win Rate:             {m['win_rate']:>11.2f}%\n"
            f"  Profit/Loss Ratio:    {m['profit_loss_ratio']:>11.2f}\n"
            f"  Total Trades:         {m['total_trades']:>11}\n"
            f"{'='*60}"
        )

    def plot(self, save_dir: str = "./reports"):
        os.makedirs(save_dir, exist_ok=True)
        equity = self.equity
        trades = self.trades
        plt.style.use("seaborn-v0_8-darkgrid")

        # Equity Curve
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(equity.index, equity.values, 'b-', linewidth=1.5, label='Portfolio Value')
        ax1.axhline(y=equity.iloc[0], color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(f"Equity Curve — {self.symbol}", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Drawdown
        fig, ax = plt.subplots(figsize=(12, 4))
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        ax.fill_between(drawdowns.index, 0, drawdowns.values, color="#d62728", alpha=0.5)
        ax.plot(drawdowns.index, drawdowns.values, 'r-', linewidth=0.8)
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.set_title("Drawdown", fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "drawdown.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# Compatibility alias
Analyzer = PerformanceAnalyzer
