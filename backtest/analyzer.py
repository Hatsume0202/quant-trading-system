"""Performance analyzer for backtest results: metrics, reports, and charts."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config.settings import Config


class Analyzer:
    """Calculates performance metrics and generates reports/charts."""

    def __init__(self, result: dict):
        self.result = result
        self.equity = result["equity_curve"]
        self.benchmark = result.get("benchmark_curve", self.equity.copy())
        self.trades = result.get("trades", [])
        self.config = result.get("config", Config())
        self.symbol = result.get("symbol", "STOCK")

    def analyze(self) -> dict:
        equity = self.equity
        trades = self.trades
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
        annual_vol = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0.0

        rf_daily = self.config.RISK_FREE_RATE / 252
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess = daily_returns - rf_daily
            sharpe = np.sqrt(252) * excess.mean() / daily_returns.std()
        else:
            sharpe = 0.0

        downside = daily_returns[daily_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = np.sqrt(252) * (daily_returns.mean() - rf_daily) / downside.std()
        else:
            sortino = 0.0

        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        max_dd_pct = abs(drawdowns.min()) if not drawdowns.empty else 0.0
        calmar = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

        sell_trades = [t for t in trades if t.get("action") == "sell"]
        winning = [t for t in sell_trades if t.get("profit_loss", 0) > 0]
        losing = [t for t in sell_trades if t.get("profit_loss", 0) < 0]
        total_trades = len(sell_trades)
        win_rate_pct = (len(winning) / total_trades * 100) if total_trades > 0 else 0.0

        avg_win = np.mean([t["profit_loss"] for t in winning]) if winning else 0.0
        avg_loss = abs(np.mean([t["profit_loss"] for t in losing])) if losing else 0.0
        if losing and sum(t["profit_loss"] for t in losing) != 0:
            profit_factor = sum(t["profit_loss"] for t in winning) / abs(sum(t["profit_loss"] for t in losing))
        else:
            profit_factor = float("inf")

        bench_final = self.benchmark.iloc[-1] if len(self.benchmark) > 0 else initial
        bench_return_pct = (bench_final / self.benchmark.iloc[0] - 1) * 100 if len(self.benchmark) > 0 else 0.0
        alpha = annual_return_pct - bench_return_pct

        return {
            "total_return_pct": round(total_return_pct, 2),
            "annual_return_pct": round(annual_return_pct, 2),
            "annual_volatility_pct": round(annual_vol, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "calmar_ratio": round(calmar, 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "total_trades": total_trades,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
            "benchmark_return_pct": round(bench_return_pct, 2),
            "alpha_pct": round(alpha, 2),
            "final_equity": round(final, 2),
            "initial_capital": round(initial, 2),
        }

    def report(self) -> str:
        m = self.analyze()
        return (
            f"{'='*60}\n"
            f"  PERFORMANCE REPORT — {self.symbol}\n"
            f"{'='*60}\n"
            f"  Initial Capital:      ${m['initial_capital']:>12,.2f}\n"
            f"  Final Equity:         ${m['final_equity']:>12,.2f}\n"
            f"  Total Return:         {m['total_return_pct']:>11.2f}%\n"
            f"  Annual Return:        {m['annual_return_pct']:>11.2f}%\n"
            f"  Annual Volatility:    {m['annual_volatility_pct']:>11.2f}%\n"
            f"{'-'*60}\n"
            f"  Sharpe Ratio:         {m['sharpe_ratio']:>11.2f}\n"
            f"  Sortino Ratio:        {m['sortino_ratio']:>11.2f}\n"
            f"  Calmar Ratio:         {m['calmar_ratio']:>11.2f}\n"
            f"  Max Drawdown:         {m['max_drawdown_pct']:>11.2f}%\n"
            f"{'-'*60}\n"
            f"  Total Trades:         {m['total_trades']:>11}\n"
            f"  Win Rate:             {m['win_rate_pct']:>11.2f}%\n"
            f"  Avg Win:              ${m['avg_win']:>11,.2f}\n"
            f"  Avg Loss:             ${m['avg_loss']:>11,.2f}\n"
            f"  Profit Factor:        {str(m['profit_factor']):>11}\n"
            f"{'-'*60}\n"
            f"  Benchmark Return:     {m['benchmark_return_pct']:>11.2f}%\n"
            f"  Alpha vs Benchmark:   {m['alpha_pct']:>11.2f}%\n"
            f"{'='*60}"
        )

    def plot(self, save_dir: str = "./output"):
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

        # Chart 1: Equity + Drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(self.equity.index, self.equity.values, label="Strategy", linewidth=1.5, color="#1f77b4")
        ax1.plot(self.benchmark.index, self.benchmark.values, label="Buy & Hold", linewidth=1.2,
                 color="#ff7f0e", alpha=0.7)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title(f"Equity Curve — {self.symbol}", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.grid(True, alpha=0.3)

        rolling_max = self.equity.expanding().max()
        drawdowns = (self.equity - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdowns.index, 0, drawdowns.values, color="#d62728", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Chart 2: Monthly Returns Heatmap
        daily_returns = self.equity.pct_change().dropna()
        if len(daily_returns) > 20:
            monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values * 100,
            })
            pivot = monthly_df.pivot(index="year", columns="month", values="return")
            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.8)))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
            ax.set_xticks(range(12))
            ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index)
            ax.set_title(f"Monthly Returns (%) — {self.symbol}", fontsize=14, fontweight="bold")
            for i in range(len(pivot)):
                for j in range(min(12, pivot.shape[1])):
                    val = pivot.iloc[i, j]
                    if not pd.isna(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                                color="black" if abs(val) < 5 else "white")
            plt.colorbar(im, ax=ax, label="Return (%)")
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, "monthly_returns.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Chart 3: Trade P&L
        sell_trades = [t for t in self.trades if t.get("action") == "sell" and "profit_loss" in t]
        if sell_trades:
            pnls = [t["profit_loss"] for t in sell_trades]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
            ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
            ax1.axhline(y=0, color="black", linewidth=0.5)
            ax1.set_xlabel("Trade #")
            ax1.set_ylabel("P&L ($)")
            ax1.set_title("Trade P&L", fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax2.hist(pnls, bins=20, color="#1f77b4", alpha=0.7, edgecolor="black")
            ax2.axvline(x=0, color="red", linestyle="--", linewidth=0.8)
            ax2.set_xlabel("P&L ($)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("P&L Distribution", fontweight="bold")
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, "trade_analysis.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
