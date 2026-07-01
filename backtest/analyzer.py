"""Performance analysis and visualization module."""

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Analyzer:
    """Analyze backtest results and generate performance reports.

    Computes:
    - Total return, annualized return, annualized volatility
    - Sharpe ratio, Sortino ratio, Calmar ratio
    - Maximum drawdown and drawdown period
    - Win rate, profit/loss ratio
    - Benchmark comparison
    """

    def __init__(self, result: dict):
        """Initialize with backtest result.

        Args:
            result: Dict from BacktestEngine.run() with keys:
                equity_curve, benchmark_curve, returns, trades, config, symbol.
        """
        self.result = result
        self.equity = result['equity_curve']
        self.benchmark = result.get('benchmark_curve')
        self.returns = result['returns']
        self.trades = result['trades']
        self.config = result['config']
        self.symbol = result.get('symbol', 'STOCK')
        self._metrics = None

    def analyze(self) -> dict:
        """Compute all performance metrics.

        Returns:
            Dict of metric name -> value with keys:
            symbol, total_return, annualized_return, annualized_volatility,
            sharpe_ratio, sortino_ratio, max_drawdown, drawdown_start,
            drawdown_end, calmar_ratio, win_rate, profit_loss_ratio,
            total_trades, benchmark_return, benchmark_sharpe, years.
        """
        if self._metrics is not None:
            return self._metrics

        rf_daily = self.config.RISK_FREE_RATE / 252
        trading_days = len(self.equity)
        years = max(trading_days / 252, 0.01)

        # Total return
        total_return = (self.equity.iloc[-1] / self.equity.iloc[0]) - 1

        # Annualized return
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Volatility
        daily_vol = self.returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        # Sharpe ratio
        excess_returns = self.returns - rf_daily
        sharpe_ratio = float(
            (excess_returns.mean() / daily_vol * np.sqrt(252))
            if daily_vol > 0 else 0
        )

        # Max drawdown
        max_dd, dd_start, dd_end = self._max_drawdown()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Win rate and P/L ratio
        win_rate, profit_loss_ratio = self._trade_statistics()

        # Benchmark comparison
        bench_return = 0.0
        bench_sharpe = 0.0
        if self.benchmark is not None:
            bench_return = (self.benchmark.iloc[-1] / self.benchmark.iloc[0]) - 1
            bench_ret = self.benchmark.pct_change().dropna()
            bench_vol = bench_ret.std()
            if bench_vol > 0:
                bench_excess = bench_ret - rf_daily
                bench_sharpe = float(bench_excess.mean() / bench_vol * np.sqrt(252))

        # Sortino ratio
        downside = self.returns[self.returns < 0]
        downside_vol = downside.std()
        sortino_ratio = float(
            (excess_returns.mean() / downside_vol * np.sqrt(252))
            if downside_vol > 0 else 0
        )

        self._metrics = {
            'symbol': self.symbol,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_dd,
            'drawdown_start': dd_start,
            'drawdown_end': dd_end,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': len(self.trades),
            'benchmark_return': bench_return,
            'benchmark_sharpe': bench_sharpe,
            'years': years,
        }
        return self._metrics

    def _max_drawdown(self) -> Tuple[float, Optional[str], Optional[str]]:
        """Compute maximum drawdown and its period.

        Returns:
            (max_drawdown_as_negative_float, start_date_str, end_date_str)
        """
        cumulative = self.equity / self.equity.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        if max_dd >= 0:
            return 0.0, None, None

        dd_end_idx = drawdown.idxmin()
        # Find peak before drawdown end
        cumulative_before = cumulative.loc[:dd_end_idx]
        peak_idx = cumulative_before.idxmax()

        def fmt_date(idx):
            if hasattr(idx, 'date'):
                return str(idx.date())
            return str(idx)[:10]

        return max_dd, fmt_date(peak_idx), fmt_date(dd_end_idx)

    def _trade_statistics(self) -> Tuple[float, float]:
        """Compute win rate and profit/loss ratio from trade records.

        Returns:
            (win_rate, profit_loss_ratio)
        """
        if not self.trades:
            return 0.0, 0.0

        # Pair buys and sells for round-trip P&L
        sells = [t for t in self.trades if t['action'] == 'sell']
        buys = [t for t in self.trades if t['action'] == 'buy']

        profits = []
        for i in range(min(len(buys), len(sells))):
            pnl = sells[i]['cash_flow'] + buys[i]['cash_flow']  # net per round trip
            profits.append(pnl)

        if not profits:
            return 0.0, 0.0

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        win_rate = len(wins) / len(profits)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1.0

        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        return win_rate, profit_loss_ratio

    def report(self) -> str:
        """Generate formatted performance report string.

        Returns:
            Multi-line report string.
        """
        m = self.analyze()
        lines = [
            "=" * 60,
            f"  PERFORMANCE REPORT - {m['symbol']}",
            "=" * 60,
            f"  Period: {self.equity.index[0].date()} to {self.equity.index[-1].date()}",
            f"  Trading Days: {len(self.equity)} ({m['years']:.2f} years)",
            "",
            "  --- Returns ---",
            f"  Total Return:        {m['total_return']:>10.2%}",
            f"  Annualized Return:   {m['annualized_return']:>10.2%}",
            f"  Annualized Vol:      {m['annualized_volatility']:>10.2%}",
            f"  Benchmark Return:    {m['benchmark_return']:>10.2%}",
            "",
            "  --- Risk Metrics ---",
            f"  Sharpe Ratio:        {m['sharpe_ratio']:>10.2f}",
            f"  Sortino Ratio:       {m['sortino_ratio']:>10.2f}",
            f"  Calmar Ratio:        {m['calmar_ratio']:>10.2f}",
            f"  Max Drawdown:        {m['max_drawdown']:>10.2%}",
            f"  Drawdown Period:     {m['drawdown_start']} to {m['drawdown_end']}",
            "",
            "  --- Trading ---",
            f"  Total Trades:        {m['total_trades']:>10d}",
            f"  Win Rate:            {m['win_rate']:>10.2%}",
            f"  Profit/Loss Ratio:   {m['profit_loss_ratio']:>10.2f}",
            "",
            "  --- Benchmark Comparison ---",
            f"  Strategy Sharpe:     {m['sharpe_ratio']:>10.2f}",
            f"  Benchmark Sharpe:    {m['benchmark_sharpe']:>10.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot(self, save_dir: str = "./output"):
        """Generate performance charts and save to disk.

        Creates:
        1. Equity curve (strategy vs benchmark) + drawdown subplot
        2. Monthly returns heatmap
        3. Trade P&L distribution bar chart

        Args:
            save_dir: Directory to save PNG files.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Chart 1: Equity Curve + Drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Equity curves
        ax1.plot(self.equity.index, self.equity.values,
                 label=f'{self.symbol} Strategy', linewidth=1.5, color='#1f77b4')
        if self.benchmark is not None:
            ax1.plot(self.benchmark.index, self.benchmark.values,
                     label='Buy & Hold', linewidth=1, color='#ff7f0e', alpha=0.7)
        ax1.set_title(f'Equity Curve - {self.symbol}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Drawdown
        cumulative = self.equity / self.equity.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values,
                         color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=0.8)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.symbol}_equity_curve.png'), dpi=150)
        plt.close()

        # Chart 2: Monthly Returns Heatmap
        monthly_returns = self.equity.resample('ME').last().pct_change()
        if len(monthly_returns.dropna()) > 0:
            monthly_df = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values,
            }).dropna()
            pivot = monthly_df.pivot_table(
                values='return', index='year', columns='month', aggfunc='sum')

            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.8)))
                im = ax.imshow(pivot.values * 100, cmap='RdYlGn',
                               aspect='auto', vmin=-15, vmax=15)
                ax.set_xticks(range(12))
                ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                ax.set_yticks(range(len(pivot)))
                ax.set_yticklabels(pivot.index)
                ax.set_title(f'Monthly Returns (%) - {self.symbol}',
                             fontsize=14, fontweight='bold')

                for i in range(len(pivot)):
                    for j in range(12):
                        if j < len(pivot.columns):
                            month_val = pivot.columns[j]
                            if month_val in pivot.loc[pivot.index[i]].index:
                                val = pivot.iloc[i, j]
                                if not pd.isna(val):
                                    ax.text(j, i, f'{val*100:.1f}',
                                            ha='center', va='center', fontsize=8)

                plt.colorbar(im, ax=ax, label='Return (%)')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{self.symbol}_monthly_returns.png'),
                            dpi=150)
                plt.close()

        # Chart 3: Trade P&L distribution
        if self.trades:
            sells = [t for t in self.trades if t['action'] == 'sell']
            buys = [t for t in self.trades if t['action'] == 'buy']
            pnls = []
            for i in range(min(len(buys), len(sells))):
                pnl = sells[i]['cash_flow'] + buys[i]['cash_flow']
                pnls.append(pnl)

            if pnls:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#2ca02c' if p > 0 else '#d62728' for p in pnls]
                ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                ax.axhline(y=0, color='black', linewidth=0.5)
                ax.set_title(f'Trade P&L Distribution - {self.symbol}',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Trade #')
                ax.set_ylabel('Profit / Loss ($)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{self.symbol}_trade_pnl.png'),
                            dpi=150)
                plt.close()
