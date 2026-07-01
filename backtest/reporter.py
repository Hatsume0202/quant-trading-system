"""Report generation: charts and markdown reports."""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate visual and text reports from backtest results."""

    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Set matplotlib style
        plt.style.use('ggplot')

    def generate(
        self,
        equity_curve: pd.Series,
        metrics: Dict[str, float],
        trades: list = None,
        benchmark_equity: Optional[pd.Series] = None,
        strategy_name: str = "",
        symbol: str = "",
    ) -> str:
        """Generate complete report with charts and markdown.

        Args:
            equity_curve: Portfolio equity over time.
            metrics: Performance metrics from analyzer.
            trades: List of Trade objects.
            benchmark_equity: Benchmark equity curve for comparison.
            strategy_name: Name of the strategy.
            symbol: Symbol traded.

        Returns:
            Path to the generated markdown report.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{strategy_name}_{symbol}" if strategy_name else "backtest"

        # Generate charts
        equity_path = self._plot_equity_curve(
            equity_curve, benchmark_equity, strategy_name, name, timestamp
        )
        drawdown_path = self._plot_drawdown(equity_curve, name, timestamp)
        heatmap_path = self._plot_monthly_heatmap(equity_curve, name, timestamp)

        # Generate markdown report
        md_path = self._generate_markdown_report(
            metrics, trades, strategy_name, symbol,
            equity_path, drawdown_path, heatmap_path, timestamp
        )

        logger.info(f"Report generated: {md_path}")
        return md_path

    def _plot_equity_curve(
        self, equity: pd.Series, benchmark: Optional[pd.Series],
        strategy_name: str, name: str, timestamp: str
    ) -> str:
        """Plot equity curve with optional benchmark comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize to 100
        equity_norm = equity / equity.iloc[0] * 100
        ax.plot(equity_norm.index, equity_norm.values, label=f'{strategy_name} Equity', linewidth=1.5)

        if benchmark is not None and len(benchmark) > 0:
            bench_norm = benchmark / benchmark.iloc[0] * 100
            common_idx = equity_norm.index.intersection(bench_norm.index)
            ax.plot(common_idx, bench_norm[common_idx].values,
                    label='S&P 500 Benchmark', linewidth=1, alpha=0.7, linestyle='--')

        ax.set_title(f'Equity Curve — {strategy_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value (Base=100)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        path = os.path.join(self.output_dir, f"{name}_equity_{timestamp}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_drawdown(self, equity: pd.Series, name: str, timestamp: str) -> str:
        """Plot drawdown curve."""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        path = os.path.join(self.output_dir, f"{name}_drawdown_{timestamp}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def _plot_monthly_heatmap(self, equity: pd.Series, name: str, timestamp: str) -> str:
        """Plot monthly returns heatmap."""
        monthly = equity.resample('ME').last().pct_change().dropna()

        # Build pivot table: rows=years, cols=months
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100
        })
        pivot = monthly_df.pivot_table(
            values='return', index='year', columns='month', aggfunc='sum'
        )

        if pivot.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, len(pivot) * 0.8 + 2))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # Annotate cells
        for i in range(len(pivot)):
            for j in range(12):
                val = pivot.iloc[i, j] if j < len(pivot.columns) else np.nan
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=8)

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index)
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Return (%)')

        path = os.path.join(self.output_dir, f"{name}_heatmap_{timestamp}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    def _generate_markdown_report(
        self, metrics: Dict[str, float], trades: list,
        strategy_name: str, symbol: str,
        equity_path: str, drawdown_path: str, heatmap_path: str,
        timestamp: str
    ) -> str:
        """Generate markdown report with all metrics and embedded charts."""
        lines = [
            f"# Backtest Report — {strategy_name}",
            "",
            f"**Symbol:** {symbol}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
            "---",
            "",
            "## Performance Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Return | {metrics.get('total_return', 0):.2%} |",
            f"| Annualized Return | {metrics.get('annualized_return', 0):.2%} |",
            f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |",
            f"| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |",
            f"| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |",
            f"| Drawdown Duration | {metrics.get('drawdown_duration', 0)} days |",
            f"| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |",
            f"| Win Rate | {metrics.get('win_rate', 0):.2%} |",
            f"| Profit/Loss Ratio | {metrics.get('profit_loss_ratio', 0):.2f} |",
            f"| Total Trades | {metrics.get('total_trades', 0):.0f} |",
            f"| Total P&L | ${metrics.get('total_pnl', 0):,.2f} |",
        ]

        if 'alpha' in metrics:
            lines.append(f"| Alpha (annual) | {metrics['alpha']:.2%} |")
            lines.append(f"| Beta | {metrics['beta']:.2f} |")

        if 'var_95' in metrics:
            lines.append(f"| VaR (95%) | {metrics['var_95']:.2%} |")
            lines.append(f"| CVaR (95%) | {metrics['cvar_95']:.2%} |")

        lines.extend([
            "",
            "---",
            "",
            "## Charts",
            "",
            "### Equity Curve",
            f"![Equity Curve]({os.path.basename(equity_path)})",
            "",
            "### Drawdown",
            f"![Drawdown]({os.path.basename(drawdown_path)})",
        ])

        if heatmap_path:
            lines.extend([
                "",
                "### Monthly Returns",
                f"![Monthly Returns]({os.path.basename(heatmap_path)})",
            ])

        # Trade list
        if trades and len(trades) > 0:
            lines.extend([
                "",
                "---",
                "",
                "## Trade List",
                "",
                "| Date | Symbol | Direction | Price | Shares | P&L |",
                "|------|--------|-----------|-------|--------|-----|",
            ])
            for t in trades[-20:]:  # Last 20 trades
                lines.append(
                    f"| {t.timestamp.date()} | {t.symbol} | {t.direction} | "
                    f"${t.price:.2f} | {t.shares} | ${t.pnl:,.2f} |"
                )
            if len(trades) > 20:
                lines.append(f"| ... | ... | ... | ... | ... | ... |")
                lines.append(f"| *Showing last 20 of {len(trades)} trades* | | | | | |")

        lines.extend([
            "",
            "---",
            "",
            "*Report generated by QuantTradingSystem. Past performance does not guarantee future results.*",
        ])

        path = os.path.join(self.output_dir, f"{strategy_name}_{symbol}_{timestamp}_report.md")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

        return path
