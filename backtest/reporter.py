"""Report generator for backtest results — console output, charts, and HTML."""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import REPORT_DIR

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates backtest reports in console, chart, and HTML formats."""

    def __init__(self, output_dir: str = REPORT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, result: dict, metrics: dict, data: pd.DataFrame,
                 signals: pd.DataFrame, symbol: str, strategy_name: str) -> str:
        """Generate full report — console, charts, HTML. Returns HTML path."""
        self._print_console_report(metrics, symbol, strategy_name)
        chart_paths = self._generate_charts(result, data, signals, symbol, strategy_name)
        html_path = self._generate_html_report(metrics, result, chart_paths,
                                                symbol, strategy_name)
        return html_path

    def _print_console_report(self, metrics: dict, symbol: str, strategy_name: str):
        print("\n" + "=" * 60)
        print(f"  BACKTEST REPORT")
        print(f"  Symbol: {symbol}  |  Strategy: {strategy_name}")
        print("=" * 60)
        print(f"  Final Equity:         ${metrics['final_equity']:>12,.2f}")
        print(f"  Total Return:         {metrics['total_return']:>11.2f}%")
        print(f"  Annualized Return:    {metrics['annualized_return']:>11.2f}%")
        print(f"  Max Drawdown:         {metrics['max_drawdown']:>11.2f}%")
        print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>11.2f}")
        print(f"  Win Rate:             {metrics['win_rate']:>11.2f}%")
        print(f"  Profit/Loss Ratio:    {metrics['profit_loss_ratio']:>11.2f}")
        print(f"  Total Trades:         {metrics['total_trades']:>11}")
        print("=" * 60 + "\n")

    def _generate_charts(self, result: dict, data: pd.DataFrame,
                         signals: pd.DataFrame, symbol: str,
                         strategy_name: str) -> dict:
        equity = result['equity_curve']
        trades = result['trades']
        chart_paths = {}
        plt.style.use('ggplot')

        # 1. Equity curve
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity.index, equity.values, 'b-', linewidth=1.5, label='Portfolio Value')
        ax.axhline(y=equity.iloc[0], color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax.set_title(f'{symbol} - {strategy_name} - Equity Curve', fontsize=14)
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        path = os.path.join(self.output_dir, f'{symbol}_{strategy_name}_equity.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        chart_paths['equity'] = path

        # 2. Drawdown curve
        fig, ax = plt.subplots(figsize=(12, 4))
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max * 100
        ax.fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.3)
        ax.plot(drawdowns.index, drawdowns.values, 'r-', linewidth=0.8)
        ax.set_title(f'{symbol} - {strategy_name} - Drawdown', fontsize=14)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        path = os.path.join(self.output_dir, f'{symbol}_{strategy_name}_drawdown.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        chart_paths['drawdown'] = path

        # 3. Price with trade markers
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], 'k-', linewidth=1, alpha=0.7, label='Close Price')
        for trade in trades:
            if trade['type'] in ('signal', 'stop_loss', 'take_profit_partial', 'close_out'):
                entry_dt = pd.Timestamp(trade['entry_date'])
                exit_dt = pd.Timestamp(trade['exit_date'])
                if entry_dt in data.index:
                    ax.scatter(entry_dt, data.loc[entry_dt, 'Close'],
                              color='green', marker='^', s=80, zorder=5)
                if exit_dt in data.index:
                    ax.scatter(exit_dt, data.loc[exit_dt, 'Close'],
                              color='red', marker='v', s=80, zorder=5)
        ax.set_title(f'{symbol} - {strategy_name} - Trade Signals', fontsize=14)
        ax.set_ylabel('Price ($)')
        ax.set_xlabel('Date')
        ax.legend(['Close Price', 'Buy', 'Sell'] if trades else ['Close Price'])
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        path = os.path.join(self.output_dir, f'{symbol}_{strategy_name}_signals.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        chart_paths['signals'] = path

        return chart_paths

    def _generate_html_report(self, metrics: dict, result: dict,
                               chart_paths: dict, symbol: str,
                               strategy_name: str) -> str:
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {symbol} - {strategy_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-card .label {{ font-size: 0.85em; color: #666; }}
        .metric-card .value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .trades-table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        .trades-table th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
        .trades-table td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
        .trades-table tr:nth-child(even) {{ background: #f9f9f9; }}
        .chart {{ max-width: 100%; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <p><strong>Symbol:</strong> {symbol} | <strong>Strategy:</strong> {strategy_name}</p>

    <h2>Performance Metrics</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="label">Final Equity</div>
            <div class="value">${metrics['final_equity']:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Return</div>
            <div class="value {'positive' if metrics['total_return'] >= 0 else 'negative'}">{metrics['total_return']:+.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Annualized Return</div>
            <div class="value {'positive' if metrics['annualized_return'] >= 0 else 'negative'}">{metrics['annualized_return']:+.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">-{metrics['max_drawdown']:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Sharpe Ratio</div>
            <div class="value">{metrics['sharpe_ratio']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Win Rate</div>
            <div class="value">{metrics['win_rate']:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Profit/Loss Ratio</div>
            <div class="value">{metrics['profit_loss_ratio']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Trades</div>
            <div class="value">{metrics['total_trades']}</div>
        </div>
    </div>

    <h2>Equity Curve</h2>
    <img class="chart" src="{os.path.basename(chart_paths.get('equity', ''))}" alt="Equity Curve">

    <h2>Drawdown</h2>
    <img class="chart" src="{os.path.basename(chart_paths.get('drawdown', ''))}" alt="Drawdown Curve">

    <h2>Trade Signals</h2>
    <img class="chart" src="{os.path.basename(chart_paths.get('signals', ''))}" alt="Trade Signals">

    <h2>Trade List</h2>
    <table class="trades-table">
        <tr><th>#</th><th>Entry Date</th><th>Exit Date</th><th>Entry Price</th><th>Exit Price</th><th>P&L</th><th>Return %</th><th>Type</th></tr>
"""
        for i, trade in enumerate(result['trades'], 1):
            pnl_class = 'positive' if trade['profit_loss'] >= 0 else 'negative'
            html += (
                f"        <tr>"
                f"<td>{i}</td>"
                f"<td>{str(trade['entry_date'])[:10]}</td>"
                f"<td>{str(trade['exit_date'])[:10]}</td>"
                f"<td>${trade['entry_price']:.2f}</td>"
                f"<td>${trade['exit_price']:.2f}</td>"
                f"<td class='{pnl_class}'>${trade['profit_loss']:,.2f}</td>"
                f"<td class='{pnl_class}'>{trade['profit_loss_pct']:+.2f}%</td>"
                f"<td>{trade['type']}</td>"
                f"</tr>\n"
            )

        html += """    </table>
</body>
</html>"""

        path = os.path.join(self.output_dir, f'{symbol}_{strategy_name}_report.html')
        with open(path, 'w') as f:
            f.write(html)
        logger.info(f"HTML report saved to {path}")
        return path
