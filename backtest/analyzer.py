"""Performance analysis for backtest results."""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from config import RISK_FREE_RATE, BENCHMARK_SYMBOL

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Calculate comprehensive performance metrics from backtest results."""

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        """Initialize analyzer.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%).
        """
        self.risk_free_rate = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: list = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute all performance metrics.

        Args:
            equity_curve: Portfolio equity over time.
            trades: List of Trade objects.
            benchmark_returns: Daily returns of benchmark index.

        Returns:
            Dictionary of metric names to values.
        """
        if len(equity_curve) < 2:
            return {'error': 'Not enough data points'}

        metrics = {}

        # Returns
        daily_returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics['total_return'] = total_return

        # Annualized return
        trading_days = len(daily_returns)
        years = trading_days / 252
        if years > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        metrics['annualized_return'] = annualized_return

        # Sharpe ratio
        excess_returns = daily_returns - self.rf_daily
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() - self.rf_daily) / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        metrics['sharpe_ratio'] = sharpe

        # Max drawdown
        max_dd, dd_duration, dd_start, dd_end = self._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = dd_duration

        # Trade statistics
        if trades and len(trades) > 0:
            sell_trades = [t for t in trades if t.direction == 'sell']
            if sell_trades:
                wins = [t for t in sell_trades if t.pnl > 0]
                losses = [t for t in sell_trades if t.pnl <= 0]

                metrics['total_trades'] = len(sell_trades)
                metrics['winning_trades'] = len(wins)
                metrics['losing_trades'] = len(losses)
                metrics['win_rate'] = len(wins) / len(sell_trades) if sell_trades else 0

                avg_win = np.mean([t.pnl for t in wins]) if wins else 0
                avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
                metrics['avg_win'] = avg_win
                metrics['avg_loss'] = avg_loss
                metrics['profit_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
                metrics['total_pnl'] = sum(t.pnl for t in sell_trades)
            else:
                metrics['total_trades'] = 0
                metrics['win_rate'] = 0
                metrics['profit_loss_ratio'] = 0
        else:
            metrics['total_trades'] = 0
            metrics['win_rate'] = 0
            metrics['profit_loss_ratio'] = 0

        # Calmar ratio
        if max_dd > 0:
            metrics['calmar_ratio'] = annualized_return / max_dd
        else:
            metrics['calmar_ratio'] = float('inf') if annualized_return > 0 else 0

        # Alpha and Beta vs benchmark
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            aligned_bench = benchmark_returns.reindex(daily_returns.index).dropna()
            aligned_strat = daily_returns.reindex(aligned_bench.index)
            if len(aligned_bench) > 1:
                beta, alpha = self._calculate_alpha_beta(aligned_strat, aligned_bench)
                metrics['alpha'] = alpha
                metrics['beta'] = beta
                metrics['information_ratio'] = (
                    (aligned_strat.mean() - aligned_bench.mean()) / aligned_strat.std() * np.sqrt(252)
                    if aligned_strat.std() > 0 else 0
                )

        # Sortino ratio (downside deviation only)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            metrics['sortino_ratio'] = (
                (daily_returns.mean() - self.rf_daily) / downside.std() * np.sqrt(252)
            )
        else:
            metrics['sortino_ratio'] = 0.0

        # Value at Risk (95%)
        metrics['var_95'] = daily_returns.quantile(0.05)
        metrics['cvar_95'] = daily_returns[daily_returns <= metrics['var_95']].mean()

        return metrics

    def _calculate_max_drawdown(
        self, equity: pd.Series
    ) -> Tuple[float, int, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its duration.

        Args:
            equity: Equity curve series.

        Returns:
            (max_drawdown_fraction, duration_days, start_date, end_date)
        """
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max

        max_dd = abs(drawdown.min())

        # Find drawdown duration
        max_dd_idx = drawdown.idxmin()
        peak_idx = equity[:max_dd_idx].idxmax() if max_dd_idx else equity.index[0]

        # Find recovery point
        recovery_idx = max_dd_idx
        peak_value = equity[peak_idx]
        for idx in equity.index[equity.index.get_loc(max_dd_idx):]:
            if equity[idx] >= peak_value:
                recovery_idx = idx
                break

        duration = (recovery_idx - peak_idx).days

        return max_dd, duration, peak_idx, recovery_idx

    def _calculate_alpha_beta(
        self, strategy_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate alpha and beta using CAPM regression.

        Args:
            strategy_returns: Strategy daily returns.
            benchmark_returns: Benchmark daily returns.

        Returns:
            (annualized_alpha, beta)
        """
        common = strategy_returns.index.intersection(benchmark_returns.index)
        x = benchmark_returns[common].values
        y = strategy_returns[common].values

        if len(x) < 2:
            return 0.0, 1.0

        beta, alpha, _, _, _ = stats.linregress(x, y)

        # Annualize alpha
        alpha_annual = (1 + alpha) ** 252 - 1

        return alpha_annual, beta

    def monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly return series."""
        return equity_curve.resample('ME').last().pct_change()

    def yearly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate yearly return series."""
        return equity_curve.resample('YE').last().pct_change()

    def summary(self, metrics: Dict[str, float]) -> str:
        """Generate a text summary of performance metrics."""
        lines = [
            "=" * 60,
            "PERFORMANCE SUMMARY",
            "=" * 60,
            f"Total Return:       {metrics.get('total_return', 0):>10.2%}",
            f"Annualized Return:  {metrics.get('annualized_return', 0):>10.2%}",
            f"Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>10.2f}",
            f"Sortino Ratio:      {metrics.get('sortino_ratio', 0):>10.2f}",
            f"Max Drawdown:       {metrics.get('max_drawdown', 0):>10.2%}",
            f"Calmar Ratio:       {metrics.get('calmar_ratio', 0):>10.2f}",
            f"Win Rate:           {metrics.get('win_rate', 0):>10.2%}",
            f"Profit/Loss Ratio:  {metrics.get('profit_loss_ratio', 0):>10.2f}",
            f"Total Trades:       {metrics.get('total_trades', 0):>10.0f}",
        ]
        if 'alpha' in metrics:
            lines.append(f"Alpha:              {metrics['alpha']:>10.2%}")
            lines.append(f"Beta:               {metrics['beta']:>10.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)


# Backward compatibility alias
Analyzer = PerformanceAnalyzer
