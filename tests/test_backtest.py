"""Tests for backtest engine and analyzer."""

import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.analyzer import Analyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(dates, close_values):
    """Build an OHLCV DataFrame with capitalized columns."""
    n = len(close_values)
    return pd.DataFrame({
        'Open': np.array(close_values) * 0.999,
        'High': np.array(close_values) * 1.005,
        'Low': np.array(close_values) * 0.995,
        'Close': np.array(close_values),
        'Volume': np.full(n, 50_000_000),
    }, index=dates)


def _make_signals(dates, close_values, buy_indices=None, sell_indices=None):
    """Build a signal DataFrame with Signal and Price columns."""
    n = len(dates)
    sig = pd.Series(0, index=dates, dtype=int)
    if buy_indices:
        for i in buy_indices:
            sig.iloc[i] = 1
    if sell_indices:
        for i in sell_indices:
            sig.iloc[i] = -1
    signals = pd.DataFrame({'Signal': sig, 'Price': close_values}, index=dates)
    return signals


def _analyzer_result(equity_series, trades=None):
    """Build the dict that Analyzer expects."""
    return {
        'equity_curve': equity_series,
        'trades': trades or [],
    }


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def test_initialization_defaults(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 100_000.0
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005

    def test_initialization_custom(self):
        engine = BacktestEngine(initial_capital=50_000, commission_rate=0.002, slippage_rate=0.001)
        assert engine.initial_capital == 50_000
        assert engine.commission_rate == 0.002
        assert engine.slippage_rate == 0.001

    def test_run_returns_dict(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 100)))
        data = _make_data(dates, close)
        signals = _make_signals(dates, close, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(data, signals)
        assert isinstance(result, dict)
        assert 'equity_curve' in result
        assert 'trades' in result
        assert 'final_equity' in result

    def test_equity_curve_length(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        close = np.full(100, 100.0)
        data = _make_data(dates, close)
        signals = _make_signals(dates, close, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(data, signals)
        assert len(result['equity_curve']) == 100

    def test_no_signals_preserves_capital(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        close = np.full(100, 100.0)
        data = _make_data(dates, close)
        signals = _make_signals(dates, close)
        engine = BacktestEngine()
        result = engine.run(data, signals)
        assert result['final_equity'] == pytest.approx(100_000.0, rel=0.01)

    def test_buy_then_sell_creates_trade_dicts(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        close = np.full(100, 100.0)
        data = _make_data(dates, close)
        signals = _make_signals(dates, close, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(data, signals)
        assert len(result['trades']) >= 1
        # Check that trades are dicts with expected keys
        for t in result['trades']:
            assert isinstance(t, dict)
            assert 'entry_date' in t
            assert 'exit_date' in t
            assert 'entry_price' in t
            assert 'exit_price' in t
            assert 'shares' in t
            assert 'profit_loss' in t
            assert 'profit_loss_pct' in t
            assert 'type' in t

    def test_profitable_trend_produces_gain(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        close = np.linspace(100, 150, 100)
        data = _make_data(dates, close)
        signals = _make_signals(dates, close, buy_indices=[10], sell_indices=[90])
        engine = BacktestEngine()
        result = engine.run(data, signals)
        assert result['final_equity'] > 100_000

    def test_commission_and_slippage_affect_result(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        close = np.linspace(100, 110, 50)
        data = _make_data(dates, close)
        signals1 = _make_signals(dates, close, buy_indices=[5], sell_indices=[45])
        signals2 = _make_signals(dates, close, buy_indices=[5], sell_indices=[45])

        engine_with = BacktestEngine(commission_rate=0.001, slippage_rate=0.0005)
        result_with = engine_with.run(data, signals1)

        engine_zero = BacktestEngine(commission_rate=0.0, slippage_rate=0.0)
        result_zero = engine_zero.run(data, signals2)

        assert result_with['final_equity'] <= result_zero['final_equity']

    def test_capital_override(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        close = np.full(50, 100.0)
        data = _make_data(dates, close)
        signals = _make_signals(dates, close)
        engine = BacktestEngine()
        result = engine.run(data, signals, capital=200_000)
        # No trades executed, so final equity should equal capital
        assert result['final_equity'] == pytest.approx(200_000.0, rel=0.01)

    def test_stop_loss_triggers_exit(self):
        """When price drops below stop-loss threshold, position should exit."""
        dates = pd.date_range('2024-01-01', periods=60, freq='B')
        # Price drops significantly after buy
        close = np.concatenate([
            np.full(30, 100.0),
            np.linspace(100, 70, 30),  # Drops 30% — far below 5% stop loss
        ])
        data = _make_data(dates, close)
        signals = _make_signals(dates, close, buy_indices=[5])
        engine = BacktestEngine()
        result = engine.run(data, signals)
        # Should have at least one trade (entry + forced exit)
        assert len(result['trades']) >= 1

    def test_take_profit_triggers_exit(self):
        """When price rises above take-profit threshold, position should exit."""
        dates = pd.date_range('2024-01-01', periods=60, freq='B')
        # Price rises significantly after buy
        close = np.concatenate([
            np.full(30, 100.0),
            np.linspace(100, 130, 30),  # Rises 30% — far above 15% take profit
        ])
        data = _make_data(dates, close)
        signals = _make_signals(dates, close, buy_indices=[5])
        engine = BacktestEngine()
        result = engine.run(data, signals)
        # Should have at least one trade (entry + forced exit)
        assert len(result['trades']) >= 1


# ---------------------------------------------------------------------------
# Analyzer tests (accepts dict with equity_curve + trades)
# ---------------------------------------------------------------------------

class TestAnalyzer:

    def test_annual_return(self):
        dates = pd.date_range('2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        metrics = Analyzer(_analyzer_result(equity)).analyze()
        assert 18 < metrics['annual_return_pct'] < 22

    def test_max_drawdown(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        equity = pd.Series(
            [100000] * 50 + list(np.linspace(100000, 90000, 50)),
            index=dates
        )
        metrics = Analyzer(_analyzer_result(equity)).analyze()
        assert metrics['max_drawdown_pct'] == pytest.approx(10.0, abs=0.5)

    def test_no_drawdown_when_always_up(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        equity = pd.Series(np.linspace(100000, 150000, 100), index=dates)
        metrics = Analyzer(_analyzer_result(equity)).analyze()
        assert metrics['max_drawdown_pct'] == 0.0

    def test_sharpe_ratio_positive(self):
        dates = pd.date_range('2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        metrics = Analyzer(_analyzer_result(equity)).analyze()
        assert metrics['sharpe_ratio'] > 0

    def test_win_rate(self):
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.cumsum(np.random.randn(252) * 100), index=dates)
        trades = [
            {'profit_loss': 1000},
            {'profit_loss': -500},
            {'profit_loss': 300},
        ]
        metrics = Analyzer(_analyzer_result(equity, trades)).analyze()
        assert metrics['win_rate_pct'] == pytest.approx(66.67, abs=0.1)

    def test_profit_factor(self):
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        trades = [
            {'profit_loss': 1000},
            {'profit_loss': -200},
        ]
        metrics = Analyzer(_analyzer_result(equity, trades)).analyze()
        assert metrics['profit_factor'] == pytest.approx(5.0, abs=0.1)

    def test_all_metrics_present(self):
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        metrics = Analyzer(_analyzer_result(equity)).analyze()
        required = ['annual_return_pct', 'max_drawdown_pct', 'sharpe_ratio',
                    'win_rate_pct', 'profit_factor', 'total_trades',
                    'final_equity', 'total_return_pct']
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
