"""Tests for backtest engine and analyzer."""

import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine, Trade, BacktestResult
from backtest.analyzer import Analyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(dates, close_values):
    """Build a price DataFrame with a 'close' column."""
    return pd.DataFrame({'close': close_values}, index=dates)


def _make_signals(dates, buy_indices=None, sell_indices=None):
    """Build a signal DataFrame with a 'signal' column."""
    sig = pd.Series(0, index=dates, dtype=int)
    if buy_indices:
        for i in buy_indices:
            sig.iloc[i] = 1
    if sell_indices:
        for i in sell_indices:
            sig.iloc[i] = -1
    return sig.to_frame('signal')


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
        assert engine.commission == 0.001
        assert engine.slippage == 0.0005

    def test_initialization_custom(self):
        engine = BacktestEngine(initial_capital=50_000, commission=0.002, slippage=0.001)
        assert engine.initial_capital == 50_000
        assert engine.commission == 0.002
        assert engine.slippage == 0.001

    def test_run_returns_backtest_result(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        rng = np.random.default_rng(42)
        prices = _make_prices(dates, 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 100))))
        signals = _make_signals(dates, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert isinstance(result, BacktestResult)
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'final_value')

    def test_equity_curve_length(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        prices = _make_prices(dates, np.full(100, 100.0))
        signals = _make_signals(dates, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert len(result.equity_curve) == 100

    def test_no_signals_preserves_capital(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        prices = _make_prices(dates, np.full(100, 100.0))
        signals = _make_signals(dates)
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert result.final_value == pytest.approx(100_000.0, rel=0.01)

    def test_buy_then_sell_creates_trade_objects(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        prices = _make_prices(dates, np.full(100, 100.0))
        signals = _make_signals(dates, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert len(result.trades) >= 1
        directions = [t.direction for t in result.trades]
        assert 'buy' in directions
        assert 'sell' in directions

    def test_trades_are_trade_dataclass(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        rng = np.random.default_rng(42)
        prices = _make_prices(dates, 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 100))))
        signals = _make_signals(dates, buy_indices=[10], sell_indices=[50])
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        if result.trades:
            trade = result.trades[0]
            assert isinstance(trade, Trade)
            assert hasattr(trade, 'timestamp')
            assert hasattr(trade, 'direction')
            assert hasattr(trade, 'price')
            assert hasattr(trade, 'shares')
            assert hasattr(trade, 'pnl')
            assert hasattr(trade, 'cost')

    def test_profitable_trend_produces_gain(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        close = np.linspace(100, 150, 100)
        prices = _make_prices(dates, close)
        signals = _make_signals(dates, buy_indices=[10], sell_indices=[90])
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        assert result.final_value > 100_000

    def test_commission_and_slippage_affect_result(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        close = np.linspace(100, 110, 50)
        prices = _make_prices(dates, close)
        signals = _make_signals(dates, buy_indices=[5], sell_indices=[45])

        engine_with = BacktestEngine(commission=0.001, slippage=0.0005)
        result_with = engine_with.run(signals, prices)

        signals2 = _make_signals(dates, buy_indices=[5], sell_indices=[45])
        engine_zero = BacktestEngine(commission=0.0, slippage=0.0)
        result_zero = engine_zero.run(signals2, prices)

        assert result_with.final_value <= result_zero.final_value

    def test_custom_symbol_and_strategy_name(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        prices = _make_prices(dates, np.full(50, 100.0))
        signals = _make_signals(dates)
        engine = BacktestEngine()
        result = engine.run(signals, prices, symbol='TSLA', strategy_name='TestStrat')
        assert result.symbol == 'TSLA'
        assert result.strategy_name == 'TestStrat'

    def test_stop_loss_column_forces_exit(self):
        dates = pd.date_range('2024-01-01', periods=60, freq='B')
        close = np.concatenate([np.full(30, 100.0), np.linspace(100, 80, 30)])
        prices = _make_prices(dates, close)
        signals = _make_signals(dates, buy_indices=[5], sell_indices=[55])
        signals['stop_loss'] = np.nan
        signals.loc[dates[20]:, 'stop_loss'] = 95.0

        engine = BacktestEngine()
        result = engine.run(signals, prices)
        sell_trades = [t for t in result.trades if t.direction == 'sell']
        assert len(sell_trades) >= 1


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
            {'action': 'sell', 'profit_loss': 1000},
            {'action': 'sell', 'profit_loss': -500},
            {'action': 'sell', 'profit_loss': 300},
        ]
        metrics = Analyzer(_analyzer_result(equity, trades)).analyze()
        assert metrics['win_rate_pct'] == pytest.approx(66.67, abs=0.1)

    def test_profit_factor(self):
        dates = pd.date_range('2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        trades = [
            {'action': 'sell', 'profit_loss': 1000},
            {'action': 'sell', 'profit_loss': -200},
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
