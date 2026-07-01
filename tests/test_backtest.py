"""Tests for backtest engine and analyzer."""

import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.analyzer import PerformanceAnalyzer


def create_test_data_and_signals(n_days: int = 252):
    """Create test OHLCV data and simple alternating buy/sell signals."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_days)))

    data = pd.DataFrame({
        'Open': prices * 1.001,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': rng.integers(50_000_000, 100_000_000, n_days),
    }, index=dates)

    signals = pd.DataFrame({'Signal': 0, 'Price': prices}, index=dates)
    signals.loc[dates[50], 'Signal'] = 1
    signals.loc[dates[100], 'Signal'] = -1
    signals.loc[dates[150], 'Signal'] = 1
    signals.loc[dates[200], 'Signal'] = -1

    return data, signals


class TestBacktestEngine:

    def test_initialization(self):
        engine = BacktestEngine(initial_capital=100000)
        assert engine.initial_capital == 100000
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005

    def test_run_returns_expected_keys(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert 'equity_curve' in result
        assert 'trades' in result
        assert 'final_equity' in result

    def test_equity_curve_has_correct_length(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert len(result['equity_curve']) == len(data)

    def test_no_signals_preserves_capital(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        data = pd.DataFrame({
            'Open': np.full(100, 100.0), 'High': np.full(100, 101.0),
            'Low': np.full(100, 99.0), 'Close': np.full(100, 100.0),
            'Volume': np.full(100, 50_000_000),
        }, index=dates)
        signals = pd.DataFrame({'Signal': 0, 'Price': np.full(100, 100.0)}, index=dates)
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert result['final_equity'] == 100000

    def test_buy_then_sell_creates_trades(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert len(result['trades']) == 2

    def test_trades_have_required_fields(self):
        data, signals = create_test_data_and_signals()
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        if result['trades']:
            trade = result['trades'][0]
            required = ['entry_date', 'exit_date', 'entry_price', 'exit_price',
                        'shares', 'profit_loss', 'profit_loss_pct', 'type']
            for field in required:
                assert field in trade, f"Missing field: {field}"

    def test_profitable_trend_produces_gain(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        prices = np.linspace(100, 150, 100)
        data = pd.DataFrame({
            'Open': prices, 'High': prices * 1.01, 'Low': prices * 0.99,
            'Close': prices, 'Volume': np.full(100, 50_000_000),
        }, index=dates)
        signals = pd.DataFrame({'Signal': 0, 'Price': prices}, index=dates)
        signals.loc[dates[10], 'Signal'] = 1
        signals.loc[dates[90], 'Signal'] = -1
        engine = BacktestEngine()
        result = engine.run(data, signals, 100000)
        assert result['final_equity'] > 100000

    def test_commission_and_slippage_applied(self):
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        prices = np.linspace(100, 110, 50)
        data = pd.DataFrame({
            'Open': prices, 'High': prices * 1.01, 'Low': prices * 0.99,
            'Close': prices, 'Volume': np.full(50, 50_000_000),
        }, index=dates)
        signals = pd.DataFrame({'Signal': 0, 'Price': prices}, index=dates)
        signals.loc[dates[5], 'Signal'] = 1
        signals.loc[dates[45], 'Signal'] = -1

        engine_with = BacktestEngine(commission_rate=0.001, slippage_rate=0.0005)
        result_with = engine_with.run(data, signals, 100000)

        engine_no = BacktestEngine(commission_rate=0.0, slippage_rate=0.0)
        result_no = engine_no.run(data, signals, 100000)

        assert result_with['final_equity'] < result_no['final_equity']


class TestPerformanceAnalyzer:

    def _make_result(self, equity_series, trades=None):
        return {
            'equity_curve': equity_series,
            'trades': trades or [],
            'final_equity': equity_series.iloc[-1],
        }

    def test_annualized_return(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert 18 < metrics['annualized_return'] < 22

    def test_max_drawdown(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        equity = pd.Series(
            [100000] * 50 + list(np.linspace(100000, 90000, 50)),
            index=dates
        )
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['max_drawdown'] == pytest.approx(10.0, abs=0.5)

    def test_no_drawdown_when_always_up(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        equity = pd.Series(np.linspace(100000, 150000, 100), index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['max_drawdown'] == 0.0

    def test_sharpe_ratio_positive(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['sharpe_ratio'] > 0

    def test_win_rate(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.cumsum(np.random.randn(252) * 100), index=dates)
        trades = [
            {'profit_loss': 1000, 'type': 'signal', 'entry_date': dates[0], 'exit_date': dates[10],
             'entry_price': 100, 'exit_price': 110, 'shares': 100, 'profit_loss_pct': 10.0},
            {'profit_loss': -500, 'type': 'signal', 'entry_date': dates[20], 'exit_date': dates[30],
             'entry_price': 100, 'exit_price': 95, 'shares': 100, 'profit_loss_pct': -5.0},
            {'profit_loss': 300, 'type': 'signal', 'entry_date': dates[40], 'exit_date': dates[50],
             'entry_price': 100, 'exit_price': 103, 'shares': 100, 'profit_loss_pct': 3.0},
        ]
        result = self._make_result(equity, trades)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['win_rate'] == pytest.approx(66.67, abs=0.1)

    def test_profit_loss_ratio(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        trades = [
            {'profit_loss': 1000, 'type': 'signal', 'entry_date': dates[0], 'exit_date': dates[10],
             'entry_price': 100, 'exit_price': 110, 'shares': 100, 'profit_loss_pct': 10.0},
            {'profit_loss': -200, 'type': 'signal', 'entry_date': dates[20], 'exit_date': dates[30],
             'entry_price': 100, 'exit_price': 98, 'shares': 100, 'profit_loss_pct': -2.0},
        ]
        result = self._make_result(equity, trades)
        metrics = PerformanceAnalyzer.analyze(result)
        assert metrics['profit_loss_ratio'] == pytest.approx(5.0, abs=0.1)

    def test_all_metrics_present(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        result = self._make_result(equity)
        metrics = PerformanceAnalyzer.analyze(result)
        required = ['annualized_return', 'max_drawdown', 'sharpe_ratio',
                    'win_rate', 'profit_loss_ratio', 'total_trades',
                    'final_equity', 'total_return']
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
