"""Tests for backtest engine and analyzer."""

import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.analyzer import Analyzer
from backtest.broker import Broker
from config import Config


# ---------------------------------------------------------------------------
# Helper: a minimal strategy that returns a signal Series
# ---------------------------------------------------------------------------

class _MockStrategy:
    """Minimal strategy wrapper around a pre-built signal Series."""

    def __init__(self, signal_series: pd.Series):
        self.signal_series = signal_series

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self.signal_series.reindex(data.index, fill_value=0)


def _make_data(dates, close):
    """Build OHLCV DataFrame with lowercase columns."""
    return pd.DataFrame({
        'open': close * 1.001,
        'high': close * 1.01,
        'low': close * 0.99,
        'close': close,
        'volume': np.full(len(close), 50_000_000),
    }, index=dates)


def _make_signal_series(dates, buy_indices=None, sell_indices=None):
    """Build signal Series (1=buy, -1=sell, 0=hold)."""
    sig = pd.Series(0, index=dates, dtype=int)
    if buy_indices:
        for idx in buy_indices:
            sig.iloc[idx] = 1
    if sell_indices:
        for idx in sell_indices:
            sig.iloc[idx] = -1
    return sig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    def test_initialization_with_default_config(self):
        engine = BacktestEngine()
        assert engine.config.INITIAL_CAPITAL == 100_000.0

    def test_initialization_with_custom_config(self):
        cfg = Config(INITIAL_CAPITAL=50_000.0, COMMISSION_RATE=0.001)
        engine = BacktestEngine(config=cfg)
        assert engine.config.INITIAL_CAPITAL == 50_000.0
        assert engine.config.COMMISSION_RATE == 0.001

    def test_run_returns_expected_keys(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 100)))
        data = _make_data(dates, close)
        sig = _make_signal_series(dates, buy_indices=[10], sell_indices=[50])
        strategy = _MockStrategy(sig)
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        assert 'equity_curve' in result
        assert 'trades' in result
        assert 'returns' in result
        assert 'symbol' in result
        assert result['symbol'] == 'STOCK'

    def test_equity_curve_has_correct_length(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        close = np.full(100, 100.0)
        data = _make_data(dates, close)
        sig = _make_signal_series(dates, buy_indices=[10], sell_indices=[50])
        strategy = _MockStrategy(sig)
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        assert len(result['equity_curve']) == len(data)

    def test_no_signals_preserves_capital(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        close = np.full(100, 100.0)
        data = _make_data(dates, close)
        strategy = _MockStrategy(pd.Series(0, index=dates, dtype=int))
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        final_equity = result['equity_curve'].iloc[-1]
        assert final_equity == pytest.approx(100_000.0, rel=0.01)

    def test_buy_then_sell_creates_trades(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        close = np.full(100, 100.0)
        data = _make_data(dates, close)
        sig = _make_signal_series(dates, buy_indices=[10], sell_indices=[50])
        strategy = _MockStrategy(sig)
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        # A buy then sell should produce at least one trade record
        assert len(result['trades']) >= 1
        # Sells and buys should be recorded
        actions = [t['action'] for t in result['trades']]
        assert 'buy' in actions
        assert 'sell' in actions

    def test_trades_have_required_fields(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 100)))
        data = _make_data(dates, close)
        sig = _make_signal_series(dates, buy_indices=[10], sell_indices=[50])
        strategy = _MockStrategy(sig)
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        if result['trades']:
            trade = result['trades'][0]
            required = ['date', 'symbol', 'action', 'price', 'shares',
                        'commission', 'cash_flow']
            for field in required:
                assert field in trade, f"Missing field: {field}"

    def test_profitable_trend_produces_gain(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        close = np.linspace(100, 150, 100)
        data = _make_data(dates, close)
        sig = _make_signal_series(dates, buy_indices=[10], sell_indices=[90])
        strategy = _MockStrategy(sig)
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        final_equity = result['equity_curve'].iloc[-1]
        assert final_equity > 100_000

    def test_commission_config_affects_result(self):
        """With zero fees the result should be at least as good as with fees."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        close = np.linspace(100, 110, 50)
        data = _make_data(dates, close)
        sig = _make_signal_series(dates, buy_indices=[5], sell_indices=[45])
        strategy = _MockStrategy(sig)

        cfg_with = Config(COMMISSION_RATE=0.001, SLIPPAGE=0.0005)
        engine_with = BacktestEngine(config=cfg_with)
        result_with = engine_with.run(data, strategy)

        cfg_zero = Config(COMMISSION_RATE=0.0, SLIPPAGE=0.0)
        engine_zero = BacktestEngine(config=cfg_zero)
        strategy2 = _MockStrategy(sig)
        result_zero = engine_zero.run(data, strategy2)

        assert result_with['equity_curve'].iloc[-1] <= result_zero['equity_curve'].iloc[-1]

    def test_custom_symbol_in_result(self):
        dates = pd.date_range(start='2024-01-01', periods=50, freq='B')
        close = np.full(50, 100.0)
        data = _make_data(dates, close)
        strategy = _MockStrategy(pd.Series(0, index=dates, dtype=int))
        engine = BacktestEngine()
        result = engine.run(data, strategy, symbol='TSLA')
        assert result['symbol'] == 'TSLA'


class TestAnalyzer:

    def _make_result(self, equity_series, trades=None):
        return {
            'equity_curve': equity_series,
            'trades': trades or [],
        }

    def test_annual_return(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        result = self._make_result(equity)
        metrics = Analyzer(result).analyze()
        assert 18 < metrics['annual_return_pct'] < 22

    def test_max_drawdown(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        equity = pd.Series(
            [100000] * 50 + list(np.linspace(100000, 90000, 50)),
            index=dates
        )
        result = self._make_result(equity)
        metrics = Analyzer(result).analyze()
        assert metrics['max_drawdown_pct'] == pytest.approx(10.0, abs=0.5)

    def test_no_drawdown_when_always_up(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        equity = pd.Series(np.linspace(100000, 150000, 100), index=dates)
        result = self._make_result(equity)
        metrics = Analyzer(result).analyze()
        assert metrics['max_drawdown_pct'] == 0.0

    def test_sharpe_ratio_positive(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        equity = pd.Series(np.linspace(100000, 120000, len(dates)), index=dates)
        result = self._make_result(equity)
        metrics = Analyzer(result).analyze()
        assert metrics['sharpe_ratio'] > 0

    def test_win_rate(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.cumsum(np.random.randn(252) * 100), index=dates)
        trades = [
            {'profit_loss': 1000, 'action': 'sell', 'date': dates[10], 'symbol': 'TEST',
             'price': 110, 'shares': 100, 'commission': 1.0, 'cash_flow': 1000},
            {'profit_loss': -500, 'action': 'sell', 'date': dates[30], 'symbol': 'TEST',
             'price': 95, 'shares': 100, 'commission': 1.0, 'cash_flow': -500},
            {'profit_loss': 300, 'action': 'sell', 'date': dates[50], 'symbol': 'TEST',
             'price': 103, 'shares': 100, 'commission': 1.0, 'cash_flow': 300},
        ]
        result = self._make_result(equity, trades)
        metrics = Analyzer(result).analyze()
        assert metrics['win_rate_pct'] == pytest.approx(66.67, abs=0.1)

    def test_profit_factor(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        trades = [
            {'profit_loss': 1000, 'action': 'sell', 'date': dates[10], 'symbol': 'TEST',
             'price': 110, 'shares': 100, 'commission': 1.0, 'cash_flow': 1000},
            {'profit_loss': -200, 'action': 'sell', 'date': dates[30], 'symbol': 'TEST',
             'price': 98, 'shares': 100, 'commission': 1.0, 'cash_flow': -200},
        ]
        result = self._make_result(equity, trades)
        metrics = Analyzer(result).analyze()
        assert metrics['profit_factor'] == pytest.approx(5.0, abs=0.1)

    def test_all_metrics_present(self):
        dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
        equity = pd.Series(100000 + np.arange(252) * 10, index=dates)
        result = self._make_result(equity)
        metrics = Analyzer(result).analyze()
        required = ['annual_return_pct', 'max_drawdown_pct', 'sharpe_ratio',
                    'win_rate_pct', 'profit_factor', 'total_trades',
                    'final_equity', 'total_return_pct']
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
