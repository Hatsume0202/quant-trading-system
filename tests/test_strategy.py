"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np

from strategy.ma_crossover import MACrossoverStrategy
from strategy.momentum_breakout import MomentumBreakoutStrategy


def create_test_data(n_days: int = 200, trend: float = 0.001) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
    rng = np.random.default_rng(42)
    returns = rng.normal(trend, 0.015, n_days)
    close = 100 * np.exp(np.cumsum(returns))
    data = pd.DataFrame({
        'Open': close * (1 + rng.normal(0, 0.003, n_days)),
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': rng.integers(50_000_000, 100_000_000, n_days),
    }, index=dates)
    for i in range(n_days):
        o, c = data.iloc[i]['Open'], data.iloc[i]['Close']
        data.iloc[i, data.columns.get_loc('High')] = max(data.iloc[i]['High'], o, c)
        data.iloc[i, data.columns.get_loc('Low')] = min(data.iloc[i]['Low'], o, c)
    return data


class TestMACrossoverStrategy:

    def test_initialization(self):
        strategy = MACrossoverStrategy(short_window=10, long_window=30)
        assert strategy.short_window == 10
        assert strategy.long_window == 30

    def test_default_parameters(self):
        strategy = MACrossoverStrategy()
        assert strategy.short_window == 10
        assert strategy.long_window == 30

    def test_generate_signals_shape(self):
        data = create_test_data(200)
        strategy = MACrossoverStrategy()
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)
        assert 'Signal' in signals.columns
        assert 'Price' in signals.columns

    def test_signal_values_are_valid(self):
        data = create_test_data(200)
        strategy = MACrossoverStrategy()
        signals = strategy.generate_signals(data)
        assert set(signals['Signal'].unique()).issubset({-1, 0, 1})

    def test_golden_cross_generates_buy(self):
        """When short MA crosses above long MA, should generate buy signal."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        prices = np.concatenate([
            np.linspace(100, 80, 50),
            np.linspace(80, 120, 50),
        ])
        data = pd.DataFrame({
            'Open': prices * 1.001, 'High': prices * 1.01,
            'Low': prices * 0.99, 'Close': prices,
            'Volume': np.full(100, 50_000_000),
        }, index=dates)
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        buy_signals = signals[signals['Signal'] == 1]
        assert len(buy_signals) > 0

    def test_death_cross_generates_sell(self):
        """When short MA crosses below long MA, should generate sell signal."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        prices = np.concatenate([
            np.linspace(100, 120, 50),
            np.linspace(120, 80, 50),
        ])
        data = pd.DataFrame({
            'Open': prices * 1.001, 'High': prices * 1.01,
            'Low': prices * 0.99, 'Close': prices,
            'Volume': np.full(100, 50_000_000),
        }, index=dates)
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        sell_signals = signals[signals['Signal'] == -1]
        assert len(sell_signals) > 0

    def test_no_signals_initially_during_warmup(self):
        data = create_test_data(100)
        strategy = MACrossoverStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(data)
        assert signals['Signal'].iloc[:29].sum() == 0


class TestMomentumBreakoutStrategy:

    def test_initialization(self):
        strategy = MomentumBreakoutStrategy(lookback=20, exit_lookback=10)
        assert strategy.lookback == 20
        assert strategy.exit_lookback == 10

    def test_default_parameters(self):
        strategy = MomentumBreakoutStrategy()
        assert strategy.lookback == 20
        assert strategy.exit_lookback == 10

    def test_generate_signals_shape(self):
        data = create_test_data(200)
        strategy = MomentumBreakoutStrategy()
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)
        assert 'Signal' in signals.columns
        assert 'Price' in signals.columns

    def test_signal_values_are_valid(self):
        data = create_test_data(200)
        strategy = MomentumBreakoutStrategy()
        signals = strategy.generate_signals(data)
        assert set(signals['Signal'].unique()).issubset({-1, 0, 1})

    def test_breakout_generates_buy(self):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='B')
        rng = np.random.default_rng(42)
        prices = np.concatenate([
            rng.uniform(90, 110, 30),
            np.linspace(110, 130, 30),
        ])
        data = pd.DataFrame({
            'Open': prices * 0.999, 'High': prices * 1.015,
            'Low': prices * 0.985, 'Close': prices,
            'Volume': np.full(60, 50_000_000),
        }, index=dates)
        strategy = MomentumBreakoutStrategy(lookback=20, exit_lookback=10)
        signals = strategy.generate_signals(data)
        buy_signals = signals[signals['Signal'] == 1]
        assert len(buy_signals) > 0

    def test_breakdown_generates_sell(self):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='B')
        rng = np.random.default_rng(42)
        prices = np.concatenate([
            rng.uniform(90, 110, 30),
            np.linspace(110, 80, 30),
        ])
        data = pd.DataFrame({
            'Open': prices * 0.999, 'High': prices * 1.015,
            'Low': prices * 0.985, 'Close': prices,
            'Volume': np.full(60, 50_000_000),
        }, index=dates)
        strategy = MomentumBreakoutStrategy(lookback=20, exit_lookback=10)
        signals = strategy.generate_signals(data)
        sell_signals = signals[signals['Signal'] == -1]
        assert len(sell_signals) > 0
