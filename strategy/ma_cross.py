"""Dual Moving Average Crossover strategy."""

import pandas as pd
from .base import Strategy


class MACrossStrategy(Strategy):
    """Buy when short MA crosses above long MA; sell when it crosses below.

    Params:
        short_window (int): Short moving average period (default 5).
        long_window (int): Long moving average period (default 20).
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy/sell signals based on MA crossover.

        Uses pre-computed 'sma_short' and 'sma_long' columns if present,
        otherwise computes simple moving averages from 'close'.

        Args:
            data: DataFrame with 'close' column, optionally 'sma_short' and 'sma_long'.

        Returns:
            Signal series: 1=buy, -1=sell, 0=hold.
        """
        signals = pd.Series(0, index=data.index, dtype=int)

        if 'sma_short' in data.columns and 'sma_long' in data.columns:
            short_ma = data['sma_short']
            long_ma = data['sma_long']
        else:
            short_ma = data['close'].rolling(window=self.short_window).mean()
            long_ma = data['close'].rolling(window=self.long_window).mean()

        # Crossover detection: short crosses above long
        position = 0  # 0=flat, 1=long
        for i in range(1, len(data)):
            if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
                continue

            prev_short = short_ma.iloc[i - 1]
            prev_long = long_ma.iloc[i - 1]
            curr_short = short_ma.iloc[i]
            curr_long = long_ma.iloc[i]

            if position == 0 and prev_short <= prev_long and curr_short > curr_long:
                signals.iloc[i] = 1  # buy signal
                position = 1
            elif position == 1 and prev_short >= prev_long and curr_short < curr_long:
                signals.iloc[i] = -1  # sell signal
                position = 0

        return signals
