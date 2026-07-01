"""Momentum-based trading strategy."""

import numpy as np
import pandas as pd
from .base import Strategy


class MomentumStrategy(Strategy):
    """Buy stocks with the strongest N-period momentum.

    For single-stock application: buy when momentum signal > threshold.
    For multi-stock: rank by momentum and buy top-K.

    Params:
        lookback (int): Lookback period for momentum calculation (default 20).
        threshold (float): Minimum momentum return to trigger buy (default 0.02).
        top_n (int): Number of top stocks to hold (for multi-stock, default 3).
    """

    def __init__(self, lookback: int = 20, threshold: float = 0.02, top_n: int = 3):
        super().__init__(lookback=lookback, threshold=threshold, top_n=top_n)
        self.lookback = lookback
        self.threshold = threshold
        self.top_n = top_n

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on momentum.

        Enters long when N-period return exceeds threshold.
        Exits when momentum turns negative (N-period return < 0).

        Args:
            data: DataFrame with 'close' column.

        Returns:
            Signal series.
        """
        signals = pd.Series(0, index=data.index, dtype=int)
        close = data['close']

        # Compute momentum: percentage change over lookback period
        momentum = close.pct_change(periods=self.lookback)

        position = 0
        for i in range(self.lookback, len(data)):
            if pd.isna(momentum.iloc[i]):
                continue

            if position == 0 and momentum.iloc[i] > self.threshold:
                signals.iloc[i] = 1
                position = 1
            elif position == 1 and momentum.iloc[i] < 0:
                signals.iloc[i] = -1
                position = 0

        return signals

    def rank_stocks(self, data_dict: dict) -> pd.DataFrame:
        """Rank multiple stocks by momentum for portfolio selection.

        Args:
            data_dict: Dict mapping symbol -> DataFrame with 'close' column.

        Returns:
            DataFrame with columns=['symbol', 'momentum', 'rank'],
            sorted by momentum descending.
        """
        momentums = {}
        for symbol, df in data_dict.items():
            close = df['close']
            if len(close) > self.lookback:
                mom = (close.iloc[-1] / close.iloc[-self.lookback] - 1)
            else:
                mom = 0.0
            momentums[symbol] = mom

        ranked = pd.DataFrame({
            'symbol': list(momentums.keys()),
            'momentum': list(momentums.values()),
        })
        ranked['rank'] = ranked['momentum'].rank(ascending=False)
        return ranked.sort_values('momentum', ascending=False)
