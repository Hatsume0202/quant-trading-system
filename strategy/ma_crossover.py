"""Dual Moving Average Crossover strategy."""

import pandas as pd

from config import MA_SHORT_WINDOW, MA_LONG_WINDOW
from .base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """Dual moving average crossover strategy.

    Buy signal: short MA crosses above long MA (golden cross).
    Sell signal: short MA crosses below long MA (death cross).
    """

    def __init__(self, short_window: int = MA_SHORT_WINDOW,
                 long_window: int = MA_LONG_WINDOW):
        super().__init__(short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0
        signals['Price'] = df['Close']

        df['MA_Short'] = df['Close'].rolling(window=self.short_window).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_window).mean()

        df['Short_Above'] = df['MA_Short'] > df['MA_Long']
        df['Cross_Above'] = df['Short_Above'] & (~df['Short_Above'].shift(1).fillna(False))
        df['Cross_Below'] = (~df['Short_Above']) & (df['Short_Above'].shift(1).fillna(False))

        signals.loc[df['Cross_Above'], 'Signal'] = 1
        signals.loc[df['Cross_Below'], 'Signal'] = -1

        return signals
