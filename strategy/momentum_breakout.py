"""Momentum Breakout strategy with RSI filter."""

import pandas as pd
import numpy as np

from config import MOMENTUM_LOOKBACK, MOMENTUM_EXIT_LOOKBACK, RSI_OVERBOUGHT, RSI_OVERSOLD
from .base import BaseStrategy


class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum breakout strategy.

    Buy: Price breaks above N-day high (and RSI not overbought).
    Sell: Price breaks below M-day low (and RSI not oversold).
    """

    def __init__(self, lookback: int = MOMENTUM_LOOKBACK,
                 exit_lookback: int = MOMENTUM_EXIT_LOOKBACK):
        super().__init__(lookback=lookback, exit_lookback=exit_lookback)
        self.lookback = lookback
        self.exit_lookback = exit_lookback

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0
        signals['Price'] = df['Close']

        df['Rolling_High'] = df['High'].rolling(window=self.lookback).max()
        df['Rolling_Low'] = df['Low'].rolling(window=self.exit_lookback).min()
        df['RSI'] = self._compute_rsi(df['Close'])

        df['Prev_High'] = df['Rolling_High'].shift(1)
        df['Prev_Low'] = df['Rolling_Low'].shift(1)

        buy_condition = (
            (df['Close'] > df['Prev_High']) &
            (df['Close'].shift(1) <= df['Prev_High']) &
            (df['RSI'] < RSI_OVERBOUGHT)
        )

        sell_condition = (
            (df['Close'] < df['Prev_Low']) &
            (df['Close'].shift(1) >= df['Prev_Low']) &
            (df['RSI'] > RSI_OVERSOLD)
        )

        signals.loc[buy_condition, 'Signal'] = 1
        signals.loc[sell_condition, 'Signal'] = -1

        return signals
