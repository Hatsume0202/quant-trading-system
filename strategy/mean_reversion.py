"""Mean Reversion strategy using Bollinger Bands."""

import pandas as pd
from .base import Strategy


class MeanReversionStrategy(Strategy):
    """Buy when price touches lower band; sell when it returns to mid band.

    Optionally short when price touches upper band.

    Params:
        bb_period (int): Bollinger Band lookback period (default 20).
        bb_std (float): Standard deviation multiplier (default 2.0).
        allow_short (bool): Whether to allow short selling (default False).
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, allow_short: bool = False):
        super().__init__(bb_period=bb_period, bb_std=bb_std, allow_short=allow_short)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Band touches.

        Buy (1) when close <= lower_band (was above previously).
        Sell (-1) when close >= mid_band after being in position.
        Short (2) when close >= upper_band if allow_short=True.

        Uses pre-computed 'bollinger_upper', 'bollinger_mid', 'bollinger_lower'
        columns if present, otherwise computes from 'close'.

        Args:
            data: DataFrame with 'close' column.

        Returns:
            Signal series.
        """
        signals = pd.Series(0, index=data.index, dtype=int)
        close = data['close']

        # Compute or use pre-computed bands
        if 'bollinger_mid' in data.columns:
            mid = data['bollinger_mid']
            upper = data['bollinger_upper']
            lower = data['bollinger_lower']
        else:
            mid = close.rolling(window=self.bb_period).mean()
            std = close.rolling(window=self.bb_period).std()
            upper = mid + self.bb_std * std
            lower = mid - self.bb_std * std

        position = 0  # 0=flat, 1=long, -1=short
        for i in range(1, len(data)):
            if pd.isna(lower.iloc[i]) or pd.isna(mid.iloc[i]):
                continue

            if position == 0:
                # Buy signal: price at or below lower band
                if close.iloc[i] <= lower.iloc[i]:
                    signals.iloc[i] = 1
                    position = 1
                # Short signal: price at or above upper band (if allowed)
                elif self.allow_short and close.iloc[i] >= upper.iloc[i]:
                    signals.iloc[i] = 2   # 2 = short entry
                    position = -1

            elif position == 1:
                # Exit long: price returns to or above mid band
                if close.iloc[i] >= mid.iloc[i]:
                    signals.iloc[i] = -1
                    position = 0

            elif position == -1 and self.allow_short:
                # Cover short: price returns to or below mid band
                if close.iloc[i] <= mid.iloc[i]:
                    signals.iloc[i] = -1
                    position = 0

        return signals
