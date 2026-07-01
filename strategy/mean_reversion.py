"""Mean-reversion trading strategies.

Contains:
    BollingerBands: Buy at lower band, sell at middle band.
    RSIStrategy: Buy when RSI oversold, sell when RSI overbought.
    PairTrading: Z-score based pairs trading with hedge ratio.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from config import STRATEGY_PARAMS
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class BollingerBands(BaseStrategy):
    """Bollinger Bands mean-reversion strategy.

    Generates buy (1) when close dips to or below the lower Bollinger band,
    sell (-1) when close rises to or above the middle band (SMA).
    Position tracking prevents duplicate signals.

    Params (from STRATEGY_PARAMS["BollingerBands"]):
        period (int): SMA and standard deviation period (default 20).
        num_std (float): Number of standard deviations for bands (default 2.0).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize strategy with optional parameter overrides.

        Args:
            params: Dictionary of parameter overrides.
                    If None, uses defaults from config.STRATEGY_PARAMS.
        """
        merged = dict(STRATEGY_PARAMS.get("BollingerBands", {}))
        if params:
            merged.update(params)
        super().__init__(merged)
        self.period: int = int(self.params.get("period", 20))
        self.num_std: float = float(self.params.get("num_std", 2.0))

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate signals based on Bollinger Band touches.

        Args:
            data: DataFrame with 'close' column.
            **kwargs: Ignored.

        Returns:
            DataFrame with 'signal' column of int values: 1 (buy), -1 (sell), 0 (hold).
        """
        self._validate_data(data, ["close"])

        close = data["close"]
        n = len(data)

        signals = pd.DataFrame({"signal": 0}, index=data.index, dtype=int)

        if n < self.period:
            logger.warning(
                f"{self.name}: Not enough data ({n} rows) "
                f"for Bollinger period {self.period}"
            )
            return signals

        # Compute Bollinger Bands
        mid = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        position = 0  # 0 = flat, 1 = long
        for i in range(1, n):
            if pd.isna(lower.iloc[i]) or pd.isna(mid.iloc[i]):
                continue

            curr_close = close.iloc[i]
            curr_lower = lower.iloc[i]
            curr_mid = mid.iloc[i]

            # Buy: price at or below lower band from flat state
            if position == 0 and curr_close <= curr_lower:
                signals.iloc[i, 0] = 1
                position = 1
                logger.debug(
                    f"{self.name}: BUY signal at index {data.index[i]} "
                    f"(close={curr_close:.4f}, lower={curr_lower:.4f})"
                )

            # Sell: price at or above middle band while in long position
            elif position == 1 and curr_close >= curr_mid:
                signals.iloc[i, 0] = -1
                position = 0
                logger.debug(
                    f"{self.name}: SELL signal at index {data.index[i]} "
                    f"(close={curr_close:.4f}, mid={curr_mid:.4f})"
                )

        return signals


class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) mean-reversion strategy.

    Generates buy (1) when RSI falls below the oversold threshold,
    sell (-1) when RSI rises above the overbought threshold.
    Signals persist while the condition holds; position management
    is handled by the backtest engine.

    Params (from STRATEGY_PARAMS["RSIStrategy"]):
        period (int): RSI calculation period (default 14).
        oversold (int): Oversold threshold (default 30).
        overbought (int): Overbought threshold (default 70).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize strategy with optional parameter overrides.

        Args:
            params: Dictionary of parameter overrides.
                    If None, uses defaults from config.STRATEGY_PARAMS.
        """
        merged = dict(STRATEGY_PARAMS.get("RSIStrategy", {}))
        if params:
            merged.update(params)
        super().__init__(merged)
        self.period: int = int(self.params.get("period", 14))
        self.oversold: int = int(self.params.get("oversold", 30))
        self.overbought: int = int(self.params.get("overbought", 70))

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate signals based on RSI levels.

        Signal persists while condition holds: the backtest engine
        is responsible for position management.

        Args:
            data: DataFrame with 'close' column.
            **kwargs: Ignored.

        Returns:
            DataFrame with 'signal' column of int values: 1 (buy), -1 (sell), 0 (hold).
        """
        self._validate_data(data, ["close"])

        close = data["close"]
        n = len(data)

        signals = pd.DataFrame({"signal": 0}, index=data.index, dtype=int)

        if n < self.period + 1:
            logger.warning(
                f"{self.name}: Not enough data ({n} rows) "
                f"for RSI period {self.period}"
            )
            return signals

        # Compute RSI using Wilder's smoothing
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing: avg = (prev_avg * (period-1) + value) / period
        # Equivalent to ewm(alpha=1/period, adjust=False)
        avg_gain = gain.ewm(alpha=1.0 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, adjust=False).mean()

        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        for i in range(self.period + 1, n):
            if pd.isna(rsi[i]):
                continue

            curr_rsi = rsi[i]

            # Buy: RSI oversold
            if curr_rsi < self.oversold:
                signals.iloc[i, 0] = 1
                logger.debug(
                    f"{self.name}: BUY signal at index {data.index[i]} "
                    f"(RSI={curr_rsi:.2f})"
                )

            # Sell: RSI overbought
            elif curr_rsi > self.overbought:
                signals.iloc[i, 0] = -1
                logger.debug(
                    f"{self.name}: SELL signal at index {data.index[i]} "
                    f"(RSI={curr_rsi:.2f})"
                )

        return signals


class PairTrading(BaseStrategy):
    """Z-score based pairs trading strategy.

    Computes the spread between two stocks using a rolling OLS hedge ratio,
    then generates entry/exit signals based on the z-score of the spread.

    Entry: |z| > entry_z (default 2.0)
        - z < -entry_z: primary is cheap relative to hedge -> buy primary (1)
        - z > entry_z: primary is expensive relative to hedge -> sell primary (-1)
    Exit: |z| < exit_z (default 0.5)
        - Close position when spread mean-reverts.

    Params (from STRATEGY_PARAMS["PairTrading"]):
        lookback (int): Lookback window for z-score and hedge ratio (default 60).
        entry_z (float): Z-score threshold for entry (default 2.0).
        exit_z (float): Z-score threshold for exit (default 0.5).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize strategy with optional parameter overrides.

        Args:
            params: Dictionary of parameter overrides.
                    If None, uses defaults from config.STRATEGY_PARAMS.
        """
        merged = dict(STRATEGY_PARAMS.get("PairTrading", {}))
        if params:
            merged.update(params)
        super().__init__(merged)
        self.lookback: int = int(self.params.get("lookback", 60))
        self.entry_z: float = float(self.params.get("entry_z", 2.0))
        self.exit_z: float = float(self.params.get("exit_z", 0.5))

    def generate_signals(
        self,
        data: pd.DataFrame,
        hedge_data: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate pair trading signals.

        Args:
            data: Primary stock DataFrame with 'close' column.
            hedge_data: Hedge stock DataFrame with 'close' column.
                        Required for pair trading.
            **kwargs: Ignored.

        Returns:
            DataFrame with 'signal' column of int values: 1 (buy), -1 (sell), 0 (hold).

        Raises:
            ValueError: If hedge_data is not provided.
        """
        if hedge_data is None:
            raise ValueError(
                f"{self.name}: hedge_data is required for pair trading. "
                f"Pass the second stock's DataFrame as hedge_data."
            )

        self._validate_data(data, ["close"])
        self._validate_data(hedge_data, ["close"])

        # Align data on index
        common_index = data.index.intersection(hedge_data.index)
        if len(common_index) < self.lookback:
            logger.warning(
                f"{self.name}: Not enough aligned data "
                f"({len(common_index)} rows) for lookback {self.lookback}"
            )
            return pd.DataFrame(
                {"signal": 0}, index=data.index, dtype=int
            )

        close1 = data.loc[common_index, "close"].astype(float)
        close2 = hedge_data.loc[common_index, "close"].astype(float)

        n = len(common_index)

        # Compute log prices
        log_p1 = np.log(np.maximum(close1, 1e-10))
        log_p2 = np.log(np.maximum(close2, 1e-10))

        # Rolling OLS hedge ratio: beta = Cov(log(p1), log(p2)) / Var(log(p2))
        rolling_cov = log_p1.rolling(window=self.lookback).cov(log_p2)
        rolling_var = log_p2.rolling(window=self.lookback).var()

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            hedge_ratio = rolling_cov / rolling_var
            hedge_ratio = hedge_ratio.fillna(1.0)

        # Spread: log(p1) - beta * log(p2)
        spread = log_p1 - hedge_ratio * log_p2

        # Z-score of spread
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()

        # Avoid division by zero in z-score
        with np.errstate(divide="ignore", invalid="ignore"):
            z_score = (spread - spread_mean) / spread_std
            z_score = z_score.fillna(0.0)

        # Initialize signals DataFrame aligned to the original data index
        signals = pd.DataFrame(
            {"signal": 0},
            index=data.index,
            dtype=int,
        )

        position = 0  # 0 = flat, 1 = long, -1 = short
        for i in range(self.lookback, n):
            idx = common_index[i]
            z = z_score.iloc[i]

            if pd.isna(z):
                continue

            # Entry signals (only when not in position)
            if position == 0:
                if z < -self.entry_z:
                    # Primary is cheap: buy primary
                    signals.loc[idx, "signal"] = 1
                    position = 1
                    logger.debug(
                        f"{self.name}: BUY primary at index {idx} "
                        f"(z={z:.4f}, primary cheap vs hedge)"
                    )
                elif z > self.entry_z:
                    # Primary is expensive: sell primary
                    signals.loc[idx, "signal"] = -1
                    position = -1
                    logger.debug(
                        f"{self.name}: SELL primary at index {idx} "
                        f"(z={z:.4f}, primary expensive vs hedge)"
                    )

            # Exit signals (only when in position)
            elif abs(z) < self.exit_z:
                if position == 1:
                    # Close long position
                    signals.loc[idx, "signal"] = -1
                    logger.debug(
                        f"{self.name}: CLOSE LONG at index {idx} "
                        f"(z={z:.4f}, spread converged)"
                    )
                elif position == -1:
                    # Cover short position
                    signals.loc[idx, "signal"] = 1
                    logger.debug(
                        f"{self.name}: COVER SHORT at index {idx} "
                        f"(z={z:.4f}, spread converged)"
                    )
                position = 0

        return signals
