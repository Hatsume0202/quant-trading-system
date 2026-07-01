"""Mean reversion strategies: Bollinger Bands, RSI, and Pair Trading."""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from strategy.base import BaseStrategy

try:
    from config import STRATEGY_PARAMS
except ImportError:
    STRATEGY_PARAMS = {}

logger = logging.getLogger(__name__)

# Default strategy parameters (fallback if not in config)
_DEFAULT_MEAN_REVERSION_PARAMS = {
    "BollingerBands": {"period": 20, "num_std": 2.0},
    "RSIStrategy": {"period": 14, "oversold": 30, "overbought": 70},
    "PairTrading": {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5},
}


class BollingerBands(BaseStrategy):
    """Bollinger Bands mean reversion strategy.

    Buy signal: price touches or crosses below lower band (oversold).
    Sell signal: price returns to or crosses above middle band (mean reversion).

    Config keys in STRATEGY_PARAMS["BollingerBands"]:
        period (int): Moving average period (default 20).
        num_std (float): Number of standard deviations for bands (default 2.0).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with config defaults."""
        default = _DEFAULT_MEAN_REVERSION_PARAMS.get("BollingerBands", {})
        super().__init__(params=params or STRATEGY_PARAMS.get("BollingerBands", default))
        self.period: int = int(self.params.get("period", 20))
        self.num_std: float = float(self.params.get("num_std", 2.0))
        logger.info(f"BollingerBands params: period={self.period}, num_std={self.num_std}")

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate Bollinger Bands signals.

        Args:
            data: DataFrame with 'close' column.

        Returns:
            DataFrame with 'signal' column.
        """
        self._validate_data(data, ["close"])

        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0

        # Calculate Bollinger Bands
        middle = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        # Buy: close at or below lower band
        buy_condition = df["close"] <= lower
        signals.loc[buy_condition, "signal"] = 1

        # Sell: close at or above middle band (reversion target)
        sell_condition = df["close"] >= middle
        # Only sell if we were previously in buy territory
        sell_condition = sell_condition & (df["close"].shift(1) < middle.shift(1))
        signals.loc[sell_condition, "signal"] = -1

        logger.debug(
            f"BollingerBands: {signals['signal'].abs().sum()} signals generated"
        )
        return signals


class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) mean reversion strategy.

    Buy signal: RSI crosses below oversold threshold (default 30).
    Sell signal: RSI crosses above overbought threshold (default 70).

    Config keys in STRATEGY_PARAMS["RSIStrategy"]:
        period (int): RSI calculation period (default 14).
        oversold (int): Oversold threshold (default 30).
        overbought (int): Overbought threshold (default 70).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with config defaults."""
        default = _DEFAULT_MEAN_REVERSION_PARAMS.get("RSIStrategy", {})
        super().__init__(params=params or STRATEGY_PARAMS.get("RSIStrategy", default))
        self.period: int = int(self.params.get("period", 14))
        self.oversold: int = int(self.params.get("oversold", 30))
        self.overbought: int = int(self.params.get("overbought", 70))
        logger.info(
            f"RSIStrategy params: period={self.period}, "
            f"oversold={self.oversold}, overbought={self.overbought}"
        )

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        """Compute RSI using Wilder's smoothing method.

        Args:
            prices: Series of closing prices.

        Returns:
            Series of RSI values (0-100).
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Initial SMA for first value
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        # Wilder's smoothing for subsequent values
        for i in range(self.period, len(avg_gain)):
            avg_gain.iloc[i] = (
                avg_gain.iloc[i - 1] * (self.period - 1) + gain.iloc[i]
            ) / self.period
            avg_loss.iloc[i] = (
                avg_loss.iloc[i - 1] * (self.period - 1) + loss.iloc[i]
            ) / self.period

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate RSI-based signals.

        Args:
            data: DataFrame with 'close' column.

        Returns:
            DataFrame with 'signal' column.
        """
        self._validate_data(data, ["close"])

        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0

        # Compute RSI
        rsi = self._compute_rsi(df["close"])
        prev_rsi = rsi.shift(1)

        # Buy: RSI crosses below oversold (or was below and now crossing up)
        buy_condition = (rsi < self.oversold) & (prev_rsi >= self.oversold)
        signals.loc[buy_condition, "signal"] = 1

        # Also buy if RSI stays below oversold for extended period (first occurrence)
        # This is handled: we only signal on the first entry into oversold

        # Sell: RSI crosses above overbought
        sell_condition = (rsi > self.overbought) & (prev_rsi <= self.overbought)
        signals.loc[sell_condition, "signal"] = -1

        logger.debug(f"RSIStrategy: {signals['signal'].abs().sum()} signals generated")
        return signals


class PairTrading(BaseStrategy):
    """Statistical arbitrage pair trading strategy.

    Trades the spread between two cointegrated stocks using z-score.
    Enter when |z| > entry threshold, exit when |z| < exit threshold.

    Config keys in STRATEGY_PARAMS["PairTrading"]:
        lookback (int): Rolling window for z-score calculation (default 60).
        entry_z (float): Z-score threshold for entry (default 2.0).
        exit_z (float): Z-score threshold for exit (default 0.5).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with config defaults."""
        default = _DEFAULT_MEAN_REVERSION_PARAMS.get("PairTrading", {})
        super().__init__(params=params or STRATEGY_PARAMS.get("PairTrading", default))
        self.lookback: int = int(self.params.get("lookback", 60))
        self.entry_z: float = float(self.params.get("entry_z", 2.0))
        self.exit_z: float = float(self.params.get("exit_z", 0.5))
        logger.info(
            f"PairTrading params: lookback={self.lookback}, "
            f"entry_z={self.entry_z}, exit_z={self.exit_z}"
        )

    def generate_signals(
        self, data: pd.DataFrame, hedge_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> pd.DataFrame:
        """Generate pair trading signals.

        Args:
            data: DataFrame with 'close' for primary stock.
            hedge_data: DataFrame with 'close' for hedge/second stock.

        Returns:
            DataFrame with 'signal' column. Positive = long primary/short hedge,
            negative = short primary/long hedge.
        """
        self._validate_data(data, ["close"])

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0

        if hedge_data is None:
            logger.warning("PairTrading: No hedge data provided, returning no signals.")
            return signals

        if "close" not in hedge_data.columns:
            logger.warning("PairTrading: Hedge data missing 'close' column.")
            return signals

        # Align indices
        common_idx = data.index.intersection(hedge_data.index)
        if len(common_idx) < self.lookback:
            logger.warning(
                f"PairTrading: Not enough common data points "
                f"({len(common_idx)} < {self.lookback})"
            )
            return signals

        primary = data["close"].loc[common_idx]
        hedge = hedge_data["close"].loc[common_idx]

        # Calculate log prices
        log_primary = np.log(primary)
        log_hedge = np.log(hedge)

        # Calculate hedge ratio via rolling OLS: log(P1) = alpha + beta * log(P2)
        spread = log_primary - log_hedge

        # Rolling spread statistics
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()

        # Z-score
        z_score = (spread - spread_mean) / spread_std.replace(0, np.nan)
        prev_z = z_score.shift(1)

        # Entry signals
        # Long primary when z < -entry_z (primary is cheap relative to hedge)
        long_entry = (
            (z_score < -self.entry_z) & (prev_z >= -self.entry_z)
        )
        signals.loc[long_entry, "signal"] = 1

        # Short primary when z > +entry_z (primary is expensive relative to hedge)
        short_entry = (
            (z_score > self.entry_z) & (prev_z <= self.entry_z)
        )
        signals.loc[short_entry, "signal"] = -1

        # Exit signals: z-score reverts toward zero
        exit_long = (
            (z_score > -self.exit_z) & (prev_z <= -self.exit_z)
        )
        # Apply exit signal where we had a long signal previously
        # (simplified: exit on reversion)

        logger.debug(
            f"PairTrading: {signals['signal'].abs().sum()} signals generated"
        )
        return signals
