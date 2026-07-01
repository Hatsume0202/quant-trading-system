"""Data processing module for cleaning and computing technical indicators.

Provides a DataProcessor class that normalizes OHLCV data and computes
a comprehensive set of technical indicators using pure pandas/numpy.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes OHLCV data: cleans, validates, and adds technical indicators.

    All indicator calculations use pure pandas/numpy — no external TA library
    dependencies. The class is stateless; methods operate on provided DataFrames.
    """

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV market data.

        Performs:
        - Normalizes column names to lowercase.
        - Forward-fills missing values (NaN).
        - Flags outliers (> 5 standard deviations from rolling mean).
        - Drops any remaining NaN rows.
        - Ensures high >= max(open, close) and low <= min(open, close).
        - Clips negative prices to a minimum of 0.01.

        Args:
            df: Raw OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            Cleaned DataFrame with DatetimeIndex, sorted by date, and an
            additional ``is_outlier`` boolean column.

        Raises:
            ValueError: If the DataFrame is empty after cleaning.
        """
        df = df.copy()

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns exist
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Forward-fill missing values (e.g., weekends, holidays)
        df = df.ffill()

        # Flag outliers (> 5 sigma from rolling 20-period mean)
        df["is_outlier"] = False
        close = df["close"]
        if len(close) >= 20:
            rolling_mean = close.rolling(window=20, min_periods=1).mean()
            rolling_std = close.rolling(window=20, min_periods=1).std()
            rolling_std = rolling_std.replace(0, np.nan)
            z_scores = (close - rolling_mean) / rolling_std
            df.loc[z_scores.abs() > 5, "is_outlier"] = True
            outlier_count = df["is_outlier"].sum()
            if outlier_count > 0:
                logger.warning(
                    "Detected %d outlier data points (> 5 sigma)", outlier_count
                )

        # Drop remaining NaN rows (after forward-fill, only leading NaN remain)
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        if df.empty:
            raise ValueError("DataFrame is empty after cleaning")

        # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)

        # Clip negative/zero prices
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].clip(lower=0.01)

        # Sort by index if datetime
        df = df.sort_index()

        logger.info(
            "Data cleaned: %d rows, %d columns, %d outliers",
            len(df),
            len(df.columns),
            df["is_outlier"].sum(),
        )

        return df

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame.

        Applies cleaning first, then computes every indicator in one pass.
        Indicators added:
        - MA(5, 10, 20, 60, 120)
        - EMA(12, 26)
        - MACD(12, 26, 9) with signal line and histogram
        - RSI(14) using Wilder's smoothing
        - Bollinger Bands(20, 2)
        - ATR(14)
        - Donchian Channel(20)
        - VWAP
        - Daily returns and log returns

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            DataFrame with original columns plus all indicator columns.
        """
        # Clean first
        df = self.clean_data(df)

        # Compute all indicators
        df = self._add_ma(df, periods=[5, 10, 20, 60, 120])
        df = self._add_ema(df, fast=12, slow=26)
        df = self._add_macd(df, fast=12, slow=26, signal=9)
        df = self._add_rsi(df, period=14)
        df = self._add_bollinger(df, period=20, num_std=2)
        df = self._add_atr(df, period=14)
        df = self._add_donchian(df, period=20)
        df = self._add_vwap(df)
        df = self._add_returns(df)

        logger.info(
            "All indicators added: %d columns total", len(df.columns)
        )

        return df

    # ------------------------------------------------------------------
    # Individual indicator methods
    # ------------------------------------------------------------------

    def _add_ma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Simple Moving Average columns.

        Args:
            df: OHLCV DataFrame.
            periods: List of lookback periods (e.g., [5, 10, 20, 60, 120]).

        Returns:
            DataFrame with added ``ma_{period}`` columns.
        """
        close = df["close"]
        for p in periods:
            col_name = f"ma_{p}"
            df[col_name] = close.rolling(window=p, min_periods=p).mean()
            logger.debug("Added %s", col_name)
        return df

    def _add_ema(self, df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
        """Add Exponential Moving Average columns.

        Args:
            df: OHLCV DataFrame.
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).

        Returns:
            DataFrame with added ``ema_{fast}`` and ``ema_{slow}`` columns.
        """
        close = df["close"]
        df[f"ema_{fast}"] = close.ewm(span=fast, adjust=False).mean()
        df[f"ema_{slow}"] = close.ewm(span=slow, adjust=False).mean()
        logger.debug("Added ema_%d, ema_%d", fast, slow)
        return df

    def _add_macd(
        self, df: pd.DataFrame, fast: int, slow: int, signal: int
    ) -> pd.DataFrame:
        """Add MACD indicator with signal line and histogram.

        MACD line = EMA(close, fast) - EMA(close, slow).
        Signal line = EMA(MACD line, signal).
        Histogram = MACD line - Signal line.

        Args:
            df: OHLCV DataFrame.
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).
            signal: Signal line EMA period (default 9).

        Returns:
            DataFrame with added ``macd``, ``macd_signal``, ``macd_histogram`` columns.
        """
        close = df["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram
        logger.debug("Added macd, macd_signal, macd_histogram")
        return df

    def _add_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Relative Strength Index using Wilder's smoothing method.

        Wilder's smoothing:
        - First average gain = SMA of gains over the period.
        - Subsequent: avg_gain = (prev_avg_gain * (period - 1) + gain) / period.
        - Same for average loss.
        - RS = avg_gain / avg_loss; RSI = 100 - 100 / (1 + RS).

        Args:
            df: OHLCV DataFrame.
            period: RSI lookback period (default 14).

        Returns:
            DataFrame with added ``rsi_{period}`` column.
        """
        close = df["close"]
        delta = close.diff()

        # Clip positive/negative and fill the initial NaN (from diff) with 0
        gain = delta.clip(lower=0).fillna(0)
        loss = (-delta).clip(lower=0).fillna(0)

        # Initial SMA for gains and losses
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Apply Wilder's smoothing for remaining values
        if len(avg_gain) > period:
            # Convert to numpy for performance
            avg_gain_vals = avg_gain.values.astype(float)
            avg_loss_vals = avg_loss.values.astype(float)
            gain_vals = gain.values
            loss_vals = loss.values

            for i in range(period, len(avg_gain_vals)):
                avg_gain_vals[i] = (
                    avg_gain_vals[i - 1] * (period - 1) + gain_vals[i]
                ) / period
                avg_loss_vals[i] = (
                    avg_loss_vals[i - 1] * (period - 1) + loss_vals[i]
                ) / period

            avg_gain = pd.Series(avg_gain_vals, index=avg_gain.index)
            avg_loss = pd.Series(avg_loss_vals, index=avg_loss.index)

        # RS and RSI
        col_name = f"rsi_{period}"
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[col_name] = 100.0 - (100.0 / (1.0 + rs))

        # RSI values before the period are NaN (no Wilder's seed yet)
        # Already handled by the rolling min_periods

        logger.debug("Added %s", col_name)
        return df

    def _add_bollinger(
        self, df: pd.DataFrame, period: int, num_std: float
    ) -> pd.DataFrame:
        """Add Bollinger Bands.

        - Middle band = MA(close, period).
        - Upper band = middle + num_std * std(close, period).
        - Lower band = middle - num_std * std(close, period).

        Args:
            df: OHLCV DataFrame.
            period: Lookback period for MA and std (default 20).
            num_std: Number of standard deviations for bands (default 2.0).

        Returns:
            DataFrame with added ``bb_upper``, ``bb_middle``, ``bb_lower`` columns.
        """
        close = df["close"]
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()

        df["bb_upper"] = middle + num_std * std
        df["bb_middle"] = middle
        df["bb_lower"] = middle - num_std * std

        logger.debug("Added bb_upper, bb_middle, bb_lower")
        return df

    def _add_atr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Average True Range indicator.

        True Range = max(high - low, |high - prev_close|, |low - prev_close|).
        ATR is the rolling mean of True Range over the period.

        Args:
            df: OHLCV DataFrame.
            period: ATR lookback period (default 14).

        Returns:
            DataFrame with added ``atr_{period}`` column.
        """
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        col_name = f"atr_{period}"
        df[col_name] = true_range.rolling(window=period, min_periods=period).mean()

        logger.debug("Added %s", col_name)
        return df

    def _add_donchian(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add Donchian Channel indicator.

        Donchian High = highest high over the period.
        Donchian Low = lowest low over the period.

        Args:
            df: OHLCV DataFrame.
            period: Lookback period (default 20).

        Returns:
            DataFrame with added ``donchian_high`` and ``donchian_low`` columns.
        """
        df["donchian_high"] = (
            df["high"].rolling(window=period, min_periods=period).max()
        )
        df["donchian_low"] = (
            df["low"].rolling(window=period, min_periods=period).min()
        )

        logger.debug("Added donchian_high, donchian_low")
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume-Weighted Average Price indicator.

        VWAP = cumulative(typical_price * volume) / cumulative(volume),
        where typical_price = (high + low + close) / 3.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with added ``vwap`` column.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        volume_safe = df["volume"].replace(0, np.nan)

        pv = typical_price * volume_safe
        cum_pv = pv.cumsum()
        cum_vol = volume_safe.cumsum()

        df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)

        logger.debug("Added vwap")
        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily returns and log returns columns.

        - ``daily_returns``: (close - prev_close) / prev_close (simple return).
        - ``log_returns``: ln(close / prev_close) (logarithmic return).

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with added ``daily_returns`` and ``log_returns`` columns.
        """
        close = df["close"]
        df["daily_returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        logger.debug("Added daily_returns, log_returns")
        return df


# Convenience functions for backward compatibility
_processor = DataProcessor()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate OHLCV market data (convenience function)."""
    return _processor.clean_data(df)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV data (convenience function)."""
    return _processor.add_all_indicators(df)
