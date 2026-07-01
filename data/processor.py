"""Data processing module - cleaning, alignment, and technical indicators."""

from typing import Optional

import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate OHLCV data.

    Performs:
    - Forward-fill missing values
    - Drop any remaining NaN rows
    - Ensure positive prices
    - Ensure high >= open, close, low and low <= open, close, high

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        Cleaned DataFrame with DatetimeIndex.
    """
    df = df.copy()

    # Forward fill gaps (weekends/holidays that yfinance might return)
    df = df.ffill()

    # Drop remaining NaN
    df = df.dropna()

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning")

    # Fix OHLC consistency
    for idx in df.index:
        o, h, l, c = df.at[idx, 'open'], df.at[idx, 'high'], df.at[idx, 'low'], df.at[idx, 'close']
        df.at[idx, 'high'] = max(h, o, c)
        df.at[idx, 'low'] = min(l, o, c)

    # Ensure positive
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].clip(lower=0.01)

    return df


def add_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add technical indicators to price data.

    Args:
        df: OHLCV DataFrame.
        **kwargs: Indicator parameters:
            sma_short (int): Short SMA period (default 5).
            sma_long (int): Long SMA period (default 20).
            ema_period (int): EMA period (default 20).
            bb_period (int): Bollinger Band period (default 20).
            bb_std (float): Bollinger Band std multiplier (default 2.0).
            rsi_period (int): RSI period (default 14).
            macd_fast (int): MACD fast period (default 12).
            macd_slow (int): MACD slow period (default 26).
            macd_signal (int): MACD signal period (default 9).
            atr_period (int): ATR period (default 14).

    Returns:
        DataFrame with original columns plus indicator columns:
        sma_short, sma_long, ema, bollinger_upper, bollinger_mid,
        bollinger_lower, rsi, macd, macd_signal, macd_hist, atr
    """
    df = df.copy()

    # Parameters
    sma_short = kwargs.get('sma_short', 5)
    sma_long = kwargs.get('sma_long', 20)
    ema_period = kwargs.get('ema_period', 20)
    bb_period = kwargs.get('bb_period', 20)
    bb_std = kwargs.get('bb_std', 2.0)
    rsi_period = kwargs.get('rsi_period', 14)
    macd_fast = kwargs.get('macd_fast', 12)
    macd_slow = kwargs.get('macd_slow', 26)
    macd_signal = kwargs.get('macd_signal', 9)
    atr_period = kwargs.get('atr_period', 14)

    close = df['close']

    # SMA
    df['sma_short'] = close.rolling(window=sma_short).mean()
    df['sma_long'] = close.rolling(window=sma_long).mean()

    # EMA
    df['ema'] = close.ewm(span=ema_period, adjust=False).mean()

    # Bollinger Bands
    rolling_std = close.rolling(window=bb_period).std()
    df['bollinger_mid'] = close.rolling(window=bb_period).mean()
    df['bollinger_upper'] = df['bollinger_mid'] + bb_std * rolling_std
    df['bollinger_lower'] = df['bollinger_mid'] - bb_std * rolling_std

    # RSI (Wilder's smoothing)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    for i in range(rsi_period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    high, low = df['high'], df['low']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()

    return df
