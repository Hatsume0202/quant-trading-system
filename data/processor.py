"""Data preprocessing and technical indicator calculation."""

import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values and duplicates."""
    df = df.copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to a DataFrame.

    Adds: MA_10, MA_30, MA_50, RSI_14, Volatility_20
    """
    df = df.copy()
    if 'Close' in df.columns:
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # RSI 14
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

        # Volatility (20-day std of returns)
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()

    return df


class DataProcessor:
    """Adds technical indicators and cleans OHLCV data."""

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return clean_data(df)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_indicators(df)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and add indicators in one step."""
        df = self.clean_data(df)
        return self.add_indicators(df)

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive set of technical indicators."""
        df = df.copy()
        if 'Close' in df.columns:
            df = self._add_ma(df, [5, 10, 20, 60, 120])
            df = self._add_ema(df, 12, 26)
            df = self._add_macd(df, 12, 26, 9)
            df = self._add_rsi(df, 14)
            df = self._add_bollinger(df, 20, 2.0)
            df = self._add_atr(df, 14)
            df = self._add_donchian(df, 20)
            df = self._add_returns(df)
        return df

    def _add_ma(self, df, periods):
        for p in periods:
            df[f"MA_{p}"] = df["Close"].rolling(window=p).mean()
        return df

    def _add_ema(self, df, fast, slow):
        df[f"EMA_{fast}"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df[f"EMA_{slow}"] = df["Close"].ewm(span=slow, adjust=False).mean()
        return df

    def _add_macd(self, df, fast, slow, signal):
        ema_f = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_s = df["Close"].ewm(span=slow, adjust=False).mean()
        df["MACD"] = ema_f - ema_s
        df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
        return df

    def _add_rsi(self, df, period):
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI_{period}"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def _add_bollinger(self, df, period, num_std):
        df["BB_Middle"] = df["Close"].rolling(window=period).mean()
        std = df["Close"].rolling(window=period).std()
        df["BB_Upper"] = df["BB_Middle"] + num_std * std
        df["BB_Lower"] = df["BB_Middle"] - num_std * std
        return df

    def _add_atr(self, df, period):
        tr_hl = df["High"] - df["Low"]
        tr_hc = abs(df["High"] - df["Close"].shift(1))
        tr_lc = abs(df["Low"] - df["Close"].shift(1))
        tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
        df[f"ATR_{period}"] = tr.rolling(window=period).mean()
        return df

    def _add_donchian(self, df, period):
        df["Donchian_High"] = df["High"].rolling(window=period).max()
        df["Donchian_Low"] = df["Low"].rolling(window=period).min()
        return df

    def _add_returns(self, df):
        df["Daily_Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        return df
