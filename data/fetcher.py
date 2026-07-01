"""Data fetcher: yfinance for real data, GBM for simulated data."""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches historical OHLCV data or generates simulated data."""

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Fetch data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            start: Start date "YYYY-MM-DD".
            end: End date "YYYY-MM-DD".
            source: "yfinance" or "simulated".

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        if source == "simulated":
            return self._generate_simulated(symbol, start, end)
        
        try:
            return self._fetch_yfinance(symbol, start, end)
        except Exception as e:
            logger.warning(f"yfinance fetch failed: {e}, falling back to simulated")
            return self._generate_simulated(symbol, start, end)

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download from Yahoo Finance."""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        if data.empty:
            raise ValueError(f"No data for {symbol}")
        df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()

    def _generate_simulated(
        self, symbol: str, start: str, end: str,
        initial_price: float = 100.0, mu: float = 0.0005, sigma: float = 0.02,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data using Geometric Brownian Motion.

        Args:
            symbol: Ticker symbol (for labeling).
            start: Start date.
            end: End date.
            initial_price: Starting price.
            mu: Daily drift.
            sigma: Daily volatility.

        Returns:
            DataFrame with simulated OHLCV data.
        """
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n == 0:
            raise ValueError(f"No business days between {start} and {end}")

        np.random.seed(hash(symbol) % (2**31))
        returns = np.random.normal(mu, sigma, n)
        prices = initial_price * np.exp(np.cumsum(returns))

        data = []
        for i, p in enumerate(prices):
            daily_range = p * sigma * abs(np.random.normal(0, 1))
            o = p * (1 + np.random.normal(0, sigma * 0.5))
            h = max(o, p) + daily_range * abs(np.random.normal(0, 0.3))
            l = min(o, p) - daily_range * abs(np.random.normal(0, 0.3))
            c = p
            v = int(abs(np.random.normal(1_000_000, 300_000)))
            data.append([o, h, l, c, v])

        df = pd.DataFrame(
            data,
            index=dates,
            columns=["open", "high", "low", "close", "volume"],
        )
        df.index.name = "date"
        return df
