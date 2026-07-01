"""Data fetcher using yfinance with simulated data fallback."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches stock historical data from yfinance, falls back to simulated data."""

    def __init__(self):
        self._yfinance_available = None

    def _check_yfinance(self) -> bool:
        """Check if yfinance is available and working."""
        if self._yfinance_available is not None:
            return self._yfinance_available
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="5d")
            if data is not None and not data.empty:
                self._yfinance_available = True
                return True
        except Exception:
            pass
        self._yfinance_available = False
        return False

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical daily data for a symbol.

        Returns DataFrame with columns: Open, High, Low, Close, Volume, indexed by Date.
        """
        if self._check_yfinance():
            try:
                return self._fetch_yfinance(symbol, start, end)
            except Exception as e:
                logger.warning(f"yfinance fetch failed: {e}, falling back to simulated data")

        logger.info(f"Using simulated data for {symbol}")
        return self._generate_mock_data(symbol, start, end)

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch data from yfinance."""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end)
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data.index.name = 'Date'
        return data

    def _generate_mock_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Generate realistic simulated stock data using geometric Brownian motion."""
        dates = pd.date_range(start=start, end=end, freq='B')
        n = len(dates)
        if n == 0:
            raise ValueError(f"No business days between {start} and {end}")

        seed = sum(ord(c) for c in symbol)
        rng = np.random.default_rng(seed)

        initial_price = 150.0
        mu = 0.0008    # ~20% annual drift
        sigma = 0.015  # daily volatility

        daily_returns = rng.normal(mu, sigma, n)
        prices = initial_price * np.exp(np.cumsum(daily_returns))

        noise = rng.normal(0, 0.005, n)
        df = pd.DataFrame({
            'Open': prices * (1 + noise),
            'High': prices * (1 + np.abs(rng.normal(0, 0.01, n))),
            'Low': prices * (1 - np.abs(rng.normal(0, 0.01, n))),
            'Close': prices,
            'Volume': rng.integers(50_000_000, 100_000_000, n),
        }, index=dates)

        # Ensure OHLC consistency
        for i in range(n):
            o, c = df.iloc[i]['Open'], df.iloc[i]['Close']
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i]['High'], o, c)
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i]['Low'], o, c)

        df.index.name = 'Date'
        return df
