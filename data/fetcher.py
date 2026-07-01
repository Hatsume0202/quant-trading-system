"""Data fetching module - supports yfinance and simulated data generation."""

from typing import List, Optional

import numpy as np
import pandas as pd


def fetch_data(
    symbols: List[str],
    start: str,
    end: str,
    source: str = "yfinance",
) -> dict:
    """Fetch OHLCV data for given symbols.

    Args:
        symbols: List of ticker symbols (e.g. ['AAPL', 'GOOGL']).
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
        source: Data source - 'yfinance' or 'simulated'.

    Returns:
        Dict mapping symbol -> pd.DataFrame with columns:
        ['open', 'high', 'low', 'close', 'volume'] and DatetimeIndex.

    Raises:
        ValueError: If source is unrecognized or data is empty.
    """
    result = {}
    for symbol in symbols:
        if source == "yfinance":
            result[symbol] = _fetch_yfinance(symbol, start, end)
        elif source == "simulated":
            result[symbol] = _generate_simulated(symbol, start, end)
        else:
            raise ValueError(f"Unknown data source: {source}")
    return result


def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch real market data from Yahoo Finance.

    Args:
        symbol: Ticker symbol.
        start: Start date string.
        end: End date string.

    Returns:
        DataFrame with OHLCV columns.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for real data. "
            "Install with: pip install yfinance"
        )

    ticker = yf.download(symbol, start=start, end=end, progress=False)

    if ticker.empty:
        raise ValueError(f"No data returned for {symbol} in range {start} to {end}")

    # Handle MultiIndex columns from yfinance
    if isinstance(ticker.columns, pd.MultiIndex):
        ticker = ticker.xs(symbol, axis=1, level=1)

    # Standardize column names
    ticker.columns = [c.lower() for c in ticker.columns]
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    available = [c for c in required_cols if c in ticker.columns]
    if len(available) < 5:
        missing = set(required_cols) - set(available)
        raise ValueError(f"Missing columns in data for {symbol}: {missing}")

    return ticker[required_cols]


def _generate_simulated(
    symbol: str,
    start: str,
    end: str,
    initial_price: Optional[float] = None,
    mu: float = 0.07,
    sigma: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate simulated OHLCV data using Geometric Brownian Motion.

    Args:
        symbol: Ticker symbol (used for seeding consistency).
        start: Start date.
        end: End date.
        initial_price: Starting price. If None, random 20-200.
        mu: Drift (annual return).
        sigma: Volatility (annual).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with daily OHLCV data.
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    if n < 2:
        raise ValueError(f"Date range {start} to {end} has insufficient trading days")

    rng = np.random.default_rng(seed)

    if initial_price is None:
        initial_price = 20.0 + (hash(symbol) % 180)

    dt = 1 / 252  # daily
    returns = rng.normal(
        (mu - 0.5 * sigma ** 2) * dt,
        sigma * np.sqrt(dt),
        size=n,
    )
    prices = initial_price * np.exp(np.cumsum(returns))

    volume = rng.integers(100_000, 10_000_000, size=n)

    # Build realistic OHLC with intraday noise
    high_noise = 1 + rng.uniform(0, 0.03, size=n)
    low_noise = 1 - rng.uniform(0, 0.03, size=n)
    open_noise = 1 + rng.normal(0, 0.005, size=n)

    df = pd.DataFrame({
        'open': prices * open_noise,
        'high': prices * high_noise,
        'low': prices * low_noise,
        'close': prices,
        'volume': volume,
    }, index=dates)

    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(len(df)):
        row = df.iloc[i]
        o, c = row['open'], row['close']
        df.iloc[i, df.columns.get_loc('high')] = max(row['high'], o, c)
        df.iloc[i, df.columns.get_loc('low')] = min(row['low'], o, c)

    return df
