"""Data storage for caching fetched stock data as CSV files."""

import os
import logging
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)


class DataStorage:
    """Caches stock data to local CSV files."""

    def __init__(self, cache_dir: str = DATA_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol.upper()}.csv")

    def save(self, df: pd.DataFrame, symbol: str) -> str:
        path = self.get_path(symbol)
        df.to_csv(path)
        logger.info(f"Saved {symbol} data to {path} ({len(df)} rows)")
        return path

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self.get_path(symbol)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {symbol} data from cache ({len(df)} rows)")
        return df

    def exists(self, symbol: str) -> bool:
        return os.path.exists(self.get_path(symbol))
