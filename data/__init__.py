"""Data module for fetching and processing market data."""

from .fetcher import fetch_data
from .processor import clean_data, add_indicators

__all__ = ["fetch_data", "clean_data", "add_indicators"]
