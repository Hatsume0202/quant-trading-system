"""Trend following strategies: Dual MA Crossover, MACD, and Turtle Trading."""

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
_DEFAULT_TREND_PARAMS = {
    "DualMACrossover": {"fast_period": 20, "slow_period": 50},
    "MACDStrategy": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "TurtleTrading": {"donchian_entry": 20, "donchian_exit": 10, "atr_period": 20},
}


class DualMACrossover(BaseStrategy):
    """Dual Moving Average Crossover strategy.

    Buy signal: fast MA crosses above slow MA.
    Sell signal: fast MA crosses below slow MA.

    Config keys in STRATEGY_PARAMS["DualMACrossover"]:
        fast_period (int): Fast moving average period (default 20).
        slow_period (int): Slow moving average period (default 50).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with config defaults, optionally overridden by params."""
        default = _DEFAULT_TREND_PARAMS.get("DualMACrossover", {})
        super().__init__(params=params or STRATEGY_PARAMS.get("DualMACrossover", default))
        self.fast_period: int = int(self.params.get("fast_period", 20))
        self.slow_period: int = int(self.params.get("slow_period", 50))
        logger.info(
            f"DualMACrossover params: fast={self.fast_period}, slow={self.slow_period}"
        )

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate signals based on MA crossover.

        Args:
            data: DataFrame with 'close' column.

        Returns:
            DataFrame with 'signal' column (1=buy, -1=sell, 0=hold).
        """
        self._validate_data(data, ["close"])

        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0

        # Calculate moving averages
        fast_ma = df["close"].rolling(window=self.fast_period).mean()
        slow_ma = df["close"].rolling(window=self.slow_period).mean()

        # Crossover detection: signal only on the crossover day
        prev_fast = fast_ma.shift(1)
        prev_slow = slow_ma.shift(1)

        # Buy: fast crosses above slow
        buy_condition = (prev_fast <= prev_slow) & (fast_ma > slow_ma)
        signals.loc[buy_condition, "signal"] = 1

        # Sell: fast crosses below slow
        sell_condition = (prev_fast >= prev_slow) & (fast_ma < slow_ma)
        signals.loc[sell_condition, "signal"] = -1

        logger.debug(
            f"DualMACrossover: {signals['signal'].abs().sum()} signals generated"
        )
        return signals


class MACDStrategy(BaseStrategy):
    """MACD (Moving Average Convergence Divergence) strategy.

    Buy signal: MACD line crosses above signal line (golden cross).
    Sell signal: MACD line crosses below signal line (death cross).

    Config keys in STRATEGY_PARAMS["MACDStrategy"]:
        fast_period (int): Fast EMA period (default 12).
        slow_period (int): Slow EMA period (default 26).
        signal_period (int): Signal line EMA period (default 9).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with config defaults."""
        default = _DEFAULT_TREND_PARAMS.get("MACDStrategy", {})
        super().__init__(params=params or STRATEGY_PARAMS.get("MACDStrategy", default))
        self.fast_period: int = int(self.params.get("fast_period", 12))
        self.slow_period: int = int(self.params.get("slow_period", 26))
        self.signal_period: int = int(self.params.get("signal_period", 9))
        logger.info(
            f"MACDStrategy params: fast={self.fast_period}, "
            f"slow={self.slow_period}, signal={self.signal_period}"
        )

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate signals based on MACD crossover.

        Args:
            data: DataFrame with 'close' column.

        Returns:
            DataFrame with 'signal' column.
        """
        self._validate_data(data, ["close"])

        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0

        # Calculate EMAs
        ema_fast = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Crossover detection
        prev_macd = macd_line.shift(1)
        prev_signal = signal_line.shift(1)

        # Buy: MACD crosses above signal
        buy_condition = (prev_macd <= prev_signal) & (macd_line > signal_line)
        signals.loc[buy_condition, "signal"] = 1

        # Sell: MACD crosses below signal
        sell_condition = (prev_macd >= prev_signal) & (macd_line < signal_line)
        signals.loc[sell_condition, "signal"] = -1

        logger.debug(f"MACDStrategy: {signals['signal'].abs().sum()} signals generated")
        return signals


class TurtleTrading(BaseStrategy):
    """Turtle Trading strategy using Donchian Channel breakouts with ATR stops.

    Entry: Price breaks above N-day Donchian high → buy.
    Exit: Price breaks below M-day Donchian low → sell.
    Stop-loss: ATR-based trailing stop.

    Config keys in STRATEGY_PARAMS["TurtleTrading"]:
        donchian_entry (int): Entry channel lookback (default 20).
        donchian_exit (int): Exit channel lookback (default 10).
        atr_period (int): ATR calculation period (default 20).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with config defaults."""
        default = _DEFAULT_TREND_PARAMS.get("TurtleTrading", {})
        super().__init__(params=params or STRATEGY_PARAMS.get("TurtleTrading", default))
        self.donchian_entry: int = int(self.params.get("donchian_entry", 20))
        self.donchian_exit: int = int(self.params.get("donchian_exit", 10))
        self.atr_period: int = int(self.params.get("atr_period", 20))
        logger.info(
            f"TurtleTrading params: entry={self.donchian_entry}, "
            f"exit={self.donchian_exit}, atr={self.atr_period}"
        )

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate turtle trading signals.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns.

        Returns:
            DataFrame with 'signal' and 'stop_loss' columns.
        """
        self._validate_data(data, ["high", "low", "close"])

        df = data.copy()
        signals = pd.DataFrame(index=df.index)
        signals["signal"] = 0
        signals["stop_loss"] = np.nan

        # Donchian channels
        entry_high = df["high"].rolling(window=self.donchian_entry).max()
        entry_low = df["low"].rolling(window=self.donchian_entry).min()
        exit_high = df["high"].rolling(window=self.donchian_exit).max()
        exit_low = df["low"].rolling(window=self.donchian_exit).min()

        # ATR for stop-loss
        tr_hl = df["high"] - df["low"]
        tr_hc = abs(df["high"] - df["close"].shift(1))
        tr_lc = abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()

        # Entry: price breaks above entry high
        prev_close = df["close"].shift(1)
        prev_high = entry_high.shift(1)
        prev_low = entry_low.shift(1)

        buy_condition = (prev_close <= prev_high) & (df["close"] > entry_high)
        signals.loc[buy_condition, "signal"] = 1
        # Stop loss for long: entry price - 2 * ATR
        signals.loc[buy_condition, "stop_loss"] = (
            df.loc[buy_condition, "close"] - 2 * atr.loc[buy_condition]
        )

        # Exit: price breaks below exit low
        sell_condition = (prev_close >= prev_low) & (df["close"] < exit_low)
        signals.loc[sell_condition, "signal"] = -1

        logger.debug(
            f"TurtleTrading: {signals['signal'].abs().sum()} signals generated"
        )
        return signals
