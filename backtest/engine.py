"""Backtest engine for simulating trading strategies."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, COMMISSION, SLIPPAGE

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade execution."""
    timestamp: pd.Timestamp
    symbol: str
    direction: str  # 'buy' or 'sell'
    price: float
    shares: int
    cost: float  # Including commission + slippage
    pnl: float = 0.0  # Realized P&L (filled on sell)


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    equity_curve: pd.Series        # Portfolio value over time
    trades: List[Trade]             # All executed trades
    positions: pd.DataFrame         # Holdings over time
    cash_curve: pd.Series           # Cash balance over time
    final_value: float              # Final portfolio value
    total_return: float             # Total return as fraction
    strategy_name: str = ""
    symbol: str = ""


class BacktestEngine:
    """Vectorized backtest engine with transaction cost simulation.

    Simulates trading with:
    - Commission: percentage of trade value
    - Slippage: adverse price movement on execution
    - Position tracking across multiple symbols
    - Trade logging for later analysis
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = COMMISSION,
        slippage: float = SLIPPAGE,
    ):
        """Initialize backtest engine.

        Args:
            initial_capital: Starting portfolio value.
            commission: Commission rate per trade (e.g., 0.0003 = 0.03%).
            slippage: Slippage rate per trade (e.g., 0.0001 = 0.01%).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        logger.info(
            f"BacktestEngine: capital=${initial_capital:,.0f}, "
            f"commission={commission:.3%}, slippage={slippage:.3%}"
        )

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: Optional[float] = None,
        strategy_name: str = "",
        symbol: str = "",
    ) -> BacktestResult:
        """Run backtest with given signals and prices.

        Args:
            signals: DataFrame with 'signal' column (1=buy, -1=sell, 0=hold).
                     May also have 'stop_loss' column for exit stops.
            prices: DataFrame with at least 'close' column for execution prices.
            initial_capital: Override default initial capital.
            strategy_name: Name of strategy being tested.
            symbol: Symbol being traded.

        Returns:
            BacktestResult with equity curve, trades, and final metrics.
        """
        capital = initial_capital or self.initial_capital

        # Align signals and prices
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        n = len(common_idx)
        cash = capital
        shares = 0
        equity = np.zeros(n)
        cash_series = np.zeros(n)
        position_series = np.zeros(n)
        trades: List[Trade] = []

        entry_price = 0.0
        stop_loss = None

        for i in range(n):
            date = common_idx[i]
            price = prices['close'].iloc[i]
            signal = signals['signal'].iloc[i]

            # Check stop loss on existing position
            if shares > 0 and 'stop_loss' in signals.columns:
                sl = signals['stop_loss'].iloc[i]
                if not np.isnan(sl) and price <= sl:
                    # Stop loss hit - force sell
                    signal = -1

            # Execute trades
            if signal == 1 and shares == 0:
                # Buy
                exec_price = price * (1 + self.slippage)
                max_shares = int(cash * 0.20 / exec_price)  # 20% max position
                shares = max(1, max_shares)
                cost = exec_price * shares * (1 + self.commission)

                if cost <= cash:
                    cash -= cost
                    entry_price = exec_price

                    if 'stop_loss' in signals.columns:
                        stop_loss = signals['stop_loss'].iloc[i]

                    trades.append(Trade(
                        timestamp=date, symbol=symbol, direction='buy',
                        price=exec_price, shares=shares, cost=cost
                    ))
                    logger.debug(f"BUY  {date.date()}: {shares} @ ${exec_price:.2f}")
                else:
                    shares = 0

            elif signal == -1 and shares > 0:
                # Sell
                exec_price = price * (1 - self.slippage)
                proceeds = exec_price * shares * (1 - self.commission)
                pnl = proceeds - (entry_price * shares * (1 + self.commission))
                cash += proceeds

                trades.append(Trade(
                    timestamp=date, symbol=symbol, direction='sell',
                    price=exec_price, shares=shares, cost=proceeds, pnl=pnl
                ))
                logger.debug(f"SELL {date.date()}: {shares} @ ${exec_price:.2f}, PnL=${pnl:,.2f}")
                shares = 0
                entry_price = 0.0
                stop_loss = None

            # Calculate equity
            position_value = shares * price
            equity[i] = cash + position_value
            cash_series[i] = cash
            position_series[i] = position_value

        # Close any remaining position at last price
        if shares > 0:
            final_price = prices['close'].iloc[-1] * (1 - self.slippage)
            proceeds = final_price * shares * (1 - self.commission)
            pnl = proceeds - (entry_price * shares * (1 + self.commission))
            cash += proceeds
            equity[-1] = cash
            trades.append(Trade(
                timestamp=common_idx[-1], symbol=symbol, direction='sell',
                price=final_price, shares=shares, cost=proceeds, pnl=pnl
            ))

        equity_series = pd.Series(equity, index=common_idx, name='equity')
        cash_series_s = pd.Series(cash_series, index=common_idx, name='cash')
        positions_s = pd.Series(position_series, index=common_idx, name='positions')

        total_return = (equity[-1] / capital) - 1.0

        return BacktestResult(
            equity_curve=equity_series,
            trades=trades,
            positions=pd.DataFrame({'position': positions_s, 'cash': cash_series_s}),
            cash_curve=cash_series_s,
            final_value=equity[-1],
            total_return=total_return,
            strategy_name=strategy_name,
            symbol=symbol,
        )

    def run_portfolio(
        self,
        all_signals: Dict[str, pd.DataFrame],
        all_prices: Dict[str, pd.DataFrame],
        initial_capital: Optional[float] = None,
        strategy_name: str = "",
    ) -> BacktestResult:
        """Run multi-stock portfolio backtest.

        Args:
            all_signals: Dict mapping symbol to signal DataFrame.
            all_prices: Dict mapping symbol to price DataFrame.
            initial_capital: Starting capital.
            strategy_name: Strategy name for reporting.

        Returns:
            Combined BacktestResult.
        """
        capital = initial_capital or self.initial_capital
        symbols = list(all_prices.keys())
        n_symbols = len(symbols)

        # Find common date range
        all_dates = all_prices[symbols[0]].index
        for sym in symbols[1:]:
            all_dates = all_dates.intersection(all_prices[sym].index)

        all_dates = all_dates.sort_values()
        n = len(all_dates)

        cash = capital
        holdings: Dict[str, int] = {s: 0 for s in symbols}
        entry_prices: Dict[str, float] = {s: 0.0 for s in symbols}

        equity = np.zeros(n)
        all_trades: List[Trade] = []
        capital_per_stock = capital / n_symbols

        for i, date in enumerate(all_dates):
            for sym in symbols:
                if date not in all_prices[sym].index:
                    continue

                price = all_prices[sym]['close'].loc[date]

                if sym in all_signals and date in all_signals[sym].index:
                    signal = all_signals[sym]['signal'].loc[date]
                else:
                    signal = 0

                if signal == 1 and holdings[sym] == 0:
                    exec_price = price * (1 + self.slippage)
                    alloc = min(cash * 0.20, capital_per_stock)
                    max_shares = int(alloc / exec_price)
                    shares = max(1, max_shares) if max_shares > 0 else 0
                    cost = exec_price * shares * (1 + self.commission)

                    if cost <= cash and shares > 0:
                        cash -= cost
                        holdings[sym] = shares
                        entry_prices[sym] = exec_price
                        all_trades.append(Trade(
                            timestamp=date, symbol=sym, direction='buy',
                            price=exec_price, shares=shares, cost=cost
                        ))

                elif signal == -1 and holdings[sym] > 0:
                    exec_price = price * (1 - self.slippage)
                    proceeds = exec_price * holdings[sym] * (1 - self.commission)
                    pnl = proceeds - (entry_prices[sym] * holdings[sym] * (1 + self.commission))
                    cash += proceeds
                    all_trades.append(Trade(
                        timestamp=date, symbol=sym, direction='sell',
                        price=exec_price, shares=holdings[sym], cost=proceeds, pnl=pnl
                    ))
                    holdings[sym] = 0
                    entry_prices[sym] = 0.0

            position_value = sum(
                holdings[s] * (all_prices[s]['close'].loc[date] if date in all_prices[s].index else 0)
                for s in symbols
            )
            equity[i] = cash + position_value

        # Close remaining positions
        for sym in symbols:
            if holdings[sym] > 0 and not all_prices[sym].empty:
                final_price = all_prices[sym]['close'].iloc[-1] * (1 - self.slippage)
                cash += final_price * holdings[sym] * (1 - self.commission)

        equity_series = pd.Series(equity, index=all_dates)
        total_return = (equity[-1] / capital) - 1.0 if n > 0 else 0.0

        return BacktestResult(
            equity_curve=equity_series,
            trades=all_trades,
            positions=pd.DataFrame(index=all_dates),
            cash_curve=pd.Series(cash, index=all_dates),
            final_value=equity[-1] if n > 0 else capital,
            total_return=total_return,
            strategy_name=strategy_name,
            symbol=",".join(symbols),
        )
