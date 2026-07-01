#!/usr/bin/env python3
"""Standalone backtest runner — works with current system interfaces.

Usage: python run_backtest.py
"""

import logging
import os
import sys
from datetime import date
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import (
    SYMBOLS, START_DATE, END_DATE, INITIAL_CAPITAL, RISK_FREE_RATE,
    BENCHMARK_SYMBOL, LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_backtest")


def main():
    print("=" * 70)
    print("  QUANTITATIVE TRADING SYSTEM — Backtest Runner")
    print("=" * 70)
    print(f"  Symbols:    {', '.join(SYMBOLS[:5])}...")
    print(f"  Period:     {START_DATE} to {END_DATE}")
    print(f"  Capital:    ${INITIAL_CAPITAL:,.0f}")
    print(f"  RF Rate:    {RISK_FREE_RATE:.1%}")
    print("=" * 70)

    # =========================================================================
    # 1. Generate synthetic test data (avoid network dependency for demo)
    # =========================================================================
    print("\n[1/5] Generating test data...")
    np.random.seed(42)
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    n = len(dates)
    print(f"  Trading days: {n}")

    all_data = {}
    for sym in SYMBOLS[:5]:  # Use 5 symbols for speed
        close = 100 + np.cumsum(np.random.randn(n) * 0.8)
        close = np.maximum(close, 10)
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        }, index=dates)
        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        all_data[sym] = df

    print(f"  Generated data for {len(all_data)} symbols")

    # =========================================================================
    # 2. Process data + generate signals
    # =========================================================================
    print("\n[2/5] Computing indicators and generating signals...")
    from data.processor import DataProcessor
    from strategy.trend_following import DualMACrossover, MACDStrategy, TurtleTrading
    from strategy.mean_reversion import BollingerBands, RSIStrategy, PairTrading

    processor = DataProcessor()

    strategies = [
        ("Dual MA Crossover", DualMACrossover()),
        ("MACD Strategy", MACDStrategy()),
        ("Turtle Trading", TurtleTrading()),
        ("Bollinger Bands", BollingerBands()),
        ("RSI Strategy", RSIStrategy()),
    ]

    # =========================================================================
    # 3. Run backtests
    # =========================================================================
    print("\n[3/5] Running backtests...")
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)

    all_results = []
    for strategy_label, strategy in strategies:
        print(f"\n  --- {strategy_label} ---")
        sym = SYMBOLS[0]
        df = processor.add_all_indicators(all_data[sym].copy())
        signals = strategy.generate_signals(df)

        # Convert signals to the format the engine expects
        sig_df = pd.DataFrame({
            "Signal": signals["signal"].values,
            "Price": df["close"].values,
        }, index=signals.index)

        # Convert to capitalized columns for engine
        data_for_engine = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        result = engine.run(data_for_engine, sig_df)
        final_equity = result["final_equity"]
        total_return = (final_equity / INITIAL_CAPITAL) - 1
        n_trades = len(result["trades"])

        # Calculate Sharpe from equity curve
        eq = result["equity_curve"]
        daily_ret = eq.pct_change().dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            sharpe = (daily_ret.mean() - RISK_FREE_RATE / 252) / daily_ret.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        rolling_max = eq.expanding().max()
        drawdown = (eq - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        # Win rate
        if n_trades > 0:
            wins = sum(1 for t in result["trades"] if t["profit_loss"] > 0)
            win_rate = wins / n_trades
            total_pnl = sum(t["profit_loss"] for t in result["trades"])
        else:
            win_rate = 0.0
            total_pnl = 0.0

        all_results.append({
            "strategy": strategy_label,
            "final_equity": final_equity,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trades": n_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        })

        print(f"    Final Equity:  ${final_equity:>12,.2f}")
        print(f"    Total Return:  {total_return:>11.2%}")
        print(f"    Sharpe Ratio:  {sharpe:>11.3f}")
        print(f"    Max Drawdown:  {max_dd:>11.2%}")
        print(f"    Trades:        {n_trades:>11}")
        print(f"    Win Rate:      {win_rate:>11.1%}")

    # =========================================================================
    # 4. Performance Summary
    # =========================================================================
    print(f"\n[4/5] Performance Summary")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Win%':>7}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['strategy']:<25} {r['total_return']:>7.2%} {r['sharpe']:>8.3f} "
            f"{r['max_dd']:>7.2%} {r['trades']:>7} {r['win_rate']:>6.1%}"
        )
    print("-" * 70)

    # Best strategy
    best = max(all_results, key=lambda r: r["sharpe"])
    print(f"\n  BEST: {best['strategy']} (Sharpe: {best['sharpe']:.3f}, Return: {best['total_return']:.2%})")

    # =========================================================================
    # 5. Generate Charts
    # =========================================================================
    print(f"\n[5/5] Generating charts...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        os.makedirs("reports", exist_ok=True)

        # Equity curves
        fig, ax = plt.subplots(figsize=(14, 7))
        for r, (label, strategy) in zip(all_results, strategies):
            sym = SYMBOLS[0]
            df = processor.add_all_indicators(all_data[sym].copy())
            signals = strategy.generate_signals(df)
            sig_df = pd.DataFrame({
                "Signal": signals["signal"].values,
                "Price": df["close"].values,
            }, index=signals.index)
            data_for_engine = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            result = engine.run(data_for_engine, sig_df)
            eq_norm = result["equity_curve"] / result["equity_curve"].iloc[0] * 100
            ax.plot(eq_norm.index, eq_norm.values, label=label, linewidth=1.5, alpha=0.8)

        ax.set_title("Strategy Equity Curves (Base=100)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.savefig("reports/equity_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: reports/equity_curves.png")

        # Drawdown chart for best strategy
        best_idx = max(range(len(all_results)), key=lambda i: all_results[i]["sharpe"])
        best_strategy = strategies[best_idx][1]
        sym = SYMBOLS[0]
        df = processor.add_all_indicators(all_data[sym].copy())
        signals = best_strategy.generate_signals(df)
        sig_df = pd.DataFrame({
            "Signal": signals["signal"].values, "Price": df["close"].values,
        }, index=signals.index)
        data_for_engine = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        result = engine.run(data_for_engine, sig_df)
        eq = result["equity_curve"]
        rolling_max = eq.expanding().max()
        dd = (eq - rolling_max) / rolling_max * 100

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(dd.index, 0, dd.values, color="red", alpha=0.3)
        ax.plot(dd.index, dd.values, color="red", linewidth=1)
        ax.set_title(f"Drawdown — {all_results[best_idx]['strategy']}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        fig.savefig("reports/drawdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: reports/drawdown.png")

    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPLETE")
    print(f"  Initial Capital:   ${INITIAL_CAPITAL:>12,.2f}")
    annual_returns = [r["total_return"] / 2 for r in all_results]  # 2-year period
    avg_annual = np.mean(annual_returns)
    print(f"  Avg Annual Return: {avg_annual:>11.2%}")
    print(f"  Target:            {'>=20.0%':>11}")
    print(f"{'='*70}")

    # Write markdown report
    md_path = "reports/summary.md"
    with open(md_path, "w") as f:
        f.write("# Quantitative Trading System — Backtest Report\n\n")
        f.write(f"**Period:** {START_DATE} to {END_DATE}  \n")
        f.write(f"**Initial Capital:** ${INITIAL_CAPITAL:,.0f}  \n")
        f.write(f"**Risk-Free Rate:** {RISK_FREE_RATE:.1%}  \n\n")
        f.write("## Strategy Performance\n\n")
        f.write("| Strategy | Return | Sharpe | Max DD | Trades | Win Rate |\n")
        f.write("|----------|--------|--------|--------|--------|----------|\n")
        for r in all_results:
            f.write(
                f"| {r['strategy']} | {r['total_return']:.2%} | {r['sharpe']:.3f} | "
                f"{r['max_dd']:.2%} | {r['trades']} | {r['win_rate']:.1%} |\n"
            )
        f.write(f"\n**Average Annual Return:** {avg_annual:.2%}\n\n")
        f.write("![Equity Curves](equity_curves.png)\n\n")
        f.write("![Drawdown](drawdown.png)\n\n")
        f.write("---\n")
        f.write("*Past performance does not guarantee future results.*\n")
    print(f"  Report saved: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
