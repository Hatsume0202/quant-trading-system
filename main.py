#!/usr/bin/env python3
"""Quantitative Trading System - Main Entry Point.

Usage:
    python main.py --symbol AAPL --strategy ma_cross --start 2022-01-01 --end 2023-12-31
    python main.py --symbol AAPL --strategy mean_reversion --capital 200000
    python main.py --symbol TSLA --strategy momentum --source simulated

For more options, see --help.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config, config as default_config
from data.fetcher import DataFetcher
from data.processor import clean_data, add_indicators
from strategy import MACrossStrategy, MeanReversionStrategy, MomentumStrategy
from backtest import BacktestEngine, Analyzer
from risk import RiskManager


STRATEGY_MAP = {
    'ma_cross': MACrossStrategy,
    'mean_reversion': MeanReversionStrategy,
    'momentum': MomentumStrategy,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quantitative Trading System - Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol AAPL --strategy ma_cross --start 2022-01-01 --end 2023-12-31
  python main.py --symbol AAPL,GOOGL --strategy momentum --capital 500000
  python main.py --symbol TSLA --strategy mean_reversion --source simulated
        """,
    )

    parser.add_argument(
        '--symbol', type=str, default='AAPL',
        help='Stock symbol(s), comma-separated (default: AAPL)',
    )
    parser.add_argument(
        '--strategy', type=str, default='ma_cross',
        choices=['ma_cross', 'mean_reversion', 'momentum'],
        help='Trading strategy to use (default: ma_cross)',
    )
    parser.add_argument(
        '--start', type=str, default='2022-01-01',
        help='Backtest start date YYYY-MM-DD (default: 2022-01-01)',
    )
    parser.add_argument(
        '--end', type=str, default='2023-12-31',
        help='Backtest end date YYYY-MM-DD (default: 2023-12-31)',
    )
    parser.add_argument(
        '--capital', type=float, default=100_000.0,
        help='Initial capital in USD (default: 100000)',
    )
    parser.add_argument(
        '--source', type=str, default='yfinance',
        choices=['yfinance', 'simulated'],
        help='Data source (default: yfinance)',
    )
    parser.add_argument(
        '--output', type=str, default='./output',
        help='Output directory for charts and reports (default: ./output)',
    )
    parser.add_argument(
        '--short-window', type=int, default=5,
        help='Short MA window for ma_cross strategy (default: 5)',
    )
    parser.add_argument(
        '--long-window', type=int, default=20,
        help='Long MA window for ma_cross strategy (default: 20)',
    )
    parser.add_argument(
        '--bb-period', type=int, default=20,
        help='Bollinger Band period for mean_reversion (default: 20)',
    )
    parser.add_argument(
        '--bb-std', type=float, default=2.0,
        help='Bollinger Band std multiplier (default: 2.0)',
    )
    parser.add_argument(
        '--lookback', type=int, default=20,
        help='Lookback period for momentum strategy (default: 20)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.02,
        help='Momentum threshold (default: 0.02)',
    )
    parser.add_argument(
        '--no-risk', action='store_true',
        help='Disable risk management',
    )
    parser.add_argument(
        '--commission', type=float, default=0.0003,
        help='Commission rate (default: 0.0003 = 0.03%%)',
    )
    parser.add_argument(
        '--slippage', type=float, default=0.0001,
        help='Slippage rate (default: 0.0001 = 0.01%%)',
    )

    return parser.parse_args()


def create_strategy(args):
    """Create strategy instance from CLI args.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Strategy instance.

    Raises:
        ValueError: If strategy name is unknown.
    """
    strategy_name = args.strategy
    strategy_cls = STRATEGY_MAP[strategy_name]

    if strategy_name == 'ma_cross':
        return strategy_cls(short_window=args.short_window, long_window=args.long_window)
    elif strategy_name == 'mean_reversion':
        return strategy_cls(bb_period=args.bb_period, bb_std=args.bb_std)
    elif strategy_name == 'momentum':
        return strategy_cls(lookback=args.lookback, threshold=args.threshold)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def main():
    """Main execution flow."""
    args = parse_args()

    print("=" * 60)
    print("  QUANTITATIVE TRADING SYSTEM")
    print("=" * 60)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Strategy:   {args.strategy}")
    print(f"  Symbols:    {args.symbol}")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Data:       {args.source}")
    print("-" * 60)

    # Configure
    config = Config(
        INITIAL_CAPITAL=args.capital,
        COMMISSION_RATE=args.commission,
        SLIPPAGE=args.slippage,
    )

    # Parse symbols
    symbols = [s.strip() for s in args.symbol.split(',')]

    # Fetch data
    print("\n[1/5] Fetching data...")
    fetcher = DataFetcher()
    data_dict = {}
    for symbol in symbols:
        try:
            df = fetcher.fetch(symbol, args.start, args.end)
            data_dict[symbol] = df
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}, using simulated data")
            data_dict[symbol] = fetcher.fetch(symbol, args.start, args.end)

    # Process data
    print("[2/5] Processing data and computing indicators...")
    for symbol in data_dict:
        data_dict[symbol] = clean_data(data_dict[symbol])
        data_dict[symbol] = add_indicators(data_dict[symbol])
        print(f"  {symbol}: {len(data_dict[symbol])} bars, "
              f"{data_dict[symbol].index[0].date()} to {data_dict[symbol].index[-1].date()}")

    # Create strategy
    print("[3/5] Initializing strategy...")
    strategy = create_strategy(args)
    print(f"  Strategy: {strategy}")

    # Create risk manager
    risk_manager = None if args.no_risk else RiskManager(config)
    if risk_manager:
        print("  Risk Management: ENABLED")
    else:
        print("  Risk Management: DISABLED")

    # Run backtest
    print("[4/5] Running backtest...")
    engine = BacktestEngine(config=config)

    primary_symbol = symbols[0]
    data = data_dict[primary_symbol]

    result = engine.run(
        data=data,
        strategy=strategy,
        symbol=primary_symbol,
        risk_manager=risk_manager,
    )

    print(f"  Completed: {len(result['trades'])} trades executed")

    # Analyze results
    print("[5/5] Analyzing performance...")
    analyzer = Analyzer(result)

    # Print report
    print()
    print(analyzer.report())

    # Generate charts
    print(f"\nGenerating charts in {args.output}/...")
    analyzer.plot(save_dir=args.output)
    print(f"  Charts saved to {args.output}/")

    # Save trade log
    os.makedirs(args.output, exist_ok=True)
    trade_log_path = os.path.join(args.output, 'trades.json')

    trades_serializable = []
    for t in result['trades']:
        t_copy = t.copy()
        t_copy['date'] = str(t_copy['date'])
        trades_serializable.append(t_copy)

    with open(trade_log_path, 'w') as f:
        json.dump(trades_serializable, f, indent=2, default=str)
    print(f"  Trade log saved to {trade_log_path}")

    # Summary
    final_equity = result['equity_curve'].iloc[-1]
    total_return = (final_equity / config.INITIAL_CAPITAL) - 1
    print(f"\n{'=' * 60}")
    print(f"  Backtest Complete!")
    print(f"  Initial Capital:  ${config.INITIAL_CAPITAL:>12,.2f}")
    print(f"  Final Equity:     ${final_equity:>12,.2f}")
    print(f"  Total Return:     {total_return:>12.2%}")
    print(f"{'=' * 60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
