#!/usr/bin/env python3
"""Quantitative Trading System — CLI entry point.

Usage:
    python3 main.py backtest --symbol AAPL --strategy ma_crossover --start 2023-01-01 --end 2024-12-31 --capital 100000
    python3 main.py simulate --symbol AAPL --strategy momentum_breakout --days 60 --capital 100000
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')

from config import DEFAULT_SYMBOL, DEFAULT_CAPITAL, DEFAULT_START_DATE, DEFAULT_END_DATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def get_strategy(name: str):
    """Get strategy class by name."""
    if name == "ma_crossover":
        from strategy.ma_crossover import MACrossoverStrategy
        return MACrossoverStrategy()
    elif name == "momentum_breakout":
        from strategy.momentum_breakout import MomentumBreakoutStrategy
        return MomentumBreakoutStrategy()
    else:
        raise ValueError(
            f"Unknown strategy: {name}. "
            "Choose 'ma_crossover' or 'momentum_breakout'."
        )


def cmd_backtest(args):
    """Run a backtest over a historical period."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST MODE")
    print(f"{'='*60}")
    print(f"  Symbol:    {args.symbol}")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Period:    {args.start} to {args.end}")
    print(f"  Capital:   ${args.capital:,.0f}")
    print(f"{'='*60}\n")

    # 1. Fetch data
    from data.fetcher import DataFetcher
    fetcher = DataFetcher()
    logger.info(f"Fetching data for {args.symbol}...")
    data = fetcher.fetch(args.symbol, args.start, args.end)
    if data.empty:
        print("ERROR: No data fetched. Check symbol and date range.")
        sys.exit(1)
    print(f"  Fetched {len(data)} rows of OHLCV data")

    # 2. Generate signals
    strategy = get_strategy(args.strategy)
    signals = strategy.generate_signals(data)
    buy_count = (signals['Signal'] == 1).sum()
    sell_count = (signals['Signal'] == -1).sum()
    print(f"  Signals: {buy_count} buy, {sell_count} sell")

    # 3. Run backtest
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(initial_capital=args.capital)
    result = engine.run(data, signals, capital=args.capital)
    print(f"  Trades executed: {len(result['trades'])}")
    if result['trades']:
        total_pnl = sum(t['profit_loss'] for t in result['trades'])
        print(f"  Total P&L: ${total_pnl:,.2f}")
    print(f"  Final equity: ${result['final_equity']:,.2f}")

    # 4. Analyze performance
    from backtest.analyzer import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(result)
    metrics = analyzer.analyze()
    print(analyzer.report())

    # 5. Generate report (console + charts + HTML)
    from backtest.reporter import ReportGenerator
    reporter = ReportGenerator()
    report_path = reporter.generate(
        equity_curve=result['equity_curve'],
        metrics=metrics,
        trades=result['trades'],
        strategy_name=args.strategy,
        symbol=args.symbol,
    )
    print(f"\n  Report saved to: {report_path}")
    print(f"{'='*60}\n")


def cmd_simulate(args):
    """Run a live-like simulation day-by-day."""
    print(f"\n{'='*60}")
    print(f"  SIMULATE MODE")
    print(f"{'='*60}")
    print(f"  Symbol:    {args.symbol}")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Days:      {args.days}")
    print(f"  Capital:   ${args.capital:,.0f}")
    print(f"{'='*60}\n")

    # 1. Fetch recent data (simulated since we can't rely on live data)
    from data.fetcher import DataFetcher
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days + 60)).strftime("%Y-%m-%d")

    fetcher = DataFetcher()
    logger.info(f"Fetching data for {args.symbol} from {start_date} to {end_date}...")
    data = fetcher.fetch(args.symbol, start_date, end_date, source="simulated")
    if data.empty:
        print("ERROR: No data fetched.")
        sys.exit(1)
    print(f"  Fetched {len(data)} rows of OHLCV data")

    # 2. Generate signals for the full period
    strategy = get_strategy(args.strategy)
    signals = strategy.generate_signals(data)

    # 3. Set up portfolio, broker, and logger
    from executor.portfolio import Portfolio
    from executor.broker import Broker, OrderSide
    from executor.logger import TradeLogger

    portfolio = Portfolio(initial_cash=args.capital)
    broker = Broker()
    logger_trade = TradeLogger()

    # 4. Run day-by-day simulation (only last N days)
    simulation_data = data.iloc[-args.days:] if len(data) > args.days else data
    simulation_signals = signals.loc[simulation_data.index]

    print(f"  Simulating {len(simulation_data)} days...")

    trades_executed = 0
    last_price = float(simulation_data.iloc[-1]['Close'])
    for date in simulation_data.index:
        price = float(simulation_data.loc[date, 'Close'])
        signal = int(simulation_signals.loc[date, 'Signal'])

        # Update prices
        portfolio.update_prices({args.symbol: price})

        # Execute pending orders first
        broker.execute_pending_orders({args.symbol: price}, date=date)

        # Act on signals
        if signal == 1 and portfolio.get_position(args.symbol) is None:
            # Buy: use 80% of cash
            max_capital = portfolio.cash * 0.80
            shares = int(max_capital / price)
            if shares > 0:
                try:
                    portfolio.buy(args.symbol, shares, price, date=date)
                    broker.place_market_order(args.symbol, OrderSide.BUY, shares)
                    logger_trade.log_trade({
                        'action': 'BUY', 'symbol': args.symbol,
                        'shares': shares, 'price': price, 'date': str(date),
                    })
                    trades_executed += 1
                except ValueError as e:
                    logger.warning(f"Buy failed on {date.date()}: {e}")

        elif signal == -1 and portfolio.get_position(args.symbol) is not None:
            pos = portfolio.get_position(args.symbol)
            try:
                portfolio.sell(args.symbol, pos.shares, price, date=date)
                broker.place_market_order(args.symbol, OrderSide.SELL, pos.shares)
                logger_trade.log_trade({
                    'action': 'SELL', 'symbol': args.symbol,
                    'shares': pos.shares, 'price': price, 'date': str(date),
                })
                trades_executed += 1
            except ValueError as e:
                logger.warning(f"Sell failed on {date.date()}: {e}")

        # Log portfolio snapshot weekly
        if date.dayofweek == 4:  # Friday
            logger_trade.log_portfolio_snapshot(portfolio, date=date)

    # 5. Print summary
    equity = portfolio.get_equity({args.symbol: last_price})
    total_return = (equity / args.capital - 1) * 100
    positions = portfolio.position_count

    print(f"\n{'='*60}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Initial capital:     ${args.capital:>12,.2f}")
    print(f"  Final equity:        ${equity:>12,.2f}")
    print(f"  Total return:        {total_return:>11.2f}%")
    print(f"  Open positions:      {positions:>11}")
    print(f"  Trades executed:     {trades_executed:>11}")
    print(f"  Trade log:           {logger_trade.get_log_path()}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Trading System -- Backtest and simulate strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py backtest --symbol AAPL --strategy ma_crossover --start 2023-01-01 --end 2024-12-31
  python3 main.py simulate --symbol AAPL --strategy momentum_breakout --days 60 --capital 100000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest subcommand
    backtest_parser = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest_parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help=f"Ticker symbol (default: {DEFAULT_SYMBOL})",
    )
    backtest_parser.add_argument(
        "--strategy", type=str, default="ma_crossover",
        choices=["ma_crossover", "momentum_breakout"],
        help="Trading strategy (default: ma_crossover)",
    )
    backtest_parser.add_argument(
        "--start", type=str, default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    backtest_parser.add_argument(
        "--end", type=str, default=DEFAULT_END_DATE,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})",
    )
    backtest_parser.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL,
        help=f"Initial capital (default: {DEFAULT_CAPITAL:,.0f})",
    )

    # Simulate subcommand
    simulate_parser = subparsers.add_parser("simulate", help="Run a live-like simulation")
    simulate_parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help=f"Ticker symbol (default: {DEFAULT_SYMBOL})",
    )
    simulate_parser.add_argument(
        "--strategy", type=str, default="momentum_breakout",
        choices=["ma_crossover", "momentum_breakout"],
        help="Trading strategy (default: momentum_breakout)",
    )
    simulate_parser.add_argument(
        "--days", type=int, default=60,
        help="Number of days to simulate (default: 60)",
    )
    simulate_parser.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL,
        help=f"Initial capital (default: {DEFAULT_CAPITAL:,.0f})",
    )

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
