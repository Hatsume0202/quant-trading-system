# Quantitative Trading System

A comprehensive Python-based quantitative trading system with dual-market data support, 6 trading strategies (trend following + mean reversion), vectorized backtesting engine, and multi-layer risk management.

**Real Backtest Result (AAPL 2023-2024):** MACD Strategy achieved **26.5% annualized return** with 0.90 Sharpe ratio.

## Features

- **6 Trading Strategies:** Dual MA Crossover, MACD, Turtle Trading, Bollinger Bands, RSI, Pair Trading
- **Vectorized Backtest Engine:** Commission (0.03%), slippage (0.01%), stop-loss, multi-stock portfolio
- **21 Technical Indicators:** MA, EMA, MACD, RSI, Bollinger Bands, ATR, Donchian, VWAP
- **Risk Management:** Fixed-fractional sizing, ATR stops, Kelly criterion, drawdown circuit breaker
- **Dual Data Sources:** yfinance (US) + akshare (A-shares) with CSV caching
- **Performance Analytics:** Sharpe, Sortino, Calmar, Max Drawdown, Alpha/Beta, VaR
- **Visualization:** Equity curves, drawdown charts, monthly heatmaps, markdown reports
- **CLI Interface:** argparse with fetch, backtest, optimize, and report modes

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Standalone demo (works offline with synthetic data)
python run_backtest.py

# Fetch real market data
python main.py --mode fetch

# Run backtest with real data
python main.py --mode backtest --strategy MACDStrategy --symbols AAPL

# Run all strategies
python main.py --mode backtest --strategy all

# Optimize strategy parameters
python main.py --mode optimize --strategy MACDStrategy
```

## Real Backtest Results (AAPL, 2023-01 to 2025-01, $1M capital)

| Strategy | Return | Sharpe | Max DD | Trades | Win Rate |
|----------|--------|--------|--------|--------|----------|
| MACD Strategy | 26.48% | 0.904 | 10.16% | 23 | 43.5% |
| Bollinger Bands | 17.00% | 1.037 | 3.53% | 7 | 85.7% |
| Dual MA Crossover | 8.22% | 0.304 | 6.92% | 4 | 50.0% |
| AAPL Buy & Hold | 102.3% | — | — | 1 | — |

*Note: AAPL had exceptional returns in 2023-2024. Strategy returns are more risk-managed than buy & hold.*

## Project Structure

```
quant-trading-system/
├── main.py                  # CLI entry point (argparse)
├── run_backtest.py          # Standalone backtest demo
├── settings.py              # Extended system configuration
├── config.py                # Core configuration
├── requirements.txt         # Python dependencies
├── data/
│   ├── fetcher.py           # Data acquisition (yfinance + akshare)
│   ├── processor.py         # Technical indicators (21 total)
│   └── cache/               # Cached CSV data
├── strategy/
│   ├── base.py              # Abstract base strategy
│   ├── trend_following.py   # Dual MA, MACD, Turtle Trading
│   └── mean_reversion.py    # Bollinger Bands, RSI, Pair Trading
├── backtest/
│   ├── engine.py            # Vectorized backtest engine
│   ├── analyzer.py          # Performance metrics
│   └── reporter.py          # Charts + Markdown reports
├── risk/
│   └── manager.py           # Position sizing, stops, Kelly criterion
├── reports/                 # Generated reports + charts
└── tests/                   # pytest test suite
```

## Configuration

Edit `settings.py` to customize the system:

- **Market:** Switch between `"us"` (yfinance) and `"a"` (akshare A-shares)
- **Symbols:** Default pool of 10 liquid US large-cap stocks
- **Capital:** $1,000,000 initial capital (configurable)
- **Risk:** 2% risk per trade, 20% max position, 15% drawdown limit
- **Strategies:** All parameters configurable (MA periods, RSI thresholds, etc.)

## Risk Warning

**IMPORTANT:** This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always conduct thorough backtesting and paper trading before deploying any strategy with real capital.

## License

MIT
