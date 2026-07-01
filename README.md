# Quantitative Trading System

A modular, extensible quantitative trading system in Python with multiple strategies, event-driven backtesting, performance analysis, and risk management.

## Features

- **3 Trading Strategies**: Dual MA Crossover, Bollinger Band Mean Reversion, Momentum
- **Event-Driven Backtest Engine**: Realistic simulation with commissions, slippage, and stamp duty
- **Performance Analytics**: Sharpe ratio, max drawdown, Calmar ratio, win rate, P&L ratio
- **Risk Management**: Position sizing, stop-loss/take-profit, drawdown circuit breaker, Kelly criterion
- **Data Sources**: yfinance (real market data) + Geometric Brownian Motion simulator
- **Visualization**: Equity curves, drawdown charts, monthly return heatmaps, trade P&L distributions

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run a dual MA crossover backtest on AAPL (2022-2023)
python main.py --symbol AAPL --strategy ma_cross --start 2022-01-01 --end 2023-12-31

# Run mean reversion strategy on TSLA
python main.py --symbol TSLA --strategy mean_reversion --start 2022-01-01 --end 2023-12-31

# Run momentum strategy with simulated data
python main.py --symbol AAPL --strategy momentum --source simulated

# Custom capital and strategy parameters
python main.py --symbol AAPL --strategy ma_cross --capital 500000 --short-window 10 --long-window 50
```

## Project Structure

```
quant-trading-system/
├── config/              # System configuration
│   └── settings.py      # Parameters (capital, fees, slippage, etc.)
├── data/                # Data module
│   ├── fetcher.py       # yfinance + simulated data
│   └── processor.py     # Cleaning + technical indicators
├── strategy/            # Strategy module
│   ├── base.py          # Abstract base class
│   ├── ma_cross.py      # Dual MA crossover
│   ├── mean_reversion.py # Bollinger Band mean reversion
│   └── momentum.py      # Momentum strategy
├── backtest/            # Backtesting module
│   ├── engine.py        # Event-driven backtest engine
│   ├── broker.py        # Simulated broker
│   └── analyzer.py      # Performance analysis + charts
├── risk/                # Risk management
│   └── manager.py       # Position sizing, stops, circuit breakers
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Strategies

### Dual MA Crossover
Buy when short MA crosses above long MA; sell on reverse crossover.
Parameters: `--short-window 5 --long-window 20`

### Bollinger Band Mean Reversion
Buy at lower band, sell at middle band. Optional short at upper band.
Parameters: `--bb-period 20 --bb-std 2.0`

### Momentum
Buy when N-period return exceeds threshold; sell when momentum turns negative.
Parameters: `--lookback 20 --threshold 0.02`

## Performance Metrics

- Annualized Return
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Maximum Drawdown (peak-to-trough)
- Calmar Ratio (return / max drawdown)
- Win Rate & Profit/Loss Ratio
- Benchmark Comparison (Buy & Hold)

## Output

All charts and trade logs are saved to `./output/` by default:
- `{SYMBOL}_equity_curve.png` - Equity curve + drawdown
- `{SYMBOL}_monthly_returns.png` - Monthly returns heatmap
- `{SYMBOL}_trade_pnl.png` - Trade P&L distribution
- `trades.json` - Complete trade log

## License

MIT
