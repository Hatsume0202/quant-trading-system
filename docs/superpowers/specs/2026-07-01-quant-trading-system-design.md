# Quantitative Trading System — Design Specification

**Date:** 2026-07-01
**Goal:** Python-based quantitative trading system targeting ≥20% annualized return
**Initial Capital:** $1,000,000 USD

## Architecture Overview

```
main.py (CLI: argparse)
    ├── data/          # Data acquisition & preprocessing
    ├── strategy/      # Signal generation (BaseStrategy → 6 implementations)
    ├── backtest/      # Vectorized backtesting + analysis + reporting
    └── risk/          # Position sizing, stops, portfolio-level risk
```

## Module Design

### 1. Data Layer (`data/`)

**fetcher.py** — Unified data acquisition
- Primary: yfinance for US equities (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, V, JNJ)
- Secondary: akshare for A-shares (config-switchable via `config.MARKET`)
- Interface: `get_data(symbols, start, end, interval='1d') -> Dict[str, pd.DataFrame]`
- Caching: CSV files in `data/cache/`, keyed by symbol-interval
- Retry: 3 attempts with exponential backoff (1s, 2s, 4s)

**processor.py** — Technical indicators & cleaning
- Indicators: MA(5,10,20,60,120), EMA(12,26), MACD(12,26,9), RSI(14), Bollinger(20,2), ATR(14), Donchian(20), VWAP
- Cleaning: forward-fill NaN, flag outliers (>5σ from rolling mean)
- Derived columns: daily_returns, log_returns

### 2. Strategy Layer (`strategy/`)

**base.py** — `BaseStrategy` ABC with `generate_signals(data) -> pd.DataFrame`

**Trend Following:**
- `DualMACrossover`: Fast MA crosses above slow MA → buy (1); crosses below → sell (-1)
- `MACDStrategy`: MACD line crosses above signal → buy; crosses below → sell
- `TurtleTrading`: Price breaks above 20-day Donchian high → buy; breaks below 20-day low → sell; ATR-based stops

**Mean Reversion:**
- `BollingerBands`: Price touches lower band → buy; returns to middle band → sell
- `RSIStrategy`: RSI < 30 → buy; RSI > 70 → sell
- `PairTrading`: Z-score of spread between two cointegrated stocks; |z| > 2 → entry; z → 0 → exit

### 3. Backtest Layer (`backtest/`)

**engine.py** — Vectorized backtest
- Input: price data + signal DataFrame
- Simulates: commission (0.03%), slippage (0.01%)
- Output: equity curve, trade log, final portfolio value

**analyzer.py** — Performance metrics
- Total return, annualized return, Sharpe ratio (rf=2%), max drawdown, drawdown duration
- Win rate, profit/loss ratio, Calmar ratio
- Alpha & Beta vs S&P 500 benchmark
- Monthly/yearly return summaries

**reporter.py** — Visualization & reports
- Matplotlib (Agg backend): equity curve, drawdown curve, monthly heatmap, trade markers
- Markdown report with all metrics

### 4. Risk Layer (`risk/`)

**manager.py** — Risk controls
- Fixed-fraction: max 2% risk per trade
- ATR-based dynamic stop-loss
- Kelly criterion position sizing
- Single position cap: 20% of portfolio
- Portfolio drawdown cap: 15% (triggers liquidation)

### 5. CLI (`main.py`)

```
python main.py --mode fetch
python main.py --mode backtest --strategy trend_following
python main.py --mode backtest --strategy all
python main.py --mode optimize --strategy DualMACrossover
python main.py --mode report
```

### 6. Config (`config.py`)

All parameters externalized: symbols, date ranges, strategy params, risk params, cost params, benchmark symbol.

## Testing Strategy

- pytest with `tests/` mirroring source modules
- Test data fetcher with mocked yfinance responses
- Test indicator calculations against known values
- Test strategy signal generation with synthetic price data
- Test backtest engine with simple known-outcome scenarios
- Test risk manager position sizing math

## Data Flow

```
get_data() → processor.add_indicators() → strategy.generate_signals()
    → engine.run_backtest(signals, prices) → analyzer.analyze(results)
    → reporter.generate(results) → [charts + .md report]
```
