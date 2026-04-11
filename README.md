# Trading Strategy Backtester

A web-based backtesting engine that runs **Pine Script v5** and **MQL5 Expert Advisor** strategies against your own OHLCV data — or fetches data directly from MetaTrader 5.

## Changelog

### v3.5 — VWAP, Time Functions, Live Strategy Variables (Latest)

**New Indicators & Functions**
- `ta.vwap()` — volume-weighted average price with automatic daily reset
- `hour(time, "America/New_York")` — extract hour with timezone conversion
- `minute(time, timezone)` — extract minute with timezone conversion
- Broker UTC offset → target timezone conversion for session-based strategies

**Strategy Built-in Variables (Live Per-Bar)**
- `strategy.netprofit` — running net P&L, updated after every fill
- `strategy.grossprofit` / `strategy.grossloss` — cumulative win/loss totals
- `strategy.wintrades` / `strategy.losstrades` — win/loss trade counts
- `strategy.closedtrades` — total closed trade count
- `strategy.closedtrades.profit(idx)` — P&L of specific closed trade by index
- `strategy.closedtrades.entry_price(idx)` / `.exit_price(idx)` / `.size()`
- `strategy.initial_capital` — starting capital value

**UI & Dev**
- Custom dark modal system replacing all browser popups (alert/confirm/prompt)
- No-cache headers + template auto-reload for development
- Header layout and centering fixes

### v3.4 — Dark Theme Redesign

**UI Redesign**
- Pure black theme with dark grey accents
- Courier New monospace font throughout
- "Baktest" title in Lobster cursive font
- All sharp corners — zero border-radius
- Custom dark scrollbars matching theme
- Sidebar on right side, no-scroll body at 100% zoom
- Grey tab buttons, black action buttons with hover states
- Custom logo icon

**New Features**
- Clickable heatmap cells showing trade list for that day/hour
- Script diff tab with side-by-side metrics comparison table
- Phase 2 funded account simulation
- Screenshot/report export (PNG via html2canvas)
- Exit reason column in trade log (tp/sl/signal/opposite/end)
- Risk % properly overrides EA internal lot sizing
- SL/TP sanity check auto-fixes inverted short trades
- Daily drawdown enforcement in funded simulation
- Deterministic slippage (reproducible results)
- 15 bug fixes from critical code review

### v3.3 — Data Analysis + Funded Phases

**Data Analysis**
- **Equity curve comparison** — save multiple runs and overlay them on the same chart with dashed colored lines
- **Trade heatmap** — 7×24 grid showing P&L by hour of day and day of week, color-coded green/red
- **Consecutive loss simulator** — probability of 3/5/7/10/15/20 consecutive losses based on win rate
- **Drawdown recovery chart** — bar chart of each drawdown period with depth (%) and duration (bars)

**Strategy Development**
- **Script diff tab** — line-by-line comparison of your script vs the last run, with side-by-side metrics
- **Screenshot export** — one-click PNG capture of the full results page via html2canvas

**Funded Account**
- **Phase 2 simulation** — after phase 1 passes, auto-evaluates remaining trades with lower profit target
- **Detailed pass timeline** — shows exact date, trade #, trading days, and calendar days when target was hit
- **Drawdown breach tracking** — shows when/which trade breached the DD limit
- **SL/TP sanity check** — auto-detects and fixes inverted SL/TP on short trades
- **Exit reason tracking** — each trade shows why it closed (tp, sl, signal, opposite, end)

**Cross-Pair Fixes**
- **Universal lot sizing** — `SYMBOL_TRADE_TICK_VALUE` now dynamic based on profit currency (USD, JPY, CHF, CAD, etc.)
- **Currency-aware P&L conversion** — detects profit currency from symbol name

### v3.2 — MT5 Integration + Accuracy Overhaul

- MetaTrader 5 data fetching — symbol autocomplete, date range, SMT symbol
- Risk-based position sizing — auto-calculate lots from SL distance and equity
- 23 indicator accuracy fixes (Wilder's RSI/ATR, daily Sharpe, EMA cold-start, series cap)
- Pending exits per-position, portfolio mode fixes, interpreter timeout
- Script file upload, profile system, recent files, cancel button

### v3.1 — Multi-Timeframe, Walk-Forward, Features

- Multi-timeframe auto-aggregation from base data
- Walk-forward optimization with parallel execution
- Spread, slippage, per-lot commission, intra-bar SL/TP priority
- Monthly returns, R-multiples, metric tooltips, CSV export, auto-save, progress bar

### v3.0 — Dual-Engine Architecture

- MQL5 EA interpreter, unified Pine + MQL5 app, Cython acceleration
- Platform SMT divergence, JPY pair auto-detection

### v2.0 — Cython Engine | v1.0 — Initial Release

## Features

**Engines & Data**
- **Two interpreters** — Pine Script v5 and MQL5 Expert Advisors
- **MetaTrader 5 integration** — fetch data directly from MT5 terminal
- **CSV upload** — auto-detects MetaTrader, TradingView, and standard formats
- **Multi-timeframe** — select base TF, auto-aggregates H1/H4/D1/W1 candles
- **Cython acceleration** — ~5x speedup on both engines

**Execution & Risk**
- **Risk-based sizing** — auto-calculate lot size from SL distance and equity
- **Spread + slippage + commission** — realistic execution modeling
- **Portfolio mode** — multiple simultaneous positions with per-position SL/TP
- **Pending orders** — limit and stop entry orders
- **Exit reason tracking** — shows tp/sl/signal/opposite/end for each trade

**Indicators & Analysis**
- **Wilder's RSI/ATR** — proper exponential smoothing
- **35+ built-in indicators** — SMA, EMA, RSI, ATR, MACD, BB, Stochastic, VWMA, VWAP, pivots, crossover, valuewhen, barssince, momentum, percentrank, swma
- **Time functions** — `hour()`, `minute()` with timezone conversion for session-based strategies
- **Live strategy variables** — `strategy.netprofit`, `wintrades`, `losstrades`, `closedtrades` updated per-bar after fills
- **Platform SMT divergence** — works with any strategy
- **Walk-forward optimization** — parallel rolling train/test parameter sweep

**Metrics & Analytics**
- **45+ metrics** — Sharpe, Sortino, Calmar, R-multiples, MAE/MFE, exposure, DD duration
- **Monthly returns table** — P&L breakdown by month with win rate
- **Trade heatmap** — P&L by hour and day of week
- **Equity curve comparison** — save and overlay multiple runs
- **Consecutive loss probability** — based on actual win rate
- **Drawdown recovery chart** — depth and duration of each DD period
- **R-multiple distribution** — histogram of trade results as risk multiples
- **Script diff** — line-by-line comparison vs last run with metrics side-by-side
- **CSV export** — trades, metrics, monthly returns, equity (with symbol + timestamp)
- **Screenshot export** — one-click PNG capture of full results page

**Funded Account Simulation**
- **Phase 1 + Phase 2** — profit target, max DD, daily DD, min trading days
- **Pass timeline** — exact date, trade #, trading days, calendar days to target
- **Drawdown breach tracking** — when and which trade breached the limit

**Usability**
- **Date exclusion filter** — remove trades during holidays/news (post-backtest)
- **Script file upload** — .txt/.pine/.mq5 with auto engine detection
- **Profile system** — save/load all settings including symbol, dates, funded rules
- **Auto-save** — scripts, settings, engine choice persist in localStorage
- **Progress bar** — animated fill with cancel button
- **Ctrl+Enter** — keyboard shortcut to run backtest
- **Syntax error jumping** — "Go to Line X" highlights the error
- **Broker UTC offset** — configurable timezone for session detection
- **One-click launch** — `start.bat` builds Cython and opens browser

## Quick Start

### 1. Install Python 3.10+

Download from [python.org](https://www.python.org/downloads/).

### 2. Clone and install

```bash
git clone https://github.com/x7-u/baktest.git
cd baktest
pip install -r requirements.txt
```

### 3. (Optional) Build Cython for ~5x speed

```bash
pip install cython setuptools
python setup.py build_ext --inplace
```

### 4. (Optional) Enable MT5 data fetching

```bash
pip install MetaTrader5
```

### 5. Run

```bash
python app.py
```

Or double-click **`start.bat`** on Windows.

Open **http://localhost:1234** in your browser.

## Supported Features

### Pine Script v5
| Feature | Status |
|---------|--------|
| `strategy()` config | Supported |
| `strategy.entry/exit/close` with limit/stop | Supported |
| `input.int/float/bool/string` | Supported |
| `if/else`, `for`, `switch`, ternary | Supported |
| User-defined functions, `var`/`varip` | Supported |
| `ta.sma/ema/rsi/atr/macd/bb/wma/vwma` | Supported (Wilder's) |
| `ta.crossover/crossunder/highest/lowest` | Supported |
| `ta.pivothigh/pivotlow/valuewhen/barssince` | Supported |
| `ta.rising/falling/mom/percentrank/swma` | Supported |
| `ta.vwap()` with daily reset | Supported |
| `hour(time, timezone)`, `minute()` | Supported |
| `strategy.netprofit/wintrades/closedtrades` | Supported (live per-bar) |
| `strategy.closedtrades.profit/entry_price/exit_price` | Supported |
| `request.security()` (MTF + SMT) | Supported |
| `math.*`, `array.*`, `str.*` | Supported |

### MQL5 Expert Advisors
| Feature | Status |
|---------|--------|
| `OnInit/OnTick/OnDeinit` | Supported |
| `CTrade` (Buy/Sell/Close/Modify/Limit/Stop) | Supported |
| `iMA/iRSI/iMACD/iATR/iBands/iStoch` + `CopyBuffer` | Supported (Wilder's) |
| `PositionSelect/Get`, `SymbolInfo`, `AccountInfo` | Supported |
| `input`, `#include`, `#define`, `TimeToStruct` | Supported |
| Multi-timeframe `iHigh/iLow` via MTF engine | Supported |
| All control flow, user functions, arrays, math, strings | Supported |

## Limitations

- **Bar-by-bar execution** — OHLCV candles, not tick-by-tick
- **Pine/MQL5 subset** — drawing objects are no-ops, no `iCustom`, no order book
- **MTF approximation** — aggregated from base data, weekend gaps may differ
- **MT5 requirement** — data fetching requires MT5 terminal on Windows
- **Performance** — complex strategies on 50K+ bars take 1-2 min with Cython

## Roadmap

- [ ] **Monte Carlo simulation** — randomize trade order to show probability distributions
- [ ] **Parameter heatmap** — 2D heatmap for two-parameter optimization
- [ ] **Rolling Sharpe chart** — Sharpe ratio over a rolling window
- [ ] **Trade duration histogram** — holding periods, winners vs losers
- [ ] **Strategy templates** — pre-built strategies from dropdown
- [ ] **Risk of ruin calculator** — probability of account blowup
- [ ] **WebSocket live mode** — paper trading with live data feed

## License

MIT
