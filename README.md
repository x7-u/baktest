# Trading Strategy Backtester

A web-based backtesting engine that runs **Pine Script v5** and **MQL5 Expert Advisor** strategies against your own OHLCV data — or fetches data directly from MetaTrader 5.

## Changelog

### v3.2 — MT5 Integration + Accuracy Overhaul (Latest)

**MetaTrader 5 Integration**
- Fetch OHLCV data directly from a running MT5 terminal — no CSV export needed
- Symbol autocomplete from your broker's available instruments
- Date range picker (YYYY-MM-DD), optional SMT symbol fetch
- Green/red connection status indicator in the UI
- Requires `pip install MetaTrader5` (optional, Windows only)

**Risk-Based Position Sizing**
- New "Risk per Trade (%)" setting — auto-calculates lot size from SL distance and equity
- Set to 1% → each trade risks exactly 1% of current equity if SL is hit
- Works with both Pine Script `strategy.exit(stop=...)` and MQL5 `trade.Buy(..., sl, ...)`
- Falls back to fixed qty when no SL is provided

**Indicator Accuracy (23 fixes)**
- **Wilder's RSI** — proper exponential smoothing with persistent state (was simple average)
- **Wilder's ATR** — RMA smoothing matching TradingView/MT5 native (was simple average)
- **EMA cold-start** — uses available data instead of returning NA for early bars
- **Daily Sharpe/Sortino** — resampled to daily returns (was inflated 3-5x on intraday data)
- **Max drawdown** — consistent single-pass calculation for both % and $
- **Series lookback** — 500 → 5000 bar cap (long backtests no longer lose history)
- **Interpreter timeout** — 500K eval limit per bar catches infinite loops
- **Input validation** — rejects negative spread/commission/slippage, zero capital
- **Pending exits per-position** — SL/TP stored per trade ID, not shared globally
- **Pending orders in portfolio mode** — limit/stop entries work with existing positions
- **request.security** — returns NA for unavailable HTF (was silently wrong)
- **Date filter** — checks both entry AND exit dates
- **Stochastic** — returns NA on flat market (was arbitrary 50)
- **Profiles** — save and restore date exclusion filters
- Plus 9 more fixes (see commit history)

**UX**
- Script file upload (.txt/.pine/.mq5) with auto engine detection
- Profile system — save/load named settings configurations
- Recent files indicator — shows last 5 uploaded CSVs
- Cancel button — abort running backtests
- SMT chart overlay syncs with zoom/pan

### v3.1 — Multi-Timeframe, Walk-Forward, Features

- Multi-timeframe: select base TF, auto-aggregates H1/H4/D1/W1 from base data
- Walk-forward optimization with parallel execution across CPU cores
- Spread, slippage, per-lot commission simulation
- Intra-bar SL/TP priority with candle direction heuristic
- Monthly returns table, R-multiple distribution, exposure time, drawdown duration
- Metric tooltips, CSV export, auto-save, Ctrl+Enter, syntax error jumping
- Progress bar with SSE streaming, date exclusion filter, multi-symbol upload

### v3.0 — Dual-Engine Architecture

- MQL5 Expert Advisor interpreter (tokenizer, parser, tree-walking interpreter)
- Unified Pine Script + MQL5 in single app with engine toggle
- Cython acceleration on both engines (~5x speedup)
- Platform-level SMT divergence (works with any strategy)
- JPY pair auto-detection with P&L currency conversion

### v2.0 — Cython Engine

- Cython-optimized Pine Script interpreter (~5x speedup)
- SMT divergence via `request.security()` with secondary CSV

### v1.0 — Initial Release

- Pine Script v5 interpreter with 30+ built-in indicators
- Flask web UI with candlestick charts, equity curve, drawdown, trade log

## Features

**Engines & Data**
- **Two interpreters** — Pine Script v5 and MQL5 Expert Advisors
- **MetaTrader 5 integration** — fetch data directly from MT5 terminal (no CSV needed)
- **CSV upload** — auto-detects MetaTrader, TradingView, and standard CSV formats
- **Multi-timeframe** — select base TF, auto-aggregates H1/H4/D1/W1 candles
- **Cython acceleration** — ~5x speedup on both engines

**Execution & Risk**
- **Risk-based sizing** — auto-calculate lot size from SL distance and equity (1% risk etc.)
- **Spread + slippage + commission** — realistic execution modeling with configurable pips
- **Portfolio mode** — multiple simultaneous positions with per-position SL/TP
- **Pending orders** — limit and stop entry orders
- **Intra-bar SL/TP priority** — candle direction heuristic when both hit same bar

**Indicators & Analysis**
- **Wilder's RSI/ATR** — proper exponential smoothing matching TradingView/MT5
- **35+ built-in indicators** — SMA, EMA, RSI, ATR, MACD, BB, Stochastic, VWMA, pivots, crossover, valuewhen, barssince, momentum, percentrank, swma
- **Platform SMT divergence** — works with any strategy, no script code needed
- **Walk-forward optimization** — parallel rolling train/test parameter sweep

**Metrics & Analytics**
- **45+ metrics** — Sharpe, Sortino, Calmar, R-multiples, MAE/MFE, exposure, DD duration
- **Monthly returns table** — P&L breakdown by month with win rate
- **R-multiple distribution** — histogram of trade results as risk multiples
- **Interactive charts** — candlestick with trade markers, equity curve, drawdown, P&L distribution
- **CSV export** — download trades, metrics, monthly returns, equity curve
- **Metric tooltips** — hover any metric name for a plain-English explanation

**Usability**
- **Date exclusion filter** — remove trades during holidays, news, or low-liquidity periods (post-backtest)
- **Script file upload** — load .txt/.pine/.mq5 files with auto engine detection
- **Profile system** — save/load named settings (balance, commission, spread, dates, etc.)
- **Auto-save** — scripts, settings, engine choice persist in localStorage
- **Progress bar** — real-time bar count with cancel button
- **Ctrl+Enter** — keyboard shortcut to run backtest
- **Syntax error jumping** — "Go to Line X" button highlights the error
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
Requires MetaTrader 5 terminal running on Windows.

### 5. Run

```bash
python app.py
```

Or double-click **`start.bat`** (Windows) — builds Cython automatically and opens the browser.

Open **http://localhost:1234** in your browser.

### 6. Use it

1. **Select Engine** — click "Pine Script" or "MQL5" in the sidebar
2. **Get Data** — upload a CSV file OR fetch directly from MT5 (enter symbol + date range)
3. **Select Timeframe** — choose the base timeframe (M5, H1, etc.)
4. **Load Strategy** — paste code, or upload a .txt/.pine/.mq5 file
5. **(Optional) SMT** — upload a second CSV or enter an SMT symbol for divergence
6. **Configure** — set balance, risk %, commission, spread, slippage, UTC offset
7. **Run Backtest** — click the button (or Ctrl+Enter) and view results

## CSV Format

The backtester auto-detects common formats:

**MetaTrader export (tab-separated):**
```
<DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
2024.01.02	00:00:00	1.10432	1.10500	1.10300	1.10450	1234	0	5
```

**Standard CSV:**
```
datetime,open,high,low,close,volume
2024-01-02 00:00:00,1.10432,1.10500,1.10300,1.10450,1234
```

## Supported Features

### Pine Script v5
| Feature | Status |
|---------|--------|
| `strategy()` config | Supported |
| `strategy.entry/exit/close` with limit/stop | Supported |
| `input.int/float/bool/string` | Supported |
| `if/else`, `for`, `switch`, ternary | Supported |
| User-defined functions | Supported |
| `var` / `varip` declarations | Supported |
| History reference `close[1]` | Supported |
| `ta.sma/ema/rsi/atr/macd/bb/wma/vwma` | Supported (Wilder's smoothing) |
| `ta.crossover/crossunder` | Supported |
| `ta.highest/lowest/highestbars/lowestbars` | Supported |
| `ta.pivothigh/pivotlow` | Supported |
| `ta.valuewhen/barssince/rising/falling/mom` | Supported |
| `ta.percentrank/swma` | Supported |
| `request.security()` (multi-timeframe + SMT) | Supported |
| `math.*`, `array.*`, `str.*` | Supported |
| Platform SMT: `_smt_bull_active`, `_smt_bear_active` | Supported |

### MQL5 Expert Advisors
| Feature | Status |
|---------|--------|
| `OnInit/OnTick/OnDeinit` | Supported |
| `CTrade` (Buy/Sell/Close/Modify/BuyLimit/SellStop) | Supported |
| `iMA/iRSI/iMACD/iATR/iBands/iStoch` | Supported (Wilder's smoothing) |
| `CopyBuffer` | Supported |
| `PositionSelect/Get` queries | Supported |
| `input` variables, `#include`, `#define` | Supported |
| `if/else`, `for`, `while`, `do-while`, `switch` | Supported |
| User-defined functions | Supported |
| `ArrayResize/Copy/SetAsSeries` | Supported |
| `SymbolInfoDouble/Integer/String` | Supported |
| `TimeToStruct` (real timestamps + UTC offset) | Supported |
| Multi-timeframe `iHigh/iLow` via MTF engine | Supported |
| `MathSqrt/Abs/Max/Min/Pow/Log` | Supported |
| `StringFormat/Len/Find/Substr` | Supported |
| `NormalizeDouble`, type casting | Supported |

## Project Structure

```
baktest/
  app.py               # Flask web server (port 1234)
  backtester.py         # Trade execution engine (portfolio, pending orders, risk sizing)
  pine_parser.py        # Pine Script v5 tokenizer, parser, interpreter
  pine_fast.pyx         # Cython Pine Script interpreter (~5x faster)
  mql5_parser.py        # MQL5 tokenizer, parser, interpreter
  mql5_fast.pyx         # Cython MQL5 interpreter (~5x faster)
  mt5_source.py         # MetaTrader 5 data source (fetch OHLCV from terminal)
  mtf_engine.py         # Multi-timeframe aggregation engine
  smt_engine.py         # Platform-level SMT divergence detection
  setup.py              # Cython build script (both engines)
  start.bat             # One-click Windows launcher
  requirements.txt      # Python dependencies
  templates/
    index.html          # Web UI (SPA with engine toggle, MT5, charts, analytics)
  examples/
    sample_ea.mq5       # Simple MA crossover MQL5 EA
```

## Limitations

- **Bar-by-bar execution** — backtests run on OHLCV candles, not tick-by-tick. Intra-bar price paths are approximated with a candle direction heuristic
- **Pine Script subset** — not all v5 features are supported (drawing objects are no-ops, some advanced functions missing)
- **MQL5 subset** — no struct/class definitions, no custom indicators (`iCustom` is a stub), no order book
- **MTF approximation** — higher timeframe candles are aggregated from base data. Weekend gaps and market holidays may cause minor differences vs real broker HTF candles
- **MT5 requirement** — MT5 data fetching requires MetaTrader 5 terminal running on Windows
- **Performance** — complex strategies on 50K+ bars take 1-2 minutes even with Cython

## Roadmap

- [ ] **Monte Carlo simulation** — randomize trade order 1000x to show range of possible equity curves and probability of ruin
- [ ] **Parameter heatmap** — visual 2D heatmap of optimization results for two-parameter sweeps
- [ ] **Rolling Sharpe chart** — plot Sharpe ratio over a rolling window
- [ ] **Trade duration histogram** — distribution of holding periods, winners vs losers
- [ ] **Underwater equity chart** — visual timeline of drawdown periods with recovery time
- [ ] **Compare runs** — save multiple backtest results and overlay equity curves side-by-side
- [ ] **Strategy templates** — pre-built strategies selectable from a dropdown
- [ ] **Correlation matrix** — show return correlation across multiple strategies
- [ ] **Risk of ruin calculator** — estimate probability of account blowup
- [ ] **WebSocket live mode** — connect to a live data feed for paper trading

## License

MIT
