# Trading Strategy Backtester

A web-based backtesting engine that runs **Pine Script v5** and **MQL5 Expert Advisor** strategies against your own OHLCV data. No TradingView or MetaTrader account needed.

## Changelog

### v3.2 — Accuracy Overhaul (Latest)

**Indicator Accuracy**
- **Wilder's RSI** — proper exponential smoothing with persistent state across bars (was simple average)
- **Wilder's ATR** — RMA smoothing matching TradingView and MT5 native calculations (was simple average)
- **EMA cold-start** — uses best available data instead of returning NA for early bars
- **Daily Sharpe/Sortino** — resampled to daily returns before calculating (was inflated 3-5x on intraday data)
- **Max drawdown consistency** — both % and $ derived from single-pass calculation

**Reliability**
- **Interpreter timeout** — 500K evaluation limit per bar prevents infinite loops from freezing the server
- **Optimization error logging** — walk-forward optimizer shows errors instead of silently producing fake results
- **Input validation** — rejects negative spread, commission, slippage, zero capital
- **Series lookback** — 500 → 5000 bar cap (long backtests no longer silently lose indicator history)

**Portfolio & Orders**
- **Pending exits per-position** — SL/TP stored per trade ID, not shared globally across all positions
- **Pending orders in portfolio mode** — limit/stop entries fill even when other positions are open
- **Pending exit clearing** — per-position delete on close, no stale SL/TP carryover to new trades

**Fixes**
- **request.security** — returns NA for unavailable HTF data (was silently returning current bar value)
- **Date filter** — checks both entry AND exit dates against exclusion ranges
- **Stochastic** — returns NA on flat market instead of arbitrary 50
- **highestbars/lowestbars** — correct negative offset calculation
- **Strategy config** — `strategy()` call works on any bar, not just bar 0
- **SMT swing buffer** — stores 50 swings (was 10)
- **Profiles** — save and restore date exclusion filters
- **_get_history** — builds arrays in order directly (no unnecessary reversal)

### v3.1 — Multi-Timeframe, Walk-Forward, Features

**Multi-Timeframe**
- Select base timeframe (1m to 1M), auto-aggregates H1/H4/D1/W1 candles using real timestamp boundaries
- Pine: `request.security(syminfo.tickerid, "240", high)` — MQL5: `iHigh(_Symbol, PERIOD_H4, 1)`

**Walk-Forward Optimization**
- Rolling train/test parameter sweep with parallel execution across CPU cores
- Configurable metric target: Sharpe, Profit Factor, Net Profit, Win Rate

**Execution Modeling**
- Spread simulation (configurable pips), slippage (random 0-N pips), commission per lot
- Intra-bar SL/TP priority with candle direction heuristic
- R-multiple tracking per trade

**Analytics**
- Monthly returns table, R-multiple distribution chart, exposure time, drawdown duration
- Script file upload (.txt/.pine/.mq5), profile system, recent files, cancel button
- Metric tooltips, CSV export, auto-save, Ctrl+Enter, syntax error line jumping

### v3.0 — Dual-Engine Architecture

- MQL5 Expert Advisor interpreter (tokenizer, parser, tree-walking interpreter)
- Unified Pine Script + MQL5 in single app with engine toggle
- Cython acceleration on both engines (~5x speedup)
- Platform-level SMT divergence (works with any strategy)
- JPY pair auto-detection with P&L currency conversion

### v2.0 — Cython Engine

- Cython-optimized Pine Script interpreter (~5x speedup)
- SMT divergence via `request.security()` with secondary CSV upload

### v1.0 — Initial Release

- Pine Script v5 interpreter with 30+ built-in indicators
- Flask web UI with candlestick charts, equity curve, drawdown, trade log

## Features

- **Two interpreters** — Pine Script v5 and MQL5 Expert Advisors
- **Multi-timeframe** — select base TF, auto-aggregates H1/H4/D1/W1 candles from base data
- **Cython acceleration** — ~5x speedup on both engines
- **Walk-forward optimization** — parallel rolling train/test parameter sweep
- **Spread + slippage + commission** — realistic execution modeling with configurable pips
- **Portfolio mode** — multiple simultaneous positions with per-position SL/TP
- **Pending orders** — limit and stop entry orders
- **Platform SMT divergence** — works with any strategy, no script code needed
- **Wilder's RSI/ATR** — proper exponential smoothing matching TradingView/MT5
- **35+ built-in indicators** — SMA, EMA, RSI, ATR, MACD, BB, Stochastic, VWMA, pivots, crossover, valuewhen, barssince, momentum, percentrank
- **45+ metrics** — Sharpe, Sortino, Calmar, R-multiples, MAE/MFE, exposure, DD duration
- **Monthly returns** — P&L breakdown by month with win rate
- **Interactive charts** — candlestick with trade markers, equity, drawdown, P&L distribution, R-multiple histogram
- **Date exclusion filter** — remove trades during holidays, news events, or low-liquidity periods
- **Script file upload** — load .txt/.pine/.mq5 files directly
- **Profile system** — save/load named settings configurations
- **CSV export** — download trades, metrics, monthly returns, equity curve
- **Auto-save** — scripts, settings, engine choice persist across refreshes
- **Progress bar** — real-time bar count with cancel button
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

Requires a C compiler (MSVC on Windows, gcc on Linux/Mac):

```bash
pip install cython setuptools
python setup.py build_ext --inplace
```

### 4. Run

```bash
python app.py
```

Or double-click **`start.bat`** (Windows) — builds Cython automatically and opens the browser.

Open **http://localhost:1234** in your browser.

### 5. Use it

1. **Select Engine** — click "Pine Script" or "MQL5" in the sidebar
2. **Upload CSV** — drop your MetaTrader/TradingView OHLCV export (or upload a script file)
3. **Select Timeframe** — choose the base timeframe of your data (M5, H1, etc.)
4. **Paste Strategy** — enter code or upload a .txt/.pine/.mq5 file
5. **(Optional) SMT Data** — upload a second CSV for divergence detection
6. **Configure** — set balance, commission, spread, slippage, lot size, UTC offset
7. **Run Backtest** — click the button (or Ctrl+Enter) and view results

## CSV Format

The backtester auto-detects common formats. Any of these work:

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
| Visual functions (plot, label, line) | Parsed (no-op) |

### MQL5 Expert Advisors
| Feature | Status |
|---------|--------|
| `OnInit/OnTick/OnDeinit` | Supported |
| `CTrade` (Buy/Sell/Close/Modify/BuyLimit/SellStop) | Supported |
| `iMA/iRSI/iMACD/iATR/iBands/iStoch` | Supported (Wilder's smoothing) |
| `CopyBuffer` | Supported |
| `PositionSelect/Get` queries | Supported |
| `input` variables | Supported |
| `#include`, `#define`, `#property` | Supported |
| `if/else`, `for`, `while`, `do-while`, `switch` | Supported |
| User-defined functions | Supported |
| `ArrayResize/Copy/SetAsSeries` | Supported |
| `SymbolInfoDouble/Integer/String` | Supported |
| `TimeToStruct` (real timestamps + UTC offset) | Supported |
| Multi-timeframe `iHigh/iLow` via MTF engine | Supported |
| `MathSqrt/Abs/Max/Min/Pow/Log` | Supported |
| `StringFormat/Len/Find/Substr` | Supported |
| `NormalizeDouble`, type casting | Supported |
| Platform SMT: `_smt_bull_active`, `_smt_bear_active` | Supported |

## Project Structure

```
baktest/
  app.py               # Flask web server (port 1234)
  backtester.py         # Trade execution engine (dual-engine, portfolio, pending orders)
  pine_parser.py        # Pine Script v5 tokenizer, parser, interpreter
  pine_fast.pyx         # Cython Pine Script interpreter (~5x faster)
  mql5_parser.py        # MQL5 tokenizer, parser, interpreter
  mql5_fast.pyx         # Cython MQL5 interpreter (~5x faster)
  mtf_engine.py         # Multi-timeframe aggregation engine
  smt_engine.py         # Platform-level SMT divergence detection
  setup.py              # Cython build script (both engines)
  start.bat             # One-click Windows launcher
  requirements.txt      # Python dependencies
  templates/
    index.html          # Web UI (SPA with engine toggle, MTF, charts, analytics)
  examples/
    sample_ea.mq5       # Simple MA crossover MQL5 EA
```

## Limitations

- **Bar-by-bar execution** — backtests run on OHLCV candles, not tick-by-tick. Intra-bar price paths are approximated with a candle direction heuristic
- **Pine Script subset** — not all v5 features are supported (drawing objects are no-ops, some advanced functions missing)
- **MQL5 subset** — no struct/class definitions, no custom indicators (`iCustom` is a stub), no order book
- **MTF approximation** — higher timeframe candles are aggregated from base data. Weekend gaps and market holidays may cause minor differences vs real broker HTF candles
- **Session detection** — MQL5 `TimeToStruct` uses real timestamps with configurable UTC offset but may not perfectly match broker server timezone in all cases
- **Performance** — complex strategies on 50K+ bars take 1-2 minutes even with Cython. The interpreter is tree-walking, not compiled to native code

## Roadmap

- [ ] **Monte Carlo simulation** — randomize trade order 1000x to show range of possible equity curves and probability of ruin
- [ ] **Parameter heatmap** — visual 2D heatmap of optimization results for two-parameter sweeps
- [ ] **Rolling Sharpe chart** — plot Sharpe ratio over a rolling window to see when the strategy works vs doesn't
- [ ] **Trade duration histogram** — distribution of holding periods, winners vs losers by duration
- [ ] **Underwater equity chart** — visual timeline of drawdown periods with recovery time
- [ ] **Compare runs** — save multiple backtest results and overlay equity curves side-by-side
- [ ] **API data fetching** — pull OHLCV data directly from brokers (MT5 API, TradingView, OANDA)
- [ ] **Strategy templates** — pre-built strategies selectable from a dropdown
- [ ] **Correlation matrix** — show how correlated returns are when running multiple strategies
- [ ] **Risk of ruin calculator** — estimate probability of account blowup from win rate and risk per trade
- [ ] **WebSocket live mode** — connect to a live data feed for paper trading

## License

MIT
