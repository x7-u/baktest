# Trading Strategy Backtester

A web-based backtesting engine that runs **Pine Script v5** and **MQL5 Expert Advisor** strategies against your own OHLCV data. No TradingView or MetaTrader account needed.

## Changelog

### v3.1 — Multi-Timeframe, Walk-Forward, Accuracy Upgrade

**Multi-Timeframe Support**
- Select your base timeframe (1m to 1M) when uploading CSV data
- The engine auto-aggregates higher timeframe candles (H1, H4, D1, W1) from the base data using real timestamp boundaries
- Pine Script: `request.security(syminfo.tickerid, "240", high)` returns proper H4 high
- MQL5: `iHigh(_Symbol, PERIOD_H4, 1)` returns previous H4 candle high
- No more manual ratio-based approximation — real OHLC aggregation

**Walk-Forward Optimization**
- Rolling train/test window parameter sweep
- Configure train size, test size, step, and optimization metric (Sharpe, PF, Net Profit, Win Rate)
- Define parameter ranges via JSON: `[{"name":"length","start":5,"end":50,"step":5}]`
- Results table shows per-window best params + out-of-sample performance

**Accuracy Improvements**
- **Spread simulation** — configurable bid/ask spread in pips. Longs fill at ask, shorts at bid
- **Slippage modeling** — random 0 to N pips adverse fill on every entry/exit
- **Commission per lot** — $X per lot per side (in addition to percentage-based commission)
- **Intra-bar SL/TP priority** — when both stop loss and take profit could hit on the same bar, uses candle direction heuristic to determine which triggered first
- **R-multiple tracking** — each trade's P&L expressed as a multiple of its initial risk (SL distance)

**Analytics**
- **Monthly returns table** — P&L, return %, trade count, and win rate per month
- **R-multiple distribution** — histogram chart of trade results as risk multiples
- **Exposure time** — percentage of bars with an open position
- **Drawdown duration** — longest consecutive period in drawdown (bars)
- **R column in trade log** — see each trade's R-multiple alongside P&L

**Performance**
- **Vectorized indicators** — NumPy for RSI and ATR calculations
- **TA memoization** — per-bar cache prevents redundant indicator computation
- **Lazy series trimming** — series data capped at 500 bars, 200x memory reduction on 100K+ bar backtests

**New Order Types**
- **Pending limit/stop entries** — Pine: `strategy.entry("L", strategy.long, limit=1.1000)`, MQL5: `BuyLimit`, `SellStop`, etc.
- **Portfolio mode** — hold multiple positions simultaneously, keyed by trade ID

**Usability**
- **Progress bar** — real-time SSE streaming shows bar count during long backtests
- **Auto-save** — script, settings, engine choice, and base timeframe persist in localStorage
- **Ctrl+Enter** — keyboard shortcut to run backtest
- **Syntax error line jumping** — "Go to Line X" button that highlights the error in the editor
- **Metric tooltips** — hover any metric name for a plain-English explanation
- **CSV export** — download trade log, monthly returns, metrics, or equity curve
- **Multi-symbol upload** — up to 5 correlation symbol CSVs

### v3.0 — Dual-Engine Architecture

- Added MQL5 Expert Advisor interpreter (tokenizer, parser, tree-walking interpreter)
- Unified Pine Script + MQL5 in a single app with engine toggle
- Cython acceleration on both engines (~5x speedup)
- Platform-level SMT divergence (works with any strategy, no script code needed)
- JPY pair auto-detection with P&L currency conversion

### v2.0 — Cython Engine

- Cython-optimized Pine Script interpreter (~5x speedup)
- SMT divergence via `request.security()` with secondary CSV upload
- Customizable bar count, clearer settings labels

### v1.0 — Initial Release

- Pine Script v5 interpreter with 30+ built-in indicators
- Flask web UI with candlestick charts, equity curve, drawdown, trade log
- Full strategy support: `strategy.entry/exit/close` with SL/TP

## Features

- **Two interpreters** — Pine Script v5 and MQL5 Expert Advisors
- **Multi-timeframe** — select base TF, auto-aggregates H1/H4/D1/W1 candles from base data
- **Cython acceleration** — ~5x speedup on both engines
- **Walk-forward optimization** — rolling train/test parameter sweep
- **Spread + slippage + commission** — realistic execution modeling
- **Portfolio mode** — multiple simultaneous positions
- **Pending orders** — limit and stop entry orders
- **Platform SMT divergence** — works with any strategy
- **30+ built-in indicators** — SMA, EMA, RSI, ATR, MACD, BB, Stochastic, pivots, crossover
- **45+ metrics** — Sharpe, Sortino, Calmar, R-multiples, MAE/MFE, exposure, DD duration
- **Monthly returns** — P&L breakdown by month with win rate
- **Interactive charts** — candlestick with trade markers, equity, drawdown, P&L distribution, R-multiple histogram
- **CSV export** — download trades, metrics, monthly returns, equity curve
- **Auto-save** — scripts and settings persist across refreshes
- **Progress bar** — real-time bar count during long backtests
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
2. **Upload CSV** — drop your MetaTrader/TradingView OHLCV export
3. **Select Timeframe** — choose the base timeframe of your data (M5, H1, etc.)
4. **Paste Strategy** — enter your Pine Script or MQL5 EA code
5. **(Optional) SMT Data** — upload a second CSV for divergence detection
6. **Configure** — set balance, commission, spread, slippage, lot size, bar count
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
| `ta.sma/ema/rsi/atr/macd/bb/wma` | Supported |
| `ta.crossover/crossunder` | Supported |
| `ta.highest/lowest/change/stoch` | Supported |
| `ta.pivothigh/pivotlow` | Supported |
| `request.security()` (multi-timeframe + SMT) | Supported |
| `math.*`, `array.*`, `str.*` | Supported |
| Platform SMT: `_smt_bull_active`, `_smt_bear_active` | Supported |
| Visual functions (plot, label, line) | Parsed (no-op) |

### MQL5 Expert Advisors
| Feature | Status |
|---------|--------|
| `OnInit/OnTick/OnDeinit` | Supported |
| `CTrade` (Buy/Sell/Close/Modify/BuyLimit/SellStop) | Supported |
| `iMA/iRSI/iMACD/iATR/iBands/iStoch` | Supported |
| `CopyBuffer` | Supported |
| `PositionSelect/Get` queries | Supported |
| `input` variables | Supported |
| `#include`, `#define`, `#property` | Supported |
| `if/else`, `for`, `while`, `do-while`, `switch` | Supported |
| User-defined functions | Supported |
| `ArrayResize/Copy/SetAsSeries` | Supported |
| `SymbolInfoDouble/Integer/String` | Supported |
| `TimeToStruct` (real timestamps) | Supported |
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
  test/
    SMC_Framework_v5.11_Strategy.pine
    SMC_Framework_v511.mq5
  examples/
    sample_ea.mq5
```

## Limitations

- **Bar-by-bar execution** — backtests run on OHLCV candles, not tick-by-tick. Intra-bar price paths are approximated with a candle direction heuristic
- **Pine Script subset** — not all v5 features are supported (e.g., `ta.valuewhen`, `ta.barssince`, drawing objects are no-ops)
- **MQL5 subset** — no struct/class definitions, no custom indicators (`iCustom` is a stub), no order book
- **MTF approximation** — higher timeframe candles are aggregated from base data. Weekend gaps, session breaks, and market holidays may cause minor differences vs real broker HTF candles
- **Session detection** — MQL5 `TimeToStruct` uses real timestamps from CSV datetime columns but may not perfectly match broker server timezone
- **Performance** — complex strategies (e.g., SMC Framework with 1500 lines) on 50K+ bars take 1-2 minutes even with Cython. The interpreter is tree-walking, not compiled to native code
- **Single CSV per symbol** — each symbol requires a separate CSV upload. No API-based data fetching

## Roadmap

- [ ] **Monte Carlo simulation** — randomize trade order 1000x to show range of possible equity curves and probability of ruin
- [ ] **Parameter heatmap** — visual 2D heatmap of optimization results for two-parameter sweeps
- [ ] **Rolling Sharpe chart** — plot Sharpe ratio over a rolling window to see when the strategy works vs. doesn't
- [ ] **Trade duration histogram** — distribution of holding periods, winners vs. losers by duration
- [ ] **Underwater equity chart** — visual timeline of drawdown periods with recovery time
- [ ] **Compare runs** — save multiple backtest results and overlay equity curves side-by-side
- [ ] **Dark/light theme toggle** — CSS variable system already supports it, just needs a switch
- [ ] **Drag-resize panels** — let users resize the sidebar and chart areas
- [ ] **API data fetching** — pull OHLCV data directly from brokers (MT5 API, TradingView, OANDA)
- [ ] **Strategy templates** — pre-built strategies (MA crossover, RSI mean reversion, breakout) selectable from a dropdown
- [ ] **Proper Wilder's ATR/RSI** — use exponential smoothing instead of simple average for more accurate indicator values
- [ ] **Correlation matrix** — show how correlated returns are when running multiple strategies
- [ ] **Risk of ruin calculator** — given win rate, risk per trade, and payoff ratio, estimate probability of account blowup
- [ ] **WebSocket live mode** — connect to a live data feed and run strategies in real-time (paper trading)

## License

MIT
