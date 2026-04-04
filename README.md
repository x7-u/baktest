# Trading Strategy Backtester

A web-based backtesting engine that runs **Pine Script v5** and **MQL5 Expert Advisor** strategies against your own OHLCV data. No TradingView or MetaTrader account needed.

## What's New in v3.0

### Dual-Engine Architecture
- **Pine Script v5** and **MQL5** interpreters in a single app
- Toggle between engines with one click in the sidebar
- Both engines run on **Cython** for ~5x speedup over pure Python
- Automatic fallback to Python if Cython isn't compiled

### MQL5 Expert Advisor Support
- Full MQL5 tokenizer, parser, and tree-walking interpreter (~2000 lines)
- Supports `OnInit()` / `OnTick()` / `OnDeinit()` lifecycle
- **CTrade class**: `Buy()`, `Sell()`, `PositionClose()`, `PositionModify()`
- **Indicator handles**: `iMA`, `iRSI`, `iMACD`, `iATR`, `iBands`, `iStochastic` + `CopyBuffer`
- **Position queries**: `PositionSelect()`, `PositionGetDouble()`, `PositionGetInteger()`
- **Multi-timeframe**: `iHigh/iLow/iClose` with timeframe aggregation from base data
- **Session detection**: `TimeToStruct()` with real timestamps, killzone filtering
- `input` variables, `#define` constants, `enum` declarations, C-style control flow
- Preprocessor: `#include <Trade/Trade.mqh>` enables CTrade automatically

### Platform-Level SMT Divergence
- Upload a second CSV for any correlated symbol
- The platform automatically detects swing high/low divergence between both symbols
- **No SMT code needed in your strategy** -- any script can use it
- Variables injected per bar: `_smt_bull_active`, `_smt_bear_active`, `_smt_available`
- Works identically on both Pine Script and MQL5 engines

### JPY Pair Support
- Auto-detects JPY pairs from price levels (>10)
- P&L automatically converted from JPY to USD
- Correct lot sizing with contract size normalization

## Features

- **Two interpreters** -- Pine Script v5 and MQL5 Expert Advisors
- **Cython acceleration** -- ~5x speedup on both engines
- **Full strategy support** -- entry, exit, close, stop-loss, take-profit
- **30+ built-in indicators** -- SMA, EMA, RSI, ATR, MACD, Bollinger Bands, Stochastic, pivot highs/lows, crossover/crossunder
- **SMT Divergence** -- platform-level, works with any strategy
- **Interactive charts** -- candlestick chart with trade markers, equity curve, drawdown
- **40+ metrics** -- win rate, profit factor, Sharpe, Sortino, Calmar, max drawdown, MAE/MFE
- **Configurable** -- starting balance, commission, lot size, bar count
- **One-click launch** -- `start.bat` builds Cython and opens browser

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

Or double-click **`start.bat`** (Windows) -- builds Cython automatically and opens the browser.

Open **http://localhost:1234** in your browser.

### 5. Use it

1. **Select Engine** -- click "Pine Script" or "MQL5" in the sidebar
2. **Upload CSV** -- drop your MetaTrader/TradingView OHLCV export
3. **Paste Strategy** -- enter your Pine Script or MQL5 EA code
4. **(Optional) SMT Data** -- upload a second CSV for divergence detection
5. **Configure** -- set balance, commission, lot size, bar count
6. **Run Backtest** -- click the button and view results

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
| `strategy.entry/exit/close` | Supported |
| `input.int/float/bool/string` | Supported |
| `if/else`, `for`, `switch`, ternary | Supported |
| User-defined functions | Supported |
| `var` / `varip` declarations | Supported |
| History reference `close[1]` | Supported |
| `ta.sma/ema/rsi/atr/macd/bb/wma` | Supported |
| `ta.crossover/crossunder` | Supported |
| `ta.highest/lowest/change/stoch` | Supported |
| `ta.pivothigh/pivotlow` | Supported |
| `request.security()` (SMT) | Supported |
| `math.*`, `array.*`, `str.*` | Supported |
| Visual functions (plot, label, line) | Parsed (no-op) |

### MQL5 Expert Advisors
| Feature | Status |
|---------|--------|
| `OnInit/OnTick/OnDeinit` | Supported |
| `CTrade` (Buy/Sell/Close/Modify) | Supported |
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
| Multi-timeframe `iHigh/iLow` | Supported (aggregated from base TF) |
| `MathSqrt/Abs/Max/Min/Pow/Log` | Supported |
| `StringFormat/Len/Find/Substr` | Supported |
| `NormalizeDouble`, type casting | Supported |

## Project Structure

```
baktest/
  app.py               # Flask web server (unified, port 1234)
  backtester.py         # Trade execution engine (dual-engine)
  pine_parser.py        # Pine Script tokenizer, parser, interpreter
  pine_fast.pyx         # Cython Pine Script interpreter (~5x faster)
  mql5_parser.py        # MQL5 tokenizer, parser, interpreter
  mql5_fast.pyx         # Cython MQL5 interpreter (~5x faster)
  smt_engine.py         # Platform-level SMT divergence detection
  setup.py              # Cython build script (both engines)
  start.bat             # One-click Windows launcher
  requirements.txt      # Python dependencies
  templates/
    index.html          # Web UI (single-page app with engine toggle)
  test/
    SMC_Framework_v5.11_Strategy.pine   # Test Pine Script strategy
    SMC_Framework_v511.mq5              # Test MQL5 EA (same strategy)
  examples/
    sample_ea.mq5       # Simple MA crossover MQL5 EA
```

## Limitations

- **No multi-timeframe data** -- both engines use a single CSV for price data. Multi-timeframe indicators (e.g., `iHigh` with H4 on M5 data) are simulated by aggregating base timeframe bars, which may not perfectly match real HTF candles
- **No tick data** -- backtests run bar-by-bar on OHLCV data, not tick-by-tick. Intra-bar price movement is approximated
- **No spread simulation** -- Bid/Ask spread is not modeled; entries and exits use the close price
- **No slippage** -- orders fill at exact prices (stop/limit levels or bar close)
- **No pyramiding** -- only one position open at a time per direction
- **Pine Script subset** -- not all Pine Script v5 features are supported (e.g., `ta.valuewhen`, `ta.barssince`, drawing objects are no-ops)
- **MQL5 subset** -- no struct/class definitions, no custom indicators (`iCustom` is a stub), no order book, no pending orders (limit/stop orders are SL/TP only)
- **Session detection** -- MQL5 session functions use synthetic timestamps derived from CSV datetime columns; may not perfectly match real broker server times
- **Performance** -- complex strategies on 50K+ bars can take 1-2 minutes even with Cython. The interpreter is tree-walking, not compiled

## Roadmap

- [ ] **Progress bar** -- show bar count progress during long backtests
- [ ] **Auto-save scripts** -- persist strategy code in localStorage across refreshes
- [ ] **Syntax error highlighting** -- point to the exact line with the error
- [ ] **Ctrl+Enter to run** -- keyboard shortcut for running backtests
- [ ] **Export results** -- download trade list and equity curve as CSV
- [ ] **Metric tooltips** -- hover explanations for Sharpe, Sortino, profit factor
- [ ] **Vectorized indicators** -- use numpy for SMA/EMA/RSI instead of per-bar loops
- [ ] **TA memoization** -- cache repeated indicator calls with same parameters
- [ ] **Lazy series trimming** -- free memory for old bars beyond max lookback
- [ ] **Walk-forward optimization** -- split data into train/test windows
- [ ] **Multi-symbol support** -- backtest across multiple instruments simultaneously
- [ ] **Pending order types** -- buy/sell limit and stop orders (not just SL/TP)
- [ ] **Portfolio mode** -- track multiple positions across different strategies

## License

MIT
