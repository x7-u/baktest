"""
Unified backtesting engine: executes Pine Script or MQL5 EA signals
against market data and tracks positions, P&L, and equity curve.

v3.1 — Accuracy & analytics upgrade:
  - Spread simulation (configurable pips)
  - Slippage modeling (random range)
  - Commission models (percentage, per-lot, per-trade)
  - Intra-bar SL/TP priority (candle direction heuristic)
  - Monthly returns table
  - R-multiple distribution
  - Underwater equity (drawdown duration)
  - Exposure time tracking
"""

import pandas as pd
import numpy as np
import random
from smt_engine import SMTEngine
from mtf_engine import MTFEngine


def _get_create_interpreter(engine):
    if engine == 'mql5':
        from mql5_parser import create_interpreter, is_na, NA
    else:
        from pine_parser import create_interpreter, is_na, NA
    return create_interpreter, is_na, NA


class Trade:
    __slots__ = ('direction', 'entry_bar', 'entry_price', 'exit_bar',
                 'exit_price', 'qty', 'pnl', 'pnl_pct', 'comment', 'exit_reason',
                 'entry_date', 'exit_date', 'bars_held', 'mae', 'mfe',
                 'sl_price', 'tp_price', 'r_multiple')

    def __init__(self, direction, entry_bar, entry_price, qty, comment='', entry_date=None,
                 sl_price=None, tp_price=None):
        self.direction = direction
        self.entry_bar = entry_bar
        self.entry_price = entry_price
        self.exit_bar = None; self.exit_price = None
        self.qty = qty; self.pnl = 0.0; self.pnl_pct = 0.0
        self.comment = comment; self.entry_date = entry_date; self.exit_date = None
        self.bars_held = 0; self.mae = 0.0; self.mfe = 0.0
        self.exit_reason = ''  # 'tp', 'sl', 'signal', 'opposite', 'end'
        self.sl_price = sl_price; self.tp_price = tp_price
        self.r_multiple = 0.0

    def close(self, exit_bar, exit_price, commission_cost=0.0, exit_date=None, pnl_conversion=1.0):
        self.exit_bar = exit_bar; self.exit_price = exit_price
        self.exit_date = exit_date; self.bars_held = exit_bar - self.entry_bar
        if self.direction == 'long':
            raw = (exit_price - self.entry_price) * self.qty
        else:
            raw = (self.entry_price - exit_price) * self.qty
        raw *= pnl_conversion
        self.pnl = raw - commission_cost
        entry_value = self.entry_price * self.qty * pnl_conversion
        self.pnl_pct = (self.pnl / entry_value) * 100 if entry_value != 0 else 0
        # R-multiple: P&L as multiple of initial risk
        if self.sl_price and self.sl_price > 0:
            risk = abs(self.entry_price - self.sl_price) * self.qty * pnl_conversion
            self.r_multiple = round(self.pnl / risk, 2) if risk > 0 else 0
        else:
            self.r_multiple = 0

    def to_dict(self, engine='pine'):
        qty_display = self.qty
        if engine == 'mql5' and self.qty >= 1000:
            qty_display = round(self.qty / 100000, 2)
        return {
            'direction': self.direction,
            'entry_bar': self.entry_bar, 'entry_price': round(self.entry_price, 5),
            'exit_bar': self.exit_bar, 'exit_price': round(self.exit_price, 5) if self.exit_price else None,
            'qty': qty_display, 'pnl': round(self.pnl, 2), 'pnl_pct': round(self.pnl_pct, 2),
            'comment': self.comment,
            'entry_date': str(self.entry_date) if self.entry_date else None,
            'exit_date': str(self.exit_date) if self.exit_date else None,
            'bars_held': self.bars_held, 'mae': round(self.mae, 2), 'mfe': round(self.mfe, 2),
            'r_multiple': self.r_multiple,
            'exit_reason': self.exit_reason,
        }


class OptionTrade:
    """An options contract trade (call or put)."""
    __slots__ = ('contract_type', 'strike', 'expiry', 'direction', 'entry_bar',
                 'entry_price', 'exit_bar', 'exit_price', 'qty', 'multiplier',
                 'pnl', 'pnl_pct', 'comment', 'exit_reason', 'entry_date',
                 'exit_date', 'bars_held', 'greeks_at_entry', 'spread_id',
                 'last_known_mark', 'entry_commission')

    def __init__(self, contract_type, strike, expiry, direction, entry_bar,
                 entry_price, qty=1, multiplier=100, comment='', entry_date=None,
                 spread_id=None):
        self.contract_type = contract_type  # 'call' or 'put'
        self.strike = strike
        self.expiry = expiry                # pd.Timestamp or None
        self.direction = direction          # 'long' (bought) or 'short' (sold/written)
        self.entry_bar = entry_bar
        self.entry_price = entry_price      # premium per contract at entry
        self.exit_bar = None; self.exit_price = None
        self.qty = qty; self.multiplier = multiplier
        self.pnl = 0.0; self.pnl_pct = 0.0
        self.comment = comment; self.entry_date = entry_date; self.exit_date = None
        self.bars_held = 0; self.exit_reason = ''
        self.greeks_at_entry = {}           # {delta, gamma, theta, vega, iv}
        self.spread_id = spread_id          # links legs of multi-leg strategies
        self.last_known_mark = entry_price  # fallback for mark-to-market
        self.entry_commission = 0.0         # stored for accurate pnl reporting

    def close(self, exit_bar, exit_price, commission_cost=0.0, exit_date=None):
        self.exit_bar = exit_bar; self.exit_price = exit_price
        self.exit_date = exit_date; self.bars_held = exit_bar - self.entry_bar
        if self.direction == 'long':
            raw = (exit_price - self.entry_price) * self.qty * self.multiplier
        else:
            raw = (self.entry_price - exit_price) * self.qty * self.multiplier
        self.pnl = raw - commission_cost
        entry_value = self.entry_price * self.qty * self.multiplier
        self.pnl_pct = (self.pnl / entry_value) * 100 if entry_value != 0 else 0

    def intrinsic_value(self, underlying_price):
        """Intrinsic value at expiry."""
        if self.contract_type == 'call':
            return max(underlying_price - self.strike, 0)
        else:  # put
            return max(self.strike - underlying_price, 0)

    def to_dict(self):
        return {
            'trade_type': 'option',
            'contract_type': self.contract_type,
            'strike': round(self.strike, 2),
            'expiry': str(self.expiry.date()) if self.expiry is not None and str(self.expiry) != 'NaT' else None,
            'direction': self.direction,
            'entry_bar': self.entry_bar,
            'entry_price': round(self.entry_price, 4),
            'exit_bar': self.exit_bar,
            'exit_price': round(self.exit_price, 4) if self.exit_price is not None else None,
            'qty': self.qty,
            'multiplier': self.multiplier,
            'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct, 2),
            'comment': self.comment,
            'entry_date': str(self.entry_date) if self.entry_date else None,
            'exit_date': str(self.exit_date) if self.exit_date else None,
            'bars_held': self.bars_held,
            'exit_reason': self.exit_reason,
            'spread_id': self.spread_id,
        }


class PendingExit:
    def __init__(self, stop=None, limit=None, from_entry='', comment=''):
        self.stop = stop; self.limit = limit
        self.from_entry = from_entry; self.comment = comment


class PendingEntry:
    def __init__(self, direction, qty, price, order_type, sl=None, tp=None, comment='', created_bar=0):
        self.direction = direction; self.qty = qty; self.price = price
        self.order_type = order_type  # 'limit' or 'stop'
        self.sl = sl; self.tp = tp; self.comment = comment
        self.created_bar = created_bar


class Backtester:
    def __init__(self, data: pd.DataFrame, source: str, engine: str = 'pine',
                 initial_capital: float = 10000,
                 commission_pct: float = 0.0,
                 commission_per_lot: float = 0.0,
                 commission_per_trade: float = 0.0,
                 default_qty: float = 1.0,
                 risk_pct: float = 0.0,
                 spread_pips: float = 0.0,
                 slippage_pips: float = 0.0,
                 smt_data: pd.DataFrame = None,
                 extra_data=None,
                 base_tf: str = 'M5',
                 utc_offset: int = 0,
                 date_filters: list = None,
                 symbol_name: str = '',
                 funded_rules: dict = None,
                 options_chain=None):
        self.data = data
        self.smt_data = smt_data
        self.options_chain = options_chain
        self.extra_data = extra_data or []
        self.base_tf = base_tf
        self.utc_offset = utc_offset
        self.date_filters = date_filters or []
        self.risk_pct = risk_pct
        self.symbol_name = symbol_name.upper()
        self.source = source
        self.engine = engine
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.commission_per_lot = commission_per_lot
        self.commission_per_trade = commission_per_trade
        self.default_qty = default_qty
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.funded_rules = funded_rules
        self.trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.open_position = None
        self.open_positions = {}   # {trade_id: Trade} for portfolio mode
        self._next_trade_id = 0
        self.pending_exits: dict = {}  # {trade_id: PendingExit}
        self.pending_entries: list[PendingEntry] = []
        self.equity_curve = []
        self.balance_curve = []
        self.drawdown_curve = []
        # Options portfolio (coexists with equity positions)
        self.open_options: dict = {}     # {tid: OptionTrade}
        self.closed_options: list = []   # completed option trades
        self._next_option_tid = 0
        self._next_spread_id = 0
        self._options_unrealized = 0.0   # cached per-bar

    def _calc_spread(self):
        """Half spread in price units (applied to entry/exit)."""
        return self.spread_pips * self._pip_size / 2

    def _calc_slippage(self):
        """Random slippage in price units."""
        if self.slippage_pips <= 0: return 0.0
        return random.uniform(0, self.slippage_pips) * self._pip_size

    def _apply_entry_price(self, base_price, direction):
        """Apply spread + slippage to entry price (worse for trader)."""
        spread = self._calc_spread()
        slip = self._calc_slippage()
        if direction == 'long':
            return base_price + spread + slip  # buy at ask + slip
        else:
            return base_price - spread - slip  # sell at bid - slip

    def _apply_exit_price(self, base_price, direction):
        """Apply spread + slippage to exit price (worse for trader)."""
        spread = self._calc_spread()
        slip = self._calc_slippage()
        if direction == 'long':
            return base_price - spread - slip  # sell at bid - slip
        else:
            return base_price + spread + slip  # buy at ask + slip

    def _calc_risk_qty(self, entry_price, sl_price, balance):
        """Calculate position size based on risk % of equity.

        If risk_pct is set and the trade has a SL, size the position so that
        hitting the SL loses exactly risk_pct% of current equity.

        Returns qty in units, or None if risk sizing not applicable.
        """
        if self.risk_pct <= 0 or not sl_price:
            return None

        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0:
            return None

        risk_amount = balance * self.risk_pct / 100.0
        # qty = risk_amount / (sl_distance * pnl_conversion)
        # pnl_conversion converts price-unit P&L to account currency
        qty = risk_amount / (sl_dist * self._pnl_conversion)

        if qty <= 0:
            return None

        return qty

    def _calc_commission(self, entry_price, exit_price, qty):
        """Calculate total commission for a round-trip trade."""
        cost = 0.0
        # Percentage-based
        if self.commission_pct > 0:
            cost += (entry_price * qty + exit_price * qty) * self.commission_pct / 100
        # Per-lot (MQL5 style)
        if self.commission_per_lot > 0:
            lots = qty / 100000 if qty >= 1000 else qty
            cost += self.commission_per_lot * lots * 2  # entry + exit
        # Per-trade flat fee: charged on both entry and exit (2x per round-trip)
        if self.commission_per_trade > 0:
            cost += self.commission_per_trade * 2
        return cost * self._pnl_conversion

    def _add_position(self, trade):
        """Add a trade to portfolio, update open_position for backwards compat."""
        tid = self._next_trade_id
        self._next_trade_id += 1
        self.open_positions[tid] = trade
        self.open_position = trade
        return tid

    def _close_position(self, tid, bar_idx, exit_price, current_date, balance, is_na, reason='signal'):
        """Close a position by trade ID, return updated balance."""
        trade = self.open_positions.pop(tid)
        trade.exit_reason = reason
        exit_price = self._apply_exit_price(exit_price, trade.direction)
        comm = self._calc_commission(trade.entry_price, exit_price, trade.qty)
        trade.close(bar_idx, exit_price, comm, current_date, self._pnl_conversion)
        balance += trade.pnl
        self.closed_trades.append(trade)
        # Update backwards-compat pointer
        if self.open_positions:
            self.open_position = list(self.open_positions.values())[-1]
        else:
            self.open_position = None
        return balance

    # ── Options execution methods ──────────────────────────────────────────

    def _find_option_contract(self, chain, strike, contract_type, expiry_dte, bar_date):
        """Find the best-matching option contract from the chain for the current bar."""
        import pandas as _pd
        if chain is None or len(chain) == 0:
            return None
        filtered = chain[(chain['type'] == contract_type) & (chain['strike'] == strike)]
        if len(filtered) == 0:
            # Nearest strike fallback
            typed = chain[chain['type'] == contract_type]
            if len(typed) == 0:
                return None
            typed = typed.copy()
            typed['_sdist'] = (typed['strike'] - strike).abs()
            filtered = typed.nsmallest(5, '_sdist')
        filtered = filtered.copy()
        filtered['_dte'] = (_pd.to_datetime(filtered['expiration']) - _pd.Timestamp(bar_date)).dt.days
        filtered['_dte_dist'] = (filtered['_dte'] - expiry_dte).abs()
        best = filtered.nsmallest(1, '_dte_dist')
        return best.iloc[0] if len(best) > 0 else None

    def _open_option(self, sig, bar_idx, current_date, balance, interpreter):
        """Open a single-leg option position. Returns updated balance."""
        import pandas as _pd
        date_key = interpreter.variables.get('_options_date_key', '')
        opts_by_date = interpreter.variables.get('__options_by_date__', {})
        chain = opts_by_date.get(date_key)
        if chain is None or len(chain) == 0:
            return balance  # no options data for this date

        contract = self._find_option_contract(
            chain, sig.strike, sig.contract_type, sig.expiry_dte, date_key)
        if contract is None:
            return balance

        # Entry price: buy at ask, sell at bid (realistic fills)
        def _get_premium(row, *keys):
            for k in keys:
                v = row.get(k)
                if _pd.notna(v) and float(v) > 0:
                    return float(v)
            return 0.0
        if sig.direction == 'long':
            premium = _get_premium(contract, 'ask', 'mark', 'last')
        else:
            premium = _get_premium(contract, 'bid', 'mark', 'last')
        if premium <= 0:
            return balance  # can't price this contract

        expiry = _pd.to_datetime(contract.get('expiration')) if _pd.notna(contract.get('expiration')) else None
        qty = int(sig.qty) if sig.qty else 1
        multiplier = 100  # standard US equity options

        trade = OptionTrade(
            contract_type=sig.contract_type,
            strike=float(contract['strike']),
            expiry=expiry,
            direction=sig.direction,
            entry_bar=bar_idx,
            entry_price=premium,
            qty=qty,
            multiplier=multiplier,
            comment=sig.comment,
            entry_date=current_date,
            spread_id=getattr(sig, 'spread_id', None),
        )
        # Snapshot greeks at entry
        for g in ('delta', 'gamma', 'theta', 'vega', 'implied_volatility'):
            v = contract.get(g)
            if _pd.notna(v):
                trade.greeks_at_entry[g] = float(v)

        # Debit/credit premium from balance
        cost = premium * qty * multiplier
        comm = self._calc_option_commission(premium, qty, multiplier)
        trade.entry_commission = comm
        if sig.direction == 'long':
            balance -= cost + comm  # pay premium + commission
        else:
            balance += cost - comm  # receive premium - commission

        tid = self._next_option_tid
        self._next_option_tid += 1
        self.open_options[tid] = trade
        return balance

    def _close_option_trade(self, tid, bar_idx, current_date, balance, interpreter, reason='signal'):
        """Close a single option position by trade ID. Returns updated balance."""
        import pandas as _pd
        trade = self.open_options.pop(tid)
        trade.exit_reason = reason

        # Find current mark for exit price
        exit_price = self._get_option_mark(trade, interpreter)
        if exit_price is None:
            exit_price = trade.last_known_mark  # fallback

        exit_comm = self._calc_option_commission(exit_price, trade.qty, trade.multiplier)
        total_comm = trade.entry_commission + exit_comm
        trade.close(bar_idx, exit_price, total_comm, current_date)

        # Credit/debit exit premium
        exit_value = exit_price * trade.qty * trade.multiplier
        if trade.direction == 'long':
            balance += exit_value - exit_comm   # sell to close → receive premium
        else:
            balance -= exit_value + exit_comm   # buy to close → pay premium

        self.closed_options.append(trade)
        return balance

    def _close_option_by_id(self, comment, bar_idx, current_date, balance, interpreter):
        """Close all option positions matching a comment/label."""
        to_close = [tid for tid, t in self.open_options.items() if t.comment == comment]
        for tid in to_close:
            balance = self._close_option_trade(tid, bar_idx, current_date, balance, interpreter)
        return balance

    def _close_all_options(self, bar_idx, current_date, balance, interpreter):
        """Close all open option positions."""
        for tid in list(self.open_options.keys()):
            balance = self._close_option_trade(tid, bar_idx, current_date, balance, interpreter)
        return balance

    def _expire_option(self, tid, bar_idx, current_date, underlying_price, balance):
        """Handle option expiration — exercise if ITM, expire worthless if OTM."""
        trade = self.open_options.pop(tid)
        intrinsic = trade.intrinsic_value(underlying_price)

        if intrinsic > 0:
            # ITM — exercise
            trade.exit_reason = 'exercise'
            exit_premium = intrinsic
        else:
            # OTM — expires worthless
            trade.exit_reason = 'expiry'
            exit_premium = 0.0

        trade.close(bar_idx, exit_premium, trade.entry_commission, current_date)

        # Settle cash
        exit_value = exit_premium * trade.qty * trade.multiplier
        if trade.direction == 'long':
            balance += exit_value   # receive intrinsic value
        else:
            balance -= exit_value   # pay intrinsic value to holder

        self.closed_options.append(trade)
        return balance

    def _get_option_mark(self, trade, interpreter):
        """Look up current market price for an open option position."""
        import pandas as _pd
        date_key = interpreter.variables.get('_options_date_key', '')
        opts_by_date = interpreter.variables.get('__options_by_date__', {})
        chain = opts_by_date.get(date_key)
        if chain is None or len(chain) == 0:
            return None
        # Find matching contract: same strike, type, expiry
        match = chain[(chain['type'] == trade.contract_type) &
                      (chain['strike'] == trade.strike)]
        if trade.expiry is not None:
            exp_match = match[_pd.to_datetime(match['expiration']) == trade.expiry]
            if len(exp_match) > 0:
                match = exp_match
        if len(match) == 0:
            return None
        row = match.iloc[0]
        mark = row.get('mark')
        if _pd.notna(mark) and float(mark) > 0:
            trade.last_known_mark = float(mark)
            return float(mark)
        last = row.get('last')
        if _pd.notna(last) and float(last) > 0:
            trade.last_known_mark = float(last)
            return float(last)
        return None

    def _get_option_greek(self, trade, greek_name, interpreter):
        """Look up current greek value for an open option position."""
        import pandas as _pd
        date_key = interpreter.variables.get('_options_date_key', '')
        opts_by_date = interpreter.variables.get('__options_by_date__', {})
        chain = opts_by_date.get(date_key)
        if chain is None or len(chain) == 0:
            return 0
        match = chain[(chain['type'] == trade.contract_type) &
                      (chain['strike'] == trade.strike)]
        if trade.expiry is not None:
            exp_match = match[_pd.to_datetime(match['expiration']) == trade.expiry]
            if len(exp_match) > 0:
                match = exp_match
        if len(match) == 0:
            return 0
        val = match.iloc[0].get(greek_name)
        return float(val) if _pd.notna(val) else 0

    def _calc_option_commission(self, premium, qty, multiplier):
        """Calculate commission for an options trade."""
        cost = 0.0
        if self.commission_per_trade > 0:
            cost += self.commission_per_trade * qty  # per-contract fee
        if self.commission_pct > 0:
            cost += premium * qty * multiplier * self.commission_pct / 100
        return cost

    def _open_spread(self, sig, bar_idx, current_date, balance, interpreter):
        """Open a multi-leg spread (vertical, condor, straddle, strangle)."""
        spread_id = f"spread_{self._next_spread_id}"
        self._next_spread_id += 1
        legs = getattr(sig, 'legs', [])
        for leg in legs:
            # Create a simple namespace for the leg signal (avoid importing Signal)
            class _LegSig:
                pass
            s = _LegSig()
            s.action = 'entry_option'
            s.direction = leg['direction']
            s.qty = sig.qty
            s.comment = sig.comment
            s.strike = leg['strike']
            s.contract_type = leg['type']
            s.expiry_dte = getattr(sig, 'expiry_dte', 30)
            s.spread_id = spread_id
            balance = self._open_option(s, bar_idx, current_date, balance, interpreter)
        return balance

    def _close_spread_by_id(self, spread_id, bar_idx, current_date, balance, interpreter):
        """Close all legs of a spread."""
        to_close = [tid for tid, t in self.open_options.items() if t.spread_id == spread_id]
        for tid in to_close:
            balance = self._close_option_trade(tid, bar_idx, current_date, balance, interpreter)
        return balance

    def _run_loop(self, streaming=False, yield_every=500):
        """Core backtest loop. If streaming=True, yields progress dicts."""
        create_interpreter, is_na, NA = _get_create_interpreter(self.engine)
        self.interpreter = interpreter = create_interpreter(self.source)

        # Set deterministic random seed for reproducible slippage
        import random as _random
        _random.seed(42)

        interpreter.setup(self.data)
        if self.smt_data is not None:
            interpreter.setup_secondary(self.smt_data)

        # Build multi-timeframe data
        self._mtf = MTFEngine(base_tf=self.base_tf)
        self._mtf.build(self.data)
        # Expose MTF engine and broker UTC offset to interpreter
        interpreter.variables['__mtf__'] = self._mtf
        interpreter.variables['__utc_offset__'] = self.utc_offset

        # Build timestamps for hour()/minute()/VWAP functions (needed by Cython interpreter)
        import pandas as _pd
        _timestamps = []
        _dt_col = None
        for _c in ('datetime', 'date', 'Date', 'Datetime'):
            if _c in self.data.columns:
                _dt_col = _c; break
        for _i in range(len(self.data)):
            _ts = 0
            if _dt_col:
                try:
                    _dt = _pd.Timestamp(self.data.iloc[_i][_dt_col])
                    _ts = int(_dt.timestamp())
                except Exception:
                    _ts = 1700000000 + _i * 300
            else:
                _ts = 1700000000 + _i * 300
            _timestamps.append(_ts)
        interpreter.variables['__timestamps__'] = _timestamps

        # Inject options chain for options-aware strategies (Phase 2 scripting access)
        if self.options_chain is not None and len(self.options_chain) > 0:
            interpreter.variables['__options_chain__'] = self.options_chain
            interpreter.variables['_options_available'] = True
            # Pre-index by date string for O(1) per-bar lookups
            try:
                _date_keys = pd.to_datetime(self.options_chain['date']).dt.strftime('%Y-%m-%d')
                interpreter.variables['__options_by_date__'] = {
                    k: g for k, g in self.options_chain.groupby(_date_keys)
                }
            except Exception:
                interpreter.variables['__options_by_date__'] = {}
        else:
            interpreter.variables['__options_chain__'] = None
            interpreter.variables['_options_available'] = False
            interpreter.variables['__options_by_date__'] = {}

        # Detect base/profit currencies from symbol name (e.g. NZDCHF → NZD, CHF)
        sym = self.symbol_name
        if len(sym) >= 6:
            interpreter.variables['__base_ccy__'] = sym[:3]
            interpreter.variables['__profit_ccy__'] = sym[3:6]
        else:
            interpreter.variables['__base_ccy__'] = 'USD'
            interpreter.variables['__profit_ccy__'] = 'USD'

        config = interpreter.strategy_config
        balance = self.initial_capital
        peak_equity = self.initial_capital

        # Auto-detect currency conversion and pip size
        avg_price = self.data['close'].iloc[:min(100, len(self.data))].mean()
        profit_ccy = self.symbol_name[3:6].upper() if len(self.symbol_name) >= 6 else ''
        if profit_ccy == 'JPY' or avg_price > 10:
            self._pnl_conversion = 1.0 / avg_price
            self._pip_size = 0.01
        elif profit_ccy == 'USD' or profit_ccy == '':
            self._pnl_conversion = 1.0
            self._pip_size = 0.0001
        else:
            # Cross pair: approximate conversion (P&L in foreign currency / price)
            self._pnl_conversion = 1.0 / avg_price
            self._pip_size = 0.0001

        # Platform-level SMT divergence engine
        self._smt_engine = SMTEngine() if self.smt_data is not None else None
        interpreter.variables['_smt_available'] = self.smt_data is not None
        for k in ('_smt_bull', '_smt_bear', '_smt_bull_active', '_smt_bear_active'):
            interpreter.variables[k] = False
        for k in ('_smt_bull_bar', '_smt_bear_bar'):
            interpreter.variables[k] = 0

        # Tracking
        bars_in_position = 0
        total_bars = len(self.data)

        date_col = None
        for col in ('datetime', 'date', 'time', 'Date', 'Time', 'Datetime'):
            if col in self.data.columns:
                date_col = col; break

        for i in range(total_bars):
            row = self.data.iloc[i]
            current_price = row['close']
            bar_high = row['high']
            bar_low = row['low']
            bar_open = row['open']
            current_date = row[date_col] if date_col else None

            # Track exposure
            if self.open_positions:
                bars_in_position += 1

            # ── Check pending SL/TP with intra-bar ordering (all positions) ──
            closed_tids = []
            for tid, pos in list(self.open_positions.items()):
                exit_order = self.pending_exits.get(tid)
                if not exit_order:
                    continue
                sl_hit = False; tp_hit = False
                sl_price = None; tp_price = None

                if pos.direction == 'long':
                    if exit_order.stop is not None and not is_na(exit_order.stop) and bar_low <= exit_order.stop:
                        sl_hit = True; sl_price = exit_order.stop
                    if exit_order.limit is not None and not is_na(exit_order.limit) and bar_high >= exit_order.limit:
                        tp_hit = True; tp_price = exit_order.limit
                else:
                    if exit_order.stop is not None and not is_na(exit_order.stop) and bar_high >= exit_order.stop:
                        sl_hit = True; sl_price = exit_order.stop
                    if exit_order.limit is not None and not is_na(exit_order.limit) and bar_low <= exit_order.limit:
                        tp_hit = True; tp_price = exit_order.limit

                if sl_hit and tp_hit:
                    bullish_bar = current_price >= bar_open
                    if pos.direction == 'long':
                        exit_price = sl_price if not bullish_bar else tp_price
                        exit_reason = 'sl' if not bullish_bar else 'tp'
                    else:
                        exit_price = sl_price if bullish_bar else tp_price
                        exit_reason = 'sl' if bullish_bar else 'tp'
                elif sl_hit:
                    exit_price = sl_price; exit_reason = 'sl'
                elif tp_hit:
                    exit_price = tp_price; exit_reason = 'tp'
                else:
                    continue

                balance = self._close_position(tid, i, exit_price, current_date, balance, is_na, reason=exit_reason)
                if tid in self.pending_exits:
                    del self.pending_exits[tid]
                closed_tids.append(tid)

            # ── MAE/MFE (all positions) ──
            for pos in self.open_positions.values():
                if pos.direction == 'long':
                    unrealized = (current_price - pos.entry_price) * pos.qty * self._pnl_conversion
                else:
                    unrealized = (pos.entry_price - current_price) * pos.qty * self._pnl_conversion
                pos.mfe = max(pos.mfe, unrealized)
                pos.mae = min(pos.mae, unrealized)

            # ── Update interpreter with position state ──
            # Aggregate across all open positions
            total_unrealized = 0.0
            total_qty_signed = 0.0
            for pos in self.open_positions.values():
                if pos.direction == 'long':
                    total_unrealized += (current_price - pos.entry_price) * pos.qty * self._pnl_conversion
                    total_qty_signed += pos.qty
                else:
                    total_unrealized += (pos.entry_price - current_price) * pos.qty * self._pnl_conversion
                    total_qty_signed -= pos.qty

            if self.engine == 'mql5':
                interpreter._position_type = -1
                interpreter._position_volume = 0.0
                interpreter._position_price_open = 0.0
                interpreter._position_profit = 0.0
                interpreter._account_balance = balance
                interpreter._account_equity = balance
                if self.open_positions:
                    # Use last-opened position for type/price (backwards compat)
                    last_pos = self.open_position
                    if last_pos:
                        qty = last_pos.qty
                        interpreter._position_type = 0 if last_pos.direction == 'long' else 1
                        interpreter._position_volume = qty / interpreter._contract_size
                        interpreter._position_price_open = last_pos.entry_price
                    interpreter._position_profit = total_unrealized
                    interpreter._account_equity = balance + total_unrealized
            else:
                interpreter.variables['_position_size'] = total_qty_signed
                interpreter.variables['_equity'] = balance + total_unrealized
                interpreter.variables['_open_profit'] = total_unrealized
                interpreter.variables['_entry_price'] = self.open_position.entry_price if self.open_position else NA

            # ── Update ALL strategy.* built-in variables ──
            # These must be set AFTER order fills and BEFORE script execution
            net_profit = balance - self.initial_capital
            win_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
            loss_trades = sum(1 for t in self.closed_trades if t.pnl <= 0)
            closed_count = len(self.closed_trades)

            interpreter.variables['_netprofit'] = net_profit
            interpreter.variables['_grossprofit'] = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
            interpreter.variables['_grossloss'] = sum(t.pnl for t in self.closed_trades if t.pnl < 0)
            interpreter.variables['_wintrades'] = win_trades
            interpreter.variables['_losstrades'] = loss_trades
            interpreter.variables['_closedtrades'] = closed_count
            interpreter.variables['_closedtrades_list'] = self.closed_trades  # for closedtrades.profit(idx)

            # ── Platform SMT divergence ──
            if self._smt_engine is not None and i < len(self.smt_data):
                smt_row = self.smt_data.iloc[i]
                self._smt_engine.update(i, bar_high, bar_low, smt_row['high'], smt_row['low'])
                interpreter.variables.update(self._smt_engine.get_variables(i))

            # ── Options: set current bar's date key for options.* lookups ──
            if self.options_chain is not None and i < len(_timestamps):
                _opt_ts = _timestamps[i]
                if _opt_ts and _opt_ts > 946684800:  # after 2000-01-01 — skip invalid/fallback timestamps
                    try:
                        import datetime as _dt
                        interpreter.variables['_options_date_key'] = _dt.datetime.utcfromtimestamp(_opt_ts).strftime('%Y-%m-%d')
                    except Exception:
                        interpreter.variables['_options_date_key'] = ''
                else:
                    interpreter.variables['_options_date_key'] = ''

            # ── Run interpreter ──
            interpreter.run_bar(i)

            # Get strategy config on first bar
            if interpreter.strategy_config and not hasattr(self, '_config_applied'):
                self._config_applied = True
                config = interpreter.strategy_config
                if 'initial_capital' in config and not is_na(config['initial_capital']):
                    self.initial_capital = float(config['initial_capital'])
                    balance = self.initial_capital
                if 'default_qty_value' in config and not is_na(config['default_qty_value']):
                    self.default_qty = float(config['default_qty_value'])
                if 'commission_value' in config and not is_na(config['commission_value']):
                    self.commission_pct = float(config['commission_value'])

            # ── Check pending entry orders ──
            for pe in self.pending_entries[:]:
                triggered = False
                if pe.order_type == 'limit':
                    if pe.direction == 'long' and bar_low <= pe.price:
                        triggered = True
                    elif pe.direction == 'short' and bar_high >= pe.price:
                        triggered = True
                elif pe.order_type == 'stop':
                    if pe.direction == 'long' and bar_high >= pe.price:
                        triggered = True
                    elif pe.direction == 'short' and bar_low <= pe.price:
                        triggered = True
                if triggered:
                    entry_price = self._apply_entry_price(pe.price, pe.direction)
                    new_trade = Trade(pe.direction, i, entry_price, pe.qty, pe.comment, current_date,
                                     sl_price=pe.sl, tp_price=pe.tp)
                    tid = self._add_position(new_trade)
                    if pe.sl or pe.tp:
                        self.pending_exits[tid] = PendingExit(stop=pe.sl, limit=pe.tp, comment=pe.comment)
                    self.pending_entries.remove(pe)
                    break

            # ── Process signals ──
            bar_signals = [s for idx, s in interpreter.all_signals if idx == i]
            interpreter.all_signals = [(idx, s) for idx, s in interpreter.all_signals if idx != i]

            for signal in bar_signals:
                if signal.action == 'entry':
                    order_type = getattr(signal, 'order_type', 'market')
                    entry_price_level = getattr(signal, 'entry_price', None)

                    if order_type in ('limit', 'stop') and entry_price_level:
                        # Queue as pending order — risk % takes priority when set
                        risk_qty = self._calc_risk_qty(entry_price_level, signal.stop, balance) if self.risk_pct > 0 else None
                        if risk_qty:
                            qty = risk_qty
                        elif signal.qty and not is_na(signal.qty) and signal.qty > 0:
                            qty = signal.qty
                        else:
                            qty = self.default_qty
                        if qty <= 0: qty = self.default_qty
                        self.pending_entries.append(PendingEntry(
                            signal.direction, qty,
                            entry_price_level, order_type,
                            sl=signal.stop, tp=signal.limit,
                            comment=signal.comment, created_bar=i))
                    else:
                        # Market order — close opposite positions first
                        for tid, pos in list(self.open_positions.items()):
                            if pos.direction != signal.direction:
                                balance = self._close_position(tid, i, current_price, current_date, balance, is_na, reason='opposite')
                                if tid in self.pending_exits:
                                    del self.pending_exits[tid]
                        if not self.open_positions:
                            entry_price = self._apply_entry_price(current_price, signal.direction)
                            # Risk-based sizing: find SL from this signal or from an exit signal on the same bar
                            sl_for_sizing = signal.stop if signal.stop and not is_na(signal.stop) else None
                            if not sl_for_sizing and self.risk_pct > 0:
                                # Look for a matching strategy.exit() signal on this bar that has a stop
                                for other_sig in bar_signals:
                                    if other_sig.action == 'exit' and other_sig.stop and not is_na(other_sig.stop):
                                        sl_for_sizing = other_sig.stop
                                        break
                            # Risk % takes priority when set; otherwise use signal qty or default
                            risk_qty = self._calc_risk_qty(entry_price, sl_for_sizing, balance) if self.risk_pct > 0 else None
                            if risk_qty:
                                qty = risk_qty
                            elif signal.qty and not is_na(signal.qty) and signal.qty > 0:
                                qty = signal.qty
                            else:
                                qty = self.default_qty
                            if qty <= 0: qty = self.default_qty
                            new_trade = Trade(signal.direction, i, entry_price, qty,
                                              signal.comment, current_date,
                                              sl_price=signal.stop, tp_price=signal.limit)
                            self._add_position(new_trade)

                elif signal.action == 'close':
                    for tid, pos in list(self.open_positions.items()):
                        if signal.direction == 'all' or pos.direction == signal.direction:
                            balance = self._close_position(tid, i, current_price, current_date, balance, is_na, reason='signal')
                            if tid in self.pending_exits:
                                del self.pending_exits[tid]

                elif signal.action == 'exit':
                    for tid in self.open_positions:
                        pos = self.open_positions[tid]
                        sl_val = signal.stop
                        tp_val = signal.limit

                        # Sanity check: ensure SL/TP are on the correct side of entry
                        if sl_val and tp_val and not is_na(sl_val) and not is_na(tp_val):
                            if pos.direction == 'long':
                                # Long: SL should be below entry, TP above
                                if sl_val > pos.entry_price and tp_val < pos.entry_price:
                                    sl_val, tp_val = tp_val, sl_val  # swap
                            else:
                                # Short: SL should be above entry, TP below
                                if sl_val < pos.entry_price and tp_val > pos.entry_price:
                                    sl_val, tp_val = tp_val, sl_val  # swap

                        pos.sl_price = sl_val
                        pos.tp_price = tp_val
                        self.pending_exits[tid] = PendingExit(
                            stop=sl_val, limit=tp_val,
                            from_entry=signal.from_entry, comment=signal.comment)

            # ── Process options signals ──
            for signal in bar_signals:
                if signal.action == 'entry_option':
                    balance = self._open_option(signal, i, current_date, balance, interpreter)
                elif signal.action == 'close_option':
                    balance = self._close_option_by_id(signal.comment, i, current_date, balance, interpreter)
                elif signal.action == 'close_all_options':
                    balance = self._close_all_options(i, current_date, balance, interpreter)
                elif signal.action in ('entry_spread', 'entry_condor', 'entry_straddle', 'entry_strangle'):
                    balance = self._open_spread(signal, i, current_date, balance, interpreter)

            # ── Options mark-to-market + expiry check ──
            import pandas as _opt_pd
            options_unrealized = 0.0
            for otid, opt in list(self.open_options.items()):
                # Check expiration
                if opt.expiry is not None and current_date:
                    try:
                        bar_ts = _opt_pd.Timestamp(current_date)
                        if bar_ts >= opt.expiry:
                            balance = self._expire_option(otid, i, current_date, current_price, balance)
                            continue
                    except Exception:
                        pass
                # Mark to market
                mark = self._get_option_mark(opt, interpreter)
                if mark is not None:
                    if opt.direction == 'long':
                        options_unrealized += (mark - opt.entry_price) * opt.qty * opt.multiplier
                    else:
                        options_unrealized += (opt.entry_price - mark) * opt.qty * opt.multiplier
            self._options_unrealized = options_unrealized

            # ── Update options portfolio variables ──
            interpreter.variables['_options_count'] = len(self.open_options)
            interpreter.variables['_options_profit'] = options_unrealized
            net_delta = 0.0; net_theta = 0.0
            for opt in self.open_options.values():
                sign = 1.0 if opt.direction == 'long' else -1.0
                net_delta += self._get_option_greek(opt, 'delta', interpreter) * opt.qty * sign * opt.multiplier
                net_theta += self._get_option_greek(opt, 'theta', interpreter) * opt.qty * sign * opt.multiplier
            interpreter.variables['_options_delta'] = round(net_delta, 2)
            interpreter.variables['_options_theta'] = round(net_theta, 2)

            # Equity — sum unrealized across all positions (equity + options)
            equity = balance
            for pos in self.open_positions.values():
                if pos.direction == 'long':
                    equity += (current_price - pos.entry_price) * pos.qty * self._pnl_conversion
                else:
                    equity += (pos.entry_price - current_price) * pos.qty * self._pnl_conversion
            equity += options_unrealized
            self.equity_curve.append(equity)
            self.balance_curve.append(balance)
            peak_equity = max(peak_equity, equity)
            dd = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0
            self.drawdown_curve.append(dd)

            # ── Streaming progress ──
            if streaming and (i % yield_every == 0 or i == total_bars - 1):
                yield {'status': 'progress', 'bar': i, 'total': total_bars, 'pct': round(i / total_bars * 100)}

        # Close remaining positions (equity + options)
        last_price = self.data.iloc[-1]['close'] if len(self.data) > 0 else 0
        last_date = self.data.iloc[-1][date_col] if date_col and len(self.data) > 0 else None
        if self.open_positions:
            for tid in list(self.open_positions.keys()):
                balance = self._close_position(tid, total_bars - 1, last_price, last_date, balance, is_na, reason='end')
        if self.open_options:
            for tid in list(self.open_options.keys()):
                balance = self._close_option_trade(tid, total_bars - 1, last_date, balance, interpreter, reason='end')
        if self.open_positions or self.open_options:
            if self.equity_curve:
                self.equity_curve[-1] = balance
                self.balance_curve[-1] = balance

        if not self.equity_curve:
            self.equity_curve = [self.initial_capital]
            self.balance_curve = [self.initial_capital]
            self.drawdown_curve = [0]

        self.trades = self.closed_trades
        self._exposure_pct = round(bars_in_position / total_bars * 100, 1) if total_bars > 0 else 0

        if streaming:
            yield {'status': 'done', 'results': self.get_results(interpreter)}
        else:
            return self.get_results(interpreter)

    def run(self):
        # _run_loop is a generator when streaming; consume it for non-streaming
        gen = self._run_loop(streaming=False)
        # When streaming=False, _run_loop returns via the else branch
        # But since it's a generator with yield statements, we need to exhaust it
        result = None
        try:
            while True:
                result = next(gen)
        except StopIteration as e:
            if e.value is not None:
                return e.value
        return result

    def run_streaming(self, yield_every=500):
        yield from self._run_loop(streaming=True, yield_every=yield_every)

    def _compute_options_viz(self, ohlc_dates):
        """Compute per-bar options viz data: IV overlay, PCR, OI, term structure, skew."""
        if self.options_chain is None or len(self.options_chain) == 0:
            return None
        import pandas as _pd
        try:
            _date_keys = _pd.to_datetime(self.options_chain['date']).dt.strftime('%Y-%m-%d')
            options_by_date = {k: g for k, g in self.options_chain.groupby(_date_keys)}
        except Exception:
            return None

        iv_data, pcr_data, call_oi_data, put_oi_data = [], [], [], []
        iv_front, iv_back, skew_data = [], [], []

        def _find_atm_iv(calls_df, bar_date, dte_target):
            """Find ATM IV for a given DTE target from a set of calls."""
            if len(calls_df) == 0:
                return None
            c = calls_df.copy()
            c['_dte'] = (_pd.to_datetime(c['expiration']) - bar_date).dt.days
            c['_dte_dist'] = (c['_dte'] - dte_target).abs()
            nearest = c['_dte_dist'].min()
            at_e = c[c['_dte_dist'] == nearest]
            valid = at_e[at_e['delta'].notna()] if 'delta' in at_e.columns else at_e
            if len(valid) == 0:
                return None
            valid = valid.copy()
            valid['_atm'] = (valid['delta'].fillna(0) - 0.5).abs()
            best = valid.nsmallest(1, '_atm')
            if len(best) == 0:
                return None
            v = best.iloc[0].get('implied_volatility')
            return round(float(v), 4) if _pd.notna(v) else None

        for date_str in ohlc_dates:
            lookup = date_str.replace('.', '-').split(' ')[0] if isinstance(date_str, str) else ''
            chain = options_by_date.get(lookup)
            if chain is None or len(chain) == 0:
                for lst in [iv_data, pcr_data, call_oi_data, put_oi_data, iv_front, iv_back, skew_data]:
                    lst.append(None)
                continue

            calls = chain[chain['type'] == 'call']
            puts = chain[chain['type'] == 'put']
            bar_date = _pd.Timestamp(lookup)

            # ATM IV (30-day)
            iv_data.append(_find_atm_iv(calls, bar_date, 30))

            # Put/Call Ratio
            c_oi = float(calls['open_interest'].sum()) if len(calls) > 0 else 0
            p_oi = float(puts['open_interest'].sum()) if len(puts) > 0 else 0
            pcr_data.append(round(p_oi / c_oi, 4) if c_oi > 0 else None)
            call_oi_data.append(c_oi)
            put_oi_data.append(p_oi)

            # IV Term Structure: 7d vs 60d
            iv_front.append(_find_atm_iv(calls, bar_date, 7))
            iv_back.append(_find_atm_iv(calls, bar_date, 60))

            # IV Skew: 25-delta put IV - 25-delta call IV
            skew_val = None
            if len(puts) > 0 and len(calls) > 0:
                all_opts = chain.copy()
                all_opts['_dte'] = (_pd.to_datetime(all_opts['expiration']) - bar_date).dt.days
                all_opts['_dte_dist'] = (all_opts['_dte'] - 30).abs()
                min_d = all_opts['_dte_dist'].min()
                at_e = all_opts[all_opts['_dte_dist'] == min_d]
                ep = at_e[at_e['type'] == 'put']
                ec = at_e[at_e['type'] == 'call']
                if len(ep) > 0 and len(ec) > 0:
                    ep = ep.copy(); ec = ec.copy()
                    ep['_d25'] = (ep['delta'].fillna(0).abs() - 0.25).abs()
                    ec['_d25'] = (ec['delta'].fillna(0) - 0.25).abs()
                    p25 = ep.nsmallest(1, '_d25')
                    c25 = ec.nsmallest(1, '_d25')
                    if len(p25) > 0 and len(c25) > 0:
                        piv = p25.iloc[0].get('implied_volatility')
                        civ = c25.iloc[0].get('implied_volatility')
                        if _pd.notna(piv) and _pd.notna(civ):
                            skew_val = round(float(piv) - float(civ), 4)
            skew_data.append(skew_val)

        return {
            'iv_overlay': iv_data, 'pcr': pcr_data,
            'call_oi': call_oi_data, 'put_oi': put_oi_data,
            'iv_front': iv_front, 'iv_back': iv_back,
            'iv_skew': skew_data,
        }

    def get_results(self, interpreter=None):
        _, is_na, _ = _get_create_interpreter(self.engine)

        # Post-backtest date exclusion filter: remove trades whose entry falls in excluded ranges
        filtered_trades = self.trades
        excluded_count = 0
        if self.date_filters:
            import pandas as _pd
            filtered = []
            for t in self.trades:
                excluded = False
                if t.entry_date:
                    try:
                        entry_dt = _pd.Timestamp(str(t.entry_date))
                        for filt in self.date_filters:
                            f_from = _pd.Timestamp(filt.get('from', ''))
                            f_to = _pd.Timestamp(filt.get('to', filt.get('from', '')))
                            f_to = f_to + _pd.Timedelta(days=1) - _pd.Timedelta(seconds=1)
                            if f_from <= entry_dt <= f_to:
                                excluded = True; break
                    except Exception:
                        pass
                if not excluded and t.exit_date:
                    try:
                        exit_dt = _pd.Timestamp(str(t.exit_date))
                        for filt in self.date_filters:
                            f_from = _pd.Timestamp(filt.get('from', ''))
                            f_to = _pd.Timestamp(filt.get('to', filt.get('from', '')))
                            f_to = f_to + _pd.Timedelta(days=1) - _pd.Timedelta(seconds=1)
                            if f_from <= exit_dt <= f_to:
                                excluded = True; break
                    except Exception:
                        pass
                if not excluded:
                    filtered.append(t)
            excluded_count = len(self.trades) - len(filtered)
            filtered_trades = filtered

        metrics = calculate_metrics(
            filtered_trades, self.equity_curve, self.drawdown_curve,
            self.initial_capital, self.data, self._exposure_pct)
        metrics['excluded_trades'] = excluded_count

        trade_list = [t.to_dict(self.engine) for t in filtered_trades]
        # Merge options trades into trade list
        for ot in self.closed_options:
            d = ot.to_dict()
            trade_list.append(d)

        equity = self.equity_curve
        drawdown = self.drawdown_curve
        date_col = None
        for col in ('datetime', 'date', 'time', 'Date', 'Time', 'Datetime'):
            if col in self.data.columns:
                date_col = col; break
        dates = [str(d) for d in self.data[date_col].tolist()] if date_col else list(range(len(equity)))

        max_points = 2000
        if len(equity) > max_points:
            step = len(equity) // max_points
            indices = list(range(0, len(equity), step))
            # Ensure the max drawdown point is included
            if drawdown:
                min_dd_idx = drawdown.index(min(drawdown))
                if min_dd_idx not in indices:
                    indices.append(min_dd_idx)
                    indices.sort()
            equity = [equity[i] for i in indices if i < len(equity)]
            drawdown = [drawdown[i] for i in indices if i < len(drawdown)]
            dates = [dates[i] for i in indices if i < len(dates)]

        ohlc = []
        ohlc_data = self.data[['open', 'high', 'low', 'close']].values.tolist()
        ohlc_dates = [str(d) for d in self.data[date_col].tolist()] if date_col else list(range(len(ohlc_data)))
        if len(ohlc_data) > max_points:
            step = len(ohlc_data) // max_points
            ohlc_data = ohlc_data[::step]; ohlc_dates = ohlc_dates[::step]
        for j, row in enumerate(ohlc_data):
            ohlc.append({
                'date': ohlc_dates[j] if j < len(ohlc_dates) else j,
                'open': round(row[0], 5), 'high': round(row[1], 5),
                'low': round(row[2], 5), 'close': round(row[3], 5),
            })

        smt_ohlc = []
        if self.smt_data is not None:
            smt_close = self.smt_data['close'].values.tolist()
            smt_dates = ohlc_dates
            if len(smt_close) > max_points:
                step = len(smt_close) // max_points
                smt_close = smt_close[::step]
            for j, val in enumerate(smt_close):
                smt_ohlc.append({
                    'date': smt_dates[j] if j < len(smt_dates) else j,
                    'close': round(float(val), 5),
                })

        markers = []
        for t in filtered_trades:
            markers.append({'type': 'entry', 'direction': t.direction, 'bar': t.entry_bar,
                           'price': t.entry_price, 'date': str(t.entry_date or t.entry_bar)})
            markers.append({'type': 'exit', 'direction': t.direction, 'bar': t.exit_bar,
                           'price': t.exit_price, 'date': str(t.exit_date or t.exit_bar)})

        # Monthly returns
        monthly = _monthly_returns(filtered_trades, self.initial_capital)

        # Trade heatmap: P&L by hour and day of week
        heatmap = _trade_heatmap(filtered_trades)

        # R-multiple distribution
        r_multiples = [t.r_multiple for t in filtered_trades if t.r_multiple != 0]

        # Underwater equity (consecutive bars in drawdown)
        underwater = _underwater_periods(self.drawdown_curve, dates if len(dates) == len(self.drawdown_curve) else None)

        # Funded account simulation
        funded = None
        if self.funded_rules and self.funded_rules.get('enabled'):
            if not filtered_trades:
                funded = {
                    'enabled': True, 'passed': False,
                    'target_pct': self.funded_rules.get('target', 10),
                    'net_pct': 0, 'passed_target': False,
                    'max_dd_pct': self.funded_rules.get('max_dd', 10),
                    'actual_dd': 0, 'passed_dd': True,
                    'daily_dd_pct': self.funded_rules.get('daily_dd', 5),
                    'min_days': self.funded_rules.get('min_days', 5),
                    'trading_days': 0, 'passed_min_days': False,
                    'target_hit_date': None, 'target_hit_trade': None,
                    'target_hit_days': None, 'duration_days': None,
                    'dd_breach_date': None, 'dd_breach_trade': None,
                    'daily_dd_breach': False, 'daily_dd_breach_date': None,
                    'first_trade_date': None, 'phase2': None,
                }
            else:
                target_pct = self.funded_rules.get('target', 10)
                max_dd_pct = self.funded_rules.get('max_dd', 10)
                daily_dd_pct = self.funded_rules.get('daily_dd', 5)
                min_days = self.funded_rules.get('min_days', 5)

                net_pct = metrics.get('net_profit_pct', 0)
                max_dd = abs(metrics.get('max_drawdown', 0))
                passed_target = net_pct >= target_pct
                passed_dd = max_dd <= max_dd_pct

                # Count unique trading days
                trade_dates = set()
                for t in filtered_trades:
                    if t.entry_date:
                        trade_dates.add(str(t.entry_date)[:10])
                trading_days = len(trade_dates)
                passed_min_days = trading_days >= min_days

                # Track WHEN the target was first hit
                target_hit_date = None
                target_hit_trade = None
                target_hit_days = None
                dd_breach_date = None
                dd_breach_trade = None

                running_pnl = 0
                target_amount = self.initial_capital * target_pct / 100
                dd_limit = self.initial_capital * max_dd_pct / 100
                running_peak = self.initial_capital
                days_seen = set()
                breached = False

                for i, t in enumerate(filtered_trades):
                    running_pnl += t.pnl
                    current_equity = self.initial_capital + running_pnl
                    running_peak = max(running_peak, current_equity)
                    current_dd = running_peak - current_equity

                    if t.entry_date:
                        days_seen.add(str(t.entry_date)[:10])

                    # Check if target hit for the first time
                    if target_hit_date is None and running_pnl >= target_amount:
                        target_hit_date = str(t.exit_date)[:10] if t.exit_date else str(t.entry_date)[:10] if t.entry_date else None
                        target_hit_trade = i + 1
                        target_hit_days = len(days_seen)

                    # Check if DD breached
                    if not breached and current_dd >= dd_limit:
                        breached = True
                        dd_breach_date = str(t.exit_date)[:10] if t.exit_date else str(t.entry_date)[:10] if t.entry_date else None
                        dd_breach_trade = i + 1

                # Daily drawdown check
                daily_dd_breach = False
                daily_dd_breach_date = None
                if daily_dd_pct > 0 and filtered_trades:
                    from collections import defaultdict
                    daily_pnl = defaultdict(float)
                    daily_start_equity = {}
                    running_eq = self.initial_capital
                    for t in filtered_trades:
                        day = str(t.entry_date)[:10] if t.entry_date else 'unknown'
                        if day not in daily_start_equity:
                            daily_start_equity[day] = running_eq
                        running_eq += t.pnl
                        # Check if intraday DD exceeds limit
                        day_start = daily_start_equity[day]
                        if day_start > 0:
                            intraday_dd = (day_start - running_eq) / day_start * 100
                            if intraday_dd >= daily_dd_pct:
                                daily_dd_breach = True
                                if not daily_dd_breach_date:
                                    daily_dd_breach_date = day

                passed = passed_target and passed_dd and passed_min_days
                # If DD was breached before target hit, it's a fail regardless
                if breached and (target_hit_trade is None or (dd_breach_trade and dd_breach_trade <= (target_hit_trade or 9999))):
                    passed = False
                # If daily DD was breached, fail
                if daily_dd_breach:
                    passed = False

                # Calculate duration from first trade to target hit
                first_date = str(filtered_trades[0].entry_date)[:10] if filtered_trades[0].entry_date else None
                duration_days = None
                if first_date and target_hit_date:
                    try:
                        import pandas as _pd
                        d1 = _pd.Timestamp(first_date)
                        d2 = _pd.Timestamp(target_hit_date)
                        duration_days = (d2 - d1).days
                    except Exception:
                        pass

                # Phase 2: if phase 1 passed and p2 target set
                p2_target = self.funded_rules.get('p2_target', 0)
                phase2 = None
                if passed and p2_target > 0 and target_hit_trade is not None:
                    # Phase 2 starts from the trade after phase 1 target was hit
                    p2_trades = filtered_trades[target_hit_trade:]
                    p2_pnl = sum(t.pnl for t in p2_trades)
                    p2_net_pct = (p2_pnl / self.initial_capital) * 100
                    p2_target_amount = self.initial_capital * p2_target / 100

                    # Track P2 metrics
                    p2_running_pnl = 0
                    eq_at_p2_start = self.initial_capital + sum(ft.pnl for ft in filtered_trades[:target_hit_trade])
                    p2_peak_val = eq_at_p2_start
                    p2_max_dd = 0
                    p2_target_hit_date = None
                    p2_target_hit_trade = None

                    for j, t in enumerate(p2_trades):
                        p2_running_pnl += t.pnl
                        p2_current = eq_at_p2_start + p2_running_pnl
                        p2_peak_val = max(p2_peak_val, p2_current)
                        p2_dd = (p2_peak_val - p2_current) / p2_peak_val * 100 if p2_peak_val > 0 else 0
                        p2_max_dd = max(p2_max_dd, p2_dd)

                        if p2_target_hit_date is None and p2_running_pnl >= p2_target_amount:
                            p2_target_hit_date = str(t.exit_date)[:10] if t.exit_date else None
                            p2_target_hit_trade = target_hit_trade + j + 1

                    p2_passed_target = p2_net_pct >= p2_target
                    p2_passed_dd = p2_max_dd <= max_dd_pct

                    phase2 = {
                        'passed': p2_passed_target and p2_passed_dd,
                        'target_pct': p2_target,
                        'net_pct': round(p2_net_pct, 2),
                        'passed_target': p2_passed_target,
                        'max_dd_pct': max_dd_pct,
                        'actual_dd': round(p2_max_dd, 2),
                        'passed_dd': p2_passed_dd,
                        'target_hit_date': p2_target_hit_date,
                        'target_hit_trade': p2_target_hit_trade,
                    }

                funded = {
                    'enabled': True,
                    'passed': passed,
                    'target_pct': target_pct,
                    'net_pct': round(net_pct, 2),
                    'passed_target': passed_target,
                    'max_dd_pct': max_dd_pct,
                    'actual_dd': round(max_dd, 2),
                    'passed_dd': passed_dd,
                    'daily_dd_pct': daily_dd_pct,
                    'min_days': min_days,
                    'trading_days': trading_days,
                    'passed_min_days': passed_min_days,
                    'target_hit_date': target_hit_date,
                    'target_hit_trade': target_hit_trade,
                    'target_hit_days': target_hit_days,
                    'duration_days': duration_days,
                    'dd_breach_date': dd_breach_date,
                    'dd_breach_trade': dd_breach_trade,
                    'daily_dd_breach': daily_dd_breach,
                    'daily_dd_breach_date': daily_dd_breach_date,
                    'first_trade_date': first_date,
                    'phase2': phase2,
                }

        return {
            'metrics': metrics, 'trades': trade_list,
            'equity_curve': {'dates': dates, 'values': [round(v, 2) for v in equity]},
            'drawdown_curve': {'dates': dates, 'values': [round(v, 2) for v in drawdown]},
            'ohlc': ohlc, 'smt_ohlc': smt_ohlc, 'markers': markers,
            'strategy_config': interpreter.strategy_config if interpreter else {},
            'total_bars': len(self.data),
            'monthly_returns': monthly,
            'r_multiples': r_multiples,
            'underwater': underwater,
            'heatmap': heatmap,
            'funded': funded,
            'options_viz': self._compute_options_viz(ohlc_dates) if self.options_chain is not None else None,
            'options_viz_dates': ohlc_dates if self.options_chain is not None else None,
        }


def calculate_metrics(trades, equity_curve, drawdown_curve, initial_capital, data, exposure_pct=0):
    if not trades:
        return _empty_metrics(initial_capital)
    total_trades = len(trades)
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl < 0]
    num_wins = len(winning); num_losses = len(losing)
    long_trades = [t for t in trades if t.direction == 'long']
    short_trades = [t for t in trades if t.direction == 'short']
    long_wins = [t for t in long_trades if t.pnl > 0]
    short_wins = [t for t in short_trades if t.pnl > 0]
    pnls = [t.pnl for t in trades]
    gross_profit = sum(t.pnl for t in winning)
    gross_loss = abs(sum(t.pnl for t in losing))
    net_profit = sum(pnls)
    net_profit_pct = (net_profit / initial_capital) * 100
    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
    avg_win = gross_profit / num_wins if num_wins > 0 else 0
    avg_loss = gross_loss / num_losses if num_losses > 0 else 0
    avg_trade = net_profit / total_trades
    largest_win = max(pnls) if pnls else 0
    largest_loss = min(pnls) if pnls else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.99 if gross_profit > 0 else 0)
    risk_reward = avg_win / avg_loss if avg_loss > 0 else (999.99 if avg_win > 0 else 0)
    max_consec_wins = _max_consecutive(trades, lambda t: t.pnl > 0)
    max_consec_losses = _max_consecutive(trades, lambda t: t.pnl < 0)
    # Single-pass max drawdown: both % and $ from the same iteration
    max_drawdown_pct = 0
    max_drawdown_abs = 0
    peak = equity_curve[0] if equity_curve else initial_capital
    for eq_val in equity_curve:
        peak = max(peak, eq_val)
        dd_abs = peak - eq_val
        dd_pct = (dd_abs / peak * 100) if peak > 0 else 0
        max_drawdown_abs = max(max_drawdown_abs, dd_abs)
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)
    max_drawdown = -max_drawdown_pct  # Negative convention

    avg_bars_held = sum(t.bars_held for t in trades) / total_trades

    # Resample equity to daily for realistic Sharpe/Sortino
    eq = np.array(equity_curve) if equity_curve else np.array([initial_capital])
    if len(eq) > 252:
        # Downsample to ~daily by taking every Nth point
        bars_per_day = max(1, len(eq) // max(1, len(equity_curve) // 252))
        daily_eq = eq[::bars_per_day]
    else:
        daily_eq = eq
    daily_returns = np.diff(daily_eq) / daily_eq[:-1] if len(daily_eq) > 1 else np.array([])
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    dr = daily_returns[daily_returns < 0]
    sortino = (np.mean(daily_returns) / np.std(dr)) * np.sqrt(252) if len(dr) > 0 and np.std(dr) > 0 else 0
    calmar = abs(net_profit_pct / max_drawdown) if max_drawdown != 0 else 0
    recovery = net_profit / max_drawdown_abs if max_drawdown_abs > 0 else 0
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    payoff = avg_win / avg_loss if avg_loss > 0 else 0
    final_equity = equity_curve[-1] if equity_curve else initial_capital

    # R-multiple stats
    r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
    avg_r = round(np.mean(r_multiples), 2) if r_multiples else 0
    expectancy_r = round(np.mean(r_multiples), 2) if r_multiples else 0

    # Max drawdown duration (bars)
    max_dd_duration = 0
    cur_dd_dur = 0
    for dd in drawdown_curve:
        if dd < 0:
            cur_dd_dur += 1
            max_dd_duration = max(max_dd_duration, cur_dd_dur)
        else:
            cur_dd_dur = 0

    return {
        'net_profit': round(net_profit, 2), 'net_profit_pct': round(net_profit_pct, 2),
        'gross_profit': round(gross_profit, 2), 'gross_loss': round(-gross_loss, 2),
        'initial_capital': round(initial_capital, 2), 'final_equity': round(final_equity, 2),
        'total_trades': total_trades, 'winning_trades': num_wins, 'losing_trades': num_losses,
        'breakeven_trades': len([t for t in trades if t.pnl == 0]),
        'long_trades': len(long_trades), 'short_trades': len(short_trades),
        'long_wins': len(long_wins), 'short_wins': len(short_wins),
        'long_win_rate': round((len(long_wins) / len(long_trades) * 100) if long_trades else 0, 2),
        'short_win_rate': round((len(short_wins) / len(short_trades) * 100) if short_trades else 0, 2),
        'win_rate': round(win_rate, 2),
        'avg_win': round(avg_win, 2), 'avg_loss': round(-avg_loss, 2), 'avg_trade': round(avg_trade, 2),
        'largest_win': round(largest_win, 2), 'largest_loss': round(largest_loss, 2),
        'profit_factor': round(min(profit_factor, 999.99), 2),
        'risk_reward': round(min(risk_reward, 999.99), 2),
        'payoff_ratio': round(payoff, 2), 'expectancy': round(expectancy, 2),
        'max_drawdown': round(max_drawdown, 2), 'max_drawdown_abs': round(max_drawdown_abs, 2),
        'max_dd_duration': max_dd_duration,
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(min(sortino, 999.99), 2),
        'calmar_ratio': round(calmar, 2), 'recovery_factor': round(recovery, 2),
        'max_consec_wins': max_consec_wins, 'max_consec_losses': max_consec_losses,
        'avg_bars_held': round(avg_bars_held, 1),
        'avg_mae': round(sum(t.mae for t in trades) / total_trades, 2),
        'avg_mfe': round(sum(t.mfe for t in trades) / total_trades, 2),
        'exposure_pct': exposure_pct,
        'avg_r_multiple': avg_r,
        'expectancy_r': expectancy_r,
        'pnl_distribution': _pnl_distribution(pnls),
        'win_loss_pcts': {
            'wins': [round(t.pnl_pct, 2) for t in winning],
            'losses': [round(t.pnl_pct, 2) for t in losing],
        },
    }


def _monthly_returns(trades, initial_capital):
    """Calculate monthly P&L table from trades."""
    if not trades:
        return []
    monthly = {}
    for t in trades:
        if t.exit_date:
            date_str = str(t.exit_date)
            # Extract YYYY.MM or YYYY-MM
            try:
                parts = date_str.replace('-', '.').split('.')
                if len(parts) >= 2:
                    key = f'{parts[0]}-{parts[1]}'
                else:
                    continue
            except Exception:
                continue
            if key not in monthly:
                monthly[key] = {'month': key, 'pnl': 0, 'trades': 0, 'wins': 0}
            monthly[key]['pnl'] = round(monthly[key]['pnl'] + t.pnl, 2)
            monthly[key]['trades'] += 1
            if t.pnl > 0:
                monthly[key]['wins'] += 1
    result = sorted(monthly.values(), key=lambda x: x['month'])
    # Add return % relative to initial capital
    for m in result:
        m['pnl_pct'] = round(m['pnl'] / initial_capital * 100, 2)
        m['win_rate'] = round(m['wins'] / m['trades'] * 100, 1) if m['trades'] > 0 else 0
    return result


def _underwater_periods(drawdown_curve, dates=None):
    """Find underwater (drawdown) periods with duration and dates."""
    periods = []
    in_dd = False
    start_idx = 0
    max_dd = 0
    for i, dd in enumerate(drawdown_curve):
        if dd < 0:
            if not in_dd:
                in_dd = True
                start_idx = i
                max_dd = dd
            else:
                max_dd = min(max_dd, dd)
        else:
            if in_dd:
                periods.append({
                    'start': start_idx,
                    'end': i,
                    'duration': i - start_idx,
                    'max_dd': round(max_dd, 2),
                    'start_date': dates[start_idx] if dates and start_idx < len(dates) else None,
                    'end_date': dates[i] if dates and i < len(dates) else None,
                })
                in_dd = False
    if in_dd:
        end_i = len(drawdown_curve) - 1
        periods.append({
            'start': start_idx,
            'end': end_i,
            'duration': len(drawdown_curve) - start_idx,
            'max_dd': round(max_dd, 2),
            'start_date': dates[start_idx] if dates and start_idx < len(dates) else None,
            'end_date': dates[end_i] if dates and end_i < len(dates) else None,
        })
    periods.sort(key=lambda x: x['duration'], reverse=True)
    return periods[:20]


def _trade_heatmap(trades):
    """Build P&L heatmap by hour (0-23) and day of week (0=Mon, 6=Sun)."""
    import pandas as _pd
    grid = {}  # {(day, hour): {'pnl': 0, 'count': 0, 'wins': 0}}
    for t in trades:
        if not t.entry_date:
            continue
        try:
            dt = _pd.Timestamp(str(t.entry_date))
            day = dt.dayofweek  # 0=Mon, 6=Sun
            hour = dt.hour
            key = f'{day}_{hour}'
            if key not in grid:
                grid[key] = {'day': day, 'hour': hour, 'pnl': 0, 'count': 0, 'wins': 0}
            grid[key]['pnl'] = round(grid[key]['pnl'] + t.pnl, 2)
            grid[key]['count'] += 1
            if t.pnl > 0:
                grid[key]['wins'] += 1
        except Exception:
            continue
    return list(grid.values())


def _empty_metrics(ic):
    return {k: 0 for k in [
        'net_profit','net_profit_pct','gross_profit','gross_loss','total_trades',
        'winning_trades','losing_trades','breakeven_trades','long_trades','short_trades',
        'long_wins','short_wins','long_win_rate','short_win_rate','win_rate',
        'avg_win','avg_loss','avg_trade','largest_win','largest_loss',
        'profit_factor','risk_reward','payoff_ratio','expectancy',
        'max_drawdown','max_drawdown_abs','max_dd_duration',
        'sharpe_ratio','sortino_ratio','calmar_ratio',
        'recovery_factor','max_consec_wins','max_consec_losses','avg_bars_held',
        'avg_mae','avg_mfe','exposure_pct','avg_r_multiple','expectancy_r',
    ]} | {'initial_capital': ic, 'final_equity': ic, 'pnl_distribution': [], 'win_loss_pcts': {'wins':[],'losses':[]}}

def _max_consecutive(trades, pred):
    mx = cur = 0
    for t in trades:
        if pred(t): cur += 1; mx = max(mx, cur)
        else: cur = 0
    return mx

def _pnl_distribution(pnls):
    if not pnls: return []
    mn, mx = min(pnls), max(pnls)
    if mn == mx: return [{'range': f'{mn:.0f}', 'count': len(pnls)}]
    nb = min(20, len(pnls))
    bw = (mx - mn) / nb
    return [{'range': f'{mn+i*bw:.0f} to {mn+(i+1)*bw:.0f}',
             'lo': round(mn+i*bw, 2), 'hi': round(mn+(i+1)*bw, 2),
             'count': sum(1 for p in pnls if mn+i*bw <= p < mn+(i+1)*bw or (i == nb-1 and p == mx))}
            for i in range(nb)]
