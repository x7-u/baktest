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
                 'exit_price', 'qty', 'pnl', 'pnl_pct', 'comment',
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
                 funded_rules: dict = None):
        self.data = data
        self.smt_data = smt_data
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
        # Per-trade flat fee
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

    def _close_position(self, tid, bar_idx, exit_price, current_date, balance, is_na):
        """Close a position by trade ID, return updated balance."""
        trade = self.open_positions.pop(tid)
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

    def _run_loop(self, streaming=False, yield_every=500):
        """Core backtest loop. If streaming=True, yields progress dicts."""
        create_interpreter, is_na, NA = _get_create_interpreter(self.engine)
        self.interpreter = interpreter = create_interpreter(self.source)
        interpreter.setup(self.data)
        if self.smt_data is not None:
            interpreter.setup_secondary(self.smt_data)

        # Build multi-timeframe data
        self._mtf = MTFEngine(base_tf=self.base_tf)
        self._mtf.build(self.data)
        # Expose MTF engine and broker UTC offset to interpreter
        interpreter.variables['__mtf__'] = self._mtf
        interpreter.variables['__utc_offset__'] = self.utc_offset

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
                    else:
                        exit_price = sl_price if bullish_bar else tp_price
                elif sl_hit:
                    exit_price = sl_price
                elif tp_hit:
                    exit_price = tp_price
                else:
                    continue

                balance = self._close_position(tid, i, exit_price, current_date, balance, is_na)
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

            # ── Platform SMT divergence ──
            if self._smt_engine is not None and i < len(self.smt_data):
                smt_row = self.smt_data.iloc[i]
                self._smt_engine.update(i, bar_high, bar_low, smt_row['high'], smt_row['low'])
                interpreter.variables.update(self._smt_engine.get_variables(i))

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
                        # Queue as pending order
                        risk_qty = self._calc_risk_qty(entry_price_level, signal.stop, balance)
                        qty = risk_qty if risk_qty else (signal.qty if signal.qty and not is_na(signal.qty) else self.default_qty)
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
                                balance = self._close_position(tid, i, current_price, current_date, balance, is_na)
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
                            risk_qty = self._calc_risk_qty(entry_price, sl_for_sizing, balance)
                            qty = risk_qty if risk_qty else (signal.qty if signal.qty and not is_na(signal.qty) else self.default_qty)
                            if qty <= 0: qty = self.default_qty
                            new_trade = Trade(signal.direction, i, entry_price, qty,
                                              signal.comment, current_date,
                                              sl_price=signal.stop, tp_price=signal.limit)
                            self._add_position(new_trade)

                elif signal.action == 'close':
                    for tid, pos in list(self.open_positions.items()):
                        if signal.direction == 'all' or pos.direction == signal.direction:
                            balance = self._close_position(tid, i, current_price, current_date, balance, is_na)
                            if tid in self.pending_exits:
                                del self.pending_exits[tid]

                elif signal.action == 'exit':
                    for tid in self.open_positions:
                        self.open_positions[tid].sl_price = signal.stop
                        self.open_positions[tid].tp_price = signal.limit
                        self.pending_exits[tid] = PendingExit(
                            stop=signal.stop, limit=signal.limit,
                            from_entry=signal.from_entry, comment=signal.comment)

            # Equity — sum unrealized across all positions
            equity = balance
            for pos in self.open_positions.values():
                if pos.direction == 'long':
                    equity += (current_price - pos.entry_price) * pos.qty * self._pnl_conversion
                else:
                    equity += (pos.entry_price - current_price) * pos.qty * self._pnl_conversion
            self.equity_curve.append(equity)
            self.balance_curve.append(balance)
            peak_equity = max(peak_equity, equity)
            dd = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0
            self.drawdown_curve.append(dd)

            # ── Streaming progress ──
            if streaming and (i % yield_every == 0 or i == total_bars - 1):
                yield {'status': 'progress', 'bar': i, 'total': total_bars, 'pct': round(i / total_bars * 100)}

        # Close remaining positions
        if self.open_positions:
            last_price = self.data.iloc[-1]['close']
            last_date = self.data.iloc[-1][date_col] if date_col else None
            for tid in list(self.open_positions.keys()):
                balance = self._close_position(tid, total_bars - 1, last_price, last_date, balance, is_na)
            self.equity_curve[-1] = balance
            self.balance_curve[-1] = balance

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
            equity = equity[::step]; drawdown = drawdown[::step]; dates = dates[::step]

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

        # R-multiple distribution
        r_multiples = [t.r_multiple for t in filtered_trades if t.r_multiple != 0]

        # Underwater equity (consecutive bars in drawdown)
        underwater = _underwater_periods(self.drawdown_curve, dates if len(dates) == len(self.drawdown_curve) else None)

        # Funded account simulation
        funded = None
        if self.funded_rules and self.funded_rules.get('enabled'):
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

            passed = passed_target and passed_dd and passed_min_days

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
            'funded': funded,
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
    """Find underwater (drawdown) periods with duration."""
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
                })
                in_dd = False
    if in_dd:
        periods.append({
            'start': start_idx,
            'end': len(drawdown_curve) - 1,
            'duration': len(drawdown_curve) - start_idx,
            'max_dd': round(max_dd, 2),
        })
    # Return top 10 longest
    periods.sort(key=lambda x: x['duration'], reverse=True)
    return periods[:10]


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
