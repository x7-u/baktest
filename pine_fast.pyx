# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-optimized Pine Script interpreter.
Drop-in replacement for PineInterpreter from pine_parser.py.
"""
cimport cython
from libc.math cimport sqrt, log, ceil, floor, pow as cpow, fabs, isnan, isinf, NAN, M_PI
from libc.stdlib cimport malloc, free, realloc
from cpython.ref cimport PyObject

import math

from pine_parser import (
    NumberLit, StringLit, BoolLit, NALit, Identifier, ArrayLit,
    BinOp, UnaryOp, HistoryRef, FuncCall, DotAccess, Ternary,
    Assignment, IfStmt, ForStmt, ExprStmt, ContinueStmt, BreakStmt,
    SwitchExpr, FuncDef, Program,
    PineNA, NA, Series, Signal,
    LoopBreak, LoopContinue,
)

# ─── Module-level constants ───────────────────────────────────────────────────

cdef object _NA = NA
cdef object _LoopBreak = LoopBreak
cdef object _LoopContinue = LoopContinue

# ─── Fast NA check ────────────────────────────────────────────────────────────

cdef inline bint _is_na(object v):
    if v is _NA or v is None:
        return True
    if isinstance(v, float):
        return isnan(<double>v)
    return False

cdef inline bint _truthy(object val):
    if val is _NA or val is None:
        return False
    if isinstance(val, bool):
        return <bint>val
    if isinstance(val, (int, float)):
        return val != 0
    return bool(val)

# ─── FastSeries ───────────────────────────────────────────────────────────────

cdef class FastSeries:
    cdef public list data
    cdef public int _start

    def __init__(self):
        self.data = []
        self._start = 0

    cpdef void append(self, object val):
        self.data.append(val)

    cpdef object get(self, int offset=0):
        cdef int idx = len(self.data) - 1 - offset
        if idx < 0:
            return _NA
        return self.data[idx]

    def __len__(self):
        return self._start + len(self.data)


# ─── FastPineInterpreter ─────────────────────────────────────────────────────

cdef class FastPineInterpreter:
    cdef public object ast
    cdef public dict variables
    cdef public set var_declared
    cdef public dict series_data
    cdef public int bar_index
    cdef public object bars
    cdef public list signals
    cdef public list all_signals
    cdef public dict strategy_config
    cdef public dict inputs
    cdef public dict plot_data
    cdef public dict user_functions
    cdef public bint _first_bar
    cdef public set _series_vars

    # C arrays for OHLCV fast access
    cdef double* _c_open
    cdef double* _c_high
    cdef double* _c_low
    cdef double* _c_close
    cdef double* _c_vol
    cdef int _bar_count

    # Row cache for Python fallback
    cdef public list _row_cache

    # SMT data
    cdef public list _smt_row_cache
    cdef public dict _smt_series
    cdef double* _c_smt_open
    cdef double* _c_smt_high
    cdef double* _c_smt_low
    cdef double* _c_smt_close
    cdef double* _c_smt_vol
    cdef int _smt_bar_count

    # Function name → int dispatch map
    cdef dict _func_ids

    def __init__(self, object ast):
        self.ast = ast
        self.variables = {}
        self.var_declared = set()
        self.series_data = {}
        self.bar_index = 0
        self.bars = None
        self.signals = []
        self.all_signals = []
        self.strategy_config = {}
        self.inputs = {}
        self.plot_data = {}
        self.user_functions = {}
        self._first_bar = True
        self._series_vars = set()
        self._row_cache = []
        self._smt_row_cache = None
        self._smt_series = {}
        self._c_open = NULL
        self._c_high = NULL
        self._c_low = NULL
        self._c_close = NULL
        self._c_vol = NULL
        self._bar_count = 0
        self._c_smt_open = NULL
        self._c_smt_high = NULL
        self._c_smt_low = NULL
        self._c_smt_close = NULL
        self._c_smt_vol = NULL
        self._smt_bar_count = 0
        self._func_ids = self._build_func_ids()

    def __dealloc__(self):
        if self._c_open != NULL: free(self._c_open)
        if self._c_high != NULL: free(self._c_high)
        if self._c_low != NULL: free(self._c_low)
        if self._c_close != NULL: free(self._c_close)
        if self._c_vol != NULL: free(self._c_vol)
        if self._c_smt_open != NULL: free(self._c_smt_open)
        if self._c_smt_high != NULL: free(self._c_smt_high)
        if self._c_smt_low != NULL: free(self._c_smt_low)
        if self._c_smt_close != NULL: free(self._c_smt_close)
        if self._c_smt_vol != NULL: free(self._c_smt_vol)

    cdef dict _build_func_ids(self):
        return {
            'strategy': 1, 'strategy.entry': 2, 'strategy.close': 3,
            'strategy.exit': 4, 'strategy.close_all': 5,
            'strategy.opentrades.entry_price': 6,
            'request.security': 10,
            'ta.pivothigh': 20, 'pivothigh': 20,
            'ta.pivotlow': 21, 'pivotlow': 21,
            'ta.sma': 22, 'sma': 22,
            'ta.ema': 23, 'ema': 23,
            'ta.rsi': 24, 'rsi': 24,
            'ta.atr': 25, 'atr': 25,
            'ta.wma': 26, 'wma': 26,
            'ta.macd': 27,
            'ta.crossover': 28, 'crossover': 28,
            'ta.crossunder': 29, 'crossunder': 29,
            'ta.highest': 30, 'highest': 30,
            'ta.lowest': 31, 'lowest': 31,
            'ta.change': 32, 'change': 32,
            'ta.tr': 33,
            'ta.stoch': 34,
            'ta.bb': 35,
            'ta.cum': 36,
            'math.abs': 40, 'abs': 40,
            'math.max': 41, 'max': 41,
            'math.min': 42, 'min': 42,
            'math.sqrt': 43, 'math.pow': 44, 'math.log': 45,
            'math.round': 46, 'round': 46,
            'math.ceil': 47, 'math.floor': 48, 'math.sign': 49,
            'nz': 60, 'na': 61, 'fixnan': 62, 'int': 63, 'float': 64,
            'str.tostring': 70, 'str.contains': 71, 'str.substring': 72,
            'str.pos': 73, 'str.length': 74, 'str.trim': 75,
            'array.new_float': 80, 'array.new_int': 80, 'array.new_bool': 80,
            'array.new_string': 80, 'array.new': 80,
            'array.push': 81, 'array.size': 82, 'array.get': 83,
            'array.set': 84, 'array.shift': 85, 'array.pop': 86,
            'array.remove': 87, 'array.clear': 88,
            'array.min': 89, 'array.max': 90,
        }

    # ─── AST pre-scan ─────────────────────────────────────────────────────

    cdef set _collect_history_vars(self, list stmts):
        cdef set names = set()
        self._walk_for_history(stmts, names)
        names.update(['open', 'high', 'low', 'close', 'volume', 'hl2', 'hlc3', 'ohlc4'])
        return names

    cdef void _walk_for_history(self, list stmts, set names):
        cdef object node
        for node in stmts:
            self._walk_node(node, names)

    cdef void _walk_node(self, object node, set names):
        if node is None:
            return
        cdef int tag = node._tag
        if tag == 8:  # HistoryRef
            n = self._node_name(node.expr)
            if n is not None:
                names.add(n)
            self._walk_node(node.offset, names)
        elif tag == 6:  # BinOp
            self._walk_node(node.left, names)
            self._walk_node(node.right, names)
        elif tag == 7:  # UnaryOp
            self._walk_node(node.operand, names)
        elif tag == 11:  # Ternary
            self._walk_node(node.cond, names)
            self._walk_node(node.true_val, names)
            self._walk_node(node.false_val, names)
        elif tag == 9:  # FuncCall
            for a in node.args:
                self._walk_node(a, names)
            for v in node.kwargs.values():
                self._walk_node(v, names)
        elif tag == 12:  # Assignment
            self._walk_node(node.value, names)
        elif tag == 13:  # IfStmt
            self._walk_node(node.cond, names)
            self._walk_for_history(node.body, names)
            if node.else_body:
                self._walk_for_history(node.else_body, names)
        elif tag == 14:  # ForStmt
            self._walk_node(node.start, names)
            self._walk_node(node.end, names)
            if node.step:
                self._walk_node(node.step, names)
            self._walk_for_history(node.body, names)
        elif tag == 15:  # ExprStmt
            self._walk_node(node.expr, names)
        elif tag == 18:  # SwitchExpr
            if node.expr:
                self._walk_node(node.expr, names)
            for cond, body in node.cases:
                self._walk_node(cond, names)
                self._walk_for_history(body, names)
            if node.default:
                self._walk_for_history(node.default, names)
        elif tag == 19:  # FuncDef
            self._walk_for_history(node.body, names)

    # ─── Setup ────────────────────────────────────────────────────────────

    cpdef void setup(self, object bars):
        self.bars = bars
        self.bar_index = 0
        self._first_bar = True
        cdef int n = len(bars)
        self._bar_count = n

        for col in ('open', 'high', 'low', 'close', 'volume'):
            self.series_data[col] = FastSeries()

        self._series_vars = self._collect_history_vars(self.ast.stmts)

        # Allocate C arrays
        if self._c_open != NULL: free(self._c_open)
        self._c_open = <double*>malloc(n * sizeof(double))
        self._c_high = <double*>malloc(n * sizeof(double))
        self._c_low = <double*>malloc(n * sizeof(double))
        self._c_close = <double*>malloc(n * sizeof(double))
        self._c_vol = <double*>malloc(n * sizeof(double))

        cdef int i
        self._row_cache = []
        for i in range(n):
            r = bars.iloc[i]
            o = r.get('open', 0)
            h = r.get('high', 0)
            l = r.get('low', 0)
            c = r.get('close', 0)
            v = r.get('volume', 0)
            self._c_open[i] = <double>o
            self._c_high[i] = <double>h
            self._c_low[i] = <double>l
            self._c_close[i] = <double>c
            self._c_vol[i] = <double>v
            self._row_cache.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

        # Auto-detect instrument type from price levels
        cdef double avg_price = 0.0
        cdef int sample = min(100, n)
        for i in range(sample):
            avg_price += self._c_close[i]
        avg_price /= sample if sample > 0 else 1
        if avg_price > 10:  # JPY pair
            self.variables['_syminfo_currency'] = 'JPY'
            self.variables['_syminfo_pointvalue'] = 0.01 / avg_price
        else:
            self.variables['_syminfo_currency'] = 'USD'
            self.variables['_syminfo_pointvalue'] = 1.0

    cpdef void setup_secondary(self, object bars):
        self._smt_series = {}
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self._smt_series[col] = FastSeries()
        cdef int n = len(bars)
        self._smt_bar_count = n

        if self._c_smt_open != NULL: free(self._c_smt_open)
        self._c_smt_open = <double*>malloc(n * sizeof(double))
        self._c_smt_high = <double*>malloc(n * sizeof(double))
        self._c_smt_low = <double*>malloc(n * sizeof(double))
        self._c_smt_close = <double*>malloc(n * sizeof(double))
        self._c_smt_vol = <double*>malloc(n * sizeof(double))

        self._smt_row_cache = []
        cdef int i
        for i in range(n):
            r = bars.iloc[i]
            o = r.get('open', 0); h = r.get('high', 0)
            l = r.get('low', 0); c = r.get('close', 0)
            v = r.get('volume', 0)
            self._c_smt_open[i] = <double>o
            self._c_smt_high[i] = <double>h
            self._c_smt_low[i] = <double>l
            self._c_smt_close[i] = <double>c
            self._c_smt_vol[i] = <double>v
            self._smt_row_cache.append({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

    # ─── Run bar ──────────────────────────────────────────────────────────

    cpdef void run_bar(self, int idx):
        self.bar_index = idx
        self.signals = []

        cdef double o = self._c_open[idx]
        cdef double h = self._c_high[idx]
        cdef double l = self._c_low[idx]
        cdef double c = self._c_close[idx]
        cdef double vol = self._c_vol[idx]

        (<FastSeries>self.series_data['open']).append(o)
        (<FastSeries>self.series_data['high']).append(h)
        (<FastSeries>self.series_data['low']).append(l)
        (<FastSeries>self.series_data['close']).append(c)
        (<FastSeries>self.series_data['volume']).append(vol)

        # Feed SMT data
        if self._smt_row_cache is not None and idx < self._smt_bar_count:
            (<FastSeries>self._smt_series['open']).append(self._c_smt_open[idx])
            (<FastSeries>self._smt_series['high']).append(self._c_smt_high[idx])
            (<FastSeries>self._smt_series['low']).append(self._c_smt_low[idx])
            (<FastSeries>self._smt_series['close']).append(self._c_smt_close[idx])
            (<FastSeries>self._smt_series['volume']).append(self._c_smt_vol[idx])

        cdef dict v = self.variables
        v['bar_index'] = idx
        v['close'] = c; v['open'] = o; v['high'] = h; v['low'] = l
        v['volume'] = vol
        v['hl2'] = (h + l) * 0.5
        v['hlc3'] = (h + l + c) / 3.0
        v['ohlc4'] = (o + h + l + c) * 0.25

        cdef list stmts = self.ast.stmts
        cdef int i
        cdef int ns = len(stmts)
        for i in range(ns):
            self._exec(stmts[i])

        if self.signals:
            self.all_signals.extend([(idx, s) for s in self.signals])
        self._first_bar = False

    cpdef list run_all_bars(self):
        self.all_signals = []
        cdef int i
        for i in range(self._bar_count):
            self.run_bar(i)
        return self.all_signals

    # ─── _exec ────────────────────────────────────────────────────────────

    cdef void _exec(self, object node):
        cdef int tag = node._tag
        cdef object val

        if tag == 12:  # Assignment
            self._exec_assign(node)
        elif tag == 13:  # IfStmt
            if _truthy(self._eval(node.cond)):
                for s in node.body: self._exec(s)
            elif node.else_body:
                for s in node.else_body: self._exec(s)
        elif tag == 14:  # ForStmt
            self._exec_for(node)
        elif tag == 16:  # ContinueStmt
            raise _LoopContinue
        elif tag == 17:  # BreakStmt
            raise _LoopBreak
        elif tag == 15:  # ExprStmt
            self._eval(node.expr)
        elif tag == 19:  # FuncDef
            self.user_functions[node.name] = node

    cdef void _exec_assign(self, object node):
        cdef object val = self._eval(node.value)
        cdef str name = node.name

        if node.is_var:
            if name not in self.var_declared:
                self.var_declared.add(name)
                self.variables[name] = val
            return

        cdef str op = node.op
        if op == ':=' or op == '=':
            self.variables[name] = val
        elif op == '+=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '+')
        elif op == '-=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '-')
        elif op == '*=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '*')
        elif op == '/=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '/')

        # Update series for history-referenced vars
        cdef object s
        if name in self.series_data:
            s = self.series_data[name]
            if len(s) <= self.bar_index:
                s.append(self.variables[name])
            else:
                rel = self.bar_index - s._start
                if 0 <= rel < len(s.data):
                    s.data[rel] = self.variables[name]
        elif name in self._series_vars:
            s = FastSeries()
            s._start = self.bar_index
            s.append(self.variables.get(name, _NA))
            self.series_data[name] = s

    cdef void _exec_for(self, object node):
        cdef int start = <int>self._eval(node.start)
        cdef int end = <int>self._eval(node.end)
        cdef int step
        if node.step:
            step = <int>self._eval(node.step)
        else:
            step = 1 if end >= start else -1
        if step == 0:
            step = 1

        cdef int val
        cdef str var = node.var
        cdef list body = node.body

        if step > 0:
            val = start
            while val <= end:
                self.variables[var] = val
                try:
                    for s in body: self._exec(s)
                except LoopContinue:
                    pass
                except LoopBreak:
                    break
                val += step
        else:
            val = start
            while val >= end:
                self.variables[var] = val
                try:
                    for s in body: self._exec(s)
                except LoopContinue:
                    pass
                except LoopBreak:
                    break
                val += step

    # ─── _eval ────────────────────────────────────────────────────────────

    cdef object _eval(self, object node):
        cdef int tag = node._tag
        cdef object val

        if tag == 0:  # NumberLit
            return node.value
        elif tag == 1:  # StringLit
            return node.value
        elif tag == 2:  # BoolLit
            return node.value
        elif tag == 3:  # NALit
            return _NA
        elif tag == 4:  # Identifier
            val = self.variables.get(node.name)
            if val is not None:
                return val
            if node.name in self.series_data:
                return (<FastSeries>self.series_data[node.name]).get(0)
            if node.name == 'strategy':
                return '<strategy>'
            return _NA
        elif tag == 5:  # ArrayLit
            return [self._eval(e) for e in node.elements]
        elif tag == 6:  # BinOp
            return self._eval_binop(node)
        elif tag == 7:  # UnaryOp
            val = self._eval(node.operand)
            if node.op == '-':
                return -val if not _is_na(val) else _NA
            if node.op == 'not':
                return not _truthy(val)
            return val
        elif tag == 11:  # Ternary
            return self._eval(node.true_val) if _truthy(self._eval(node.cond)) else self._eval(node.false_val)
        elif tag == 8:  # HistoryRef
            return self._eval_history(node)
        elif tag == 18:  # SwitchExpr
            return self._eval_switch(node)
        elif tag == 13:  # IfStmt (as expression)
            return self._eval_if_expr(node)
        elif tag == 9:  # FuncCall
            return self._call(node)
        elif tag == 10:  # DotAccess
            full = self._dot_name(node)
            if full is not None:
                return self._resolve_constant(full)
            return _NA
        return _NA

    cdef object _eval_history(self, object node):
        cdef int offset = <int>self._eval(node.offset)
        cdef str name = self._node_name(node.expr)
        if name is not None and name in self.series_data:
            return (<FastSeries>self.series_data[name]).get(offset)
        return _NA

    cdef object _eval_binop(self, object node):
        cdef object left = self._eval(node.left)
        cdef object right = self._eval(node.right)
        cdef int op_id = node.op_id

        if op_id == 11:  # and
            return _truthy(left) and _truthy(right)
        if op_id == 12:  # or
            return _truthy(left) or _truthy(right)
        if op_id == 9 or op_id == 10:  # == or !=
            if _is_na(left) and _is_na(right):
                return op_id == 9
            if _is_na(left) or _is_na(right):
                return op_id == 10
            return (left == right) if op_id == 9 else (left != right)
        if _is_na(left) or _is_na(right):
            return _NA

        # Fast double path
        cdef double ld, rd
        try:
            ld = <double>float(left)
            rd = <double>float(right)
        except (TypeError, ValueError):
            return _NA

        if op_id == 0: return ld + rd
        elif op_id == 1: return ld - rd
        elif op_id == 2: return ld * rd
        elif op_id == 3: return ld / rd if rd != 0.0 else _NA
        elif op_id == 4: return ld % rd if rd != 0.0 else _NA
        elif op_id == 5: return ld < rd
        elif op_id == 6: return ld > rd
        elif op_id == 7: return ld <= rd
        elif op_id == 8: return ld >= rd
        return _NA

    cdef object _eval_switch(self, object node):
        cdef object val, cv, result
        if node.expr is not None:
            val = self._eval(node.expr)
            for cond, body in node.cases:
                cv = self._eval(cond)
                if val == cv or (_is_na(val) and _is_na(cv)):
                    result = _NA
                    for stmt in body:
                        if stmt._tag == 15: result = self._eval(stmt.expr)
                        else: self._exec(stmt)
                    return result
        else:
            for cond, body in node.cases:
                if _truthy(self._eval(cond)):
                    result = _NA
                    for stmt in body:
                        if stmt._tag == 15: result = self._eval(stmt.expr)
                        else: self._exec(stmt)
                    return result
        if node.default:
            result = _NA
            for stmt in node.default:
                if stmt._tag == 15: result = self._eval(stmt.expr)
                else: self._exec(stmt)
            return result
        return _NA

    cdef object _eval_if_expr(self, object node):
        cdef object result
        if _truthy(self._eval(node.cond)):
            result = _NA
            for stmt in node.body:
                if stmt._tag == 15: result = self._eval(stmt.expr)
                else: self._exec(stmt)
            return result
        elif node.else_body:
            result = _NA
            for stmt in node.else_body:
                if stmt._tag == 15: result = self._eval(stmt.expr)
                else: self._exec(stmt)
            return result
        return _NA

    cdef object _arith(self, object left, object right, str op):
        if _is_na(left) or _is_na(right):
            return _NA
        cdef double ld, rd
        try:
            ld = <double>float(left)
            rd = <double>float(right)
        except (TypeError, ValueError):
            return _NA
        if op == '+': return ld + rd
        if op == '-': return ld - rd
        if op == '*': return ld * rd
        if op == '/': return ld / rd if rd != 0.0 else _NA
        if op == '%': return ld % rd if rd != 0.0 else _NA
        if op == '<': return ld < rd
        if op == '>': return ld > rd
        if op == '<=': return ld <= rd
        if op == '>=': return ld >= rd
        return _NA

    cdef str _dot_name(self, object node):
        if node._tag == 10:  # DotAccess
            p = self._dot_name(node.obj)
            if p is not None:
                return p + '.' + node.attr
            return None
        if node._tag == 4:  # Identifier
            return node.name
        return None

    cdef str _node_name(self, object node):
        if node._tag == 4: return node.name
        if node._tag == 10: return self._dot_name(node)
        return None

    @cython.wraparound(True)
    cdef object _resolve_constant(self, str name):
        if name == 'strategy.long': return 'long'
        if name == 'strategy.short': return 'short'
        if name == 'strategy.fixed': return 'fixed'
        if name == 'strategy.cash': return 'cash'
        if name == 'strategy.percent_of_equity': return 'percent_of_equity'
        if name == 'strategy.commission.percent': return 'percent'
        if name == 'strategy.commission.cash_per_contract': return 'cash_per_contract'
        if name == 'strategy.position_size': return self.variables.get('_position_size', 0)
        if name == 'strategy.equity': return self.variables.get('_equity', 10000)
        if name == 'strategy.openprofit': return self.variables.get('_open_profit', 0)
        if name == 'math.pi': return M_PI
        if name == 'math.e': return 2.718281828459045
        if name == 'barstate.islast':
            return self.bar_index == self._bar_count - 1 if self.bars is not None else False
        if name == 'barstate.isfirst': return self.bar_index == 0
        if name == 'barstate.isconfirmed': return True
        if name == 'syminfo.tickerid' or name == 'syminfo.ticker': return 'SYMBOL'
        if name == 'syminfo.currency': return self.variables.get('_syminfo_currency', 'USD')
        if name == 'syminfo.pointvalue': return self.variables.get('_syminfo_pointvalue', 1.0)
        if name == 'timeframe.period': return '60'
        if name == 'barmerge.lookahead_on': return True
        if name == 'barmerge.lookahead_off': return False
        if name.startswith('currency.'): return name.split('.')[-1]
        if name.startswith('label.style_'): return name.split('style_')[-1]
        if name.startswith('line.style_'): return name.split('style_')[-1]
        if name.startswith('plot.style_'): return name.split('style_')[-1]
        if name.startswith('size.'): return name.split('.')[-1]
        if name.startswith('position.'): return name.split('position.')[-1]
        if name == 'format.mintick': return '#.#####'
        if name == 'order.ascending': return 'asc'
        if name == 'order.descending': return 'desc'
        if name == 'alert.freq_once_per_bar_close': return 'once_per_bar_close'
        if name.startswith('color.'): return name
        return _NA

    # ─── _call (built-in function dispatch) ───────────────────────────────

    cdef object _call(self, object node):
        cdef str name = node.name
        cdef list args = [self._eval(a) for a in node.args]
        cdef dict kwargs = {k: self._eval(v) for k, v in node.kwargs.items()}
        cdef int fid = self._func_ids.get(name, -1)

        # Strategy
        if fid == 1: return self._fn_strategy(args, kwargs)
        if fid == 2: return self._fn_strategy_entry(args, kwargs)
        if fid == 3: return self._fn_strategy_close(args, kwargs)
        if fid == 4: return self._fn_strategy_exit(args, kwargs)
        if fid == 5:
            self.signals.append(Signal('close', 'all'))
            return None
        if fid == 6: return self.variables.get('_entry_price', _NA)

        # Input
        if name.startswith('input'):
            return self._fn_input(args, kwargs)

        # request.security
        if fid == 10: return self._fn_request_security(args, kwargs)

        # Technical indicators
        if fid == 20: return self._ta_pivothigh(args, kwargs)
        if fid == 21: return self._ta_pivotlow(args, kwargs)
        if fid == 22: return self._ta_sma(args, kwargs)
        if fid == 23: return self._ta_ema(args, kwargs)
        if fid == 24: return self._ta_rsi(args, kwargs)
        if fid == 25: return self._ta_atr(args, kwargs)
        if fid == 26: return self._ta_wma(args, kwargs)
        if fid == 27: return self._ta_macd(args, kwargs)
        if fid == 28: return self._ta_crossover(args)
        if fid == 29: return self._ta_crossunder(args)
        if fid == 30: return self._ta_highest(args, kwargs)
        if fid == 31: return self._ta_lowest(args, kwargs)
        if fid == 32: return self._ta_change(args, kwargs)
        if fid == 33:
            h = self.variables.get('high', 0); l = self.variables.get('low', 0)
            c_prev = (<FastSeries>self.series_data['close']).get(1)
            if _is_na(c_prev): return <double>h - <double>l
            return max(<double>h - <double>l, fabs(<double>h - <double>c_prev), fabs(<double>l - <double>c_prev))
        if fid == 34: return self._ta_stoch(args, kwargs)
        if fid == 35: return self._ta_bb(args, kwargs)
        if fid == 36:
            source = args[0] if args else 0
            if _is_na(source): return self.variables.get('_cum_total', 0)
            total = self.variables.get('_cum_total', 0) + float(source)
            self.variables['_cum_total'] = total
            return total

        # ── Extended TA functions — delegate to Python module ──
        if name in ('ta.valuewhen', 'valuewhen', 'ta.barssince', 'barssince',
                     'ta.highestbars', 'highestbars', 'ta.lowestbars', 'lowestbars',
                     'ta.rising', 'ta.falling', 'ta.mom', 'mom',
                     'ta.vwma', 'vwma', 'ta.percentrank', 'ta.swma'):
            from pine_parser import PineInterpreter as _PI
            # Create a temporary Python interpreter method call
            pi = _PI.__new__(_PI)
            pi.variables = self.variables
            pi.series_data = self.series_data
            pi.bar_index = self.bar_index
            pi._bar_cache = {}
            return pi._call(node)

        # Math
        if fid == 40: return fabs(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if fid == 41:
            vals = [v for v in args if not _is_na(v)]
            return max(vals) if vals else _NA
        if fid == 42:
            vals = [v for v in args if not _is_na(v)]
            return min(vals) if vals else _NA
        if fid == 43:
            return sqrt(<double>args[0]) if args and not _is_na(args[0]) and float(args[0]) >= 0 else _NA
        if fid == 44:
            return cpow(<double>args[0], <double>args[1]) if len(args) >= 2 and not _is_na(args[0]) and not _is_na(args[1]) else _NA
        if fid == 45:
            return log(<double>args[0]) if args and not _is_na(args[0]) and float(args[0]) > 0 else _NA
        if fid == 46:
            if args and not _is_na(args[0]):
                return round(float(args[0]), int(args[1]) if len(args) > 1 else 0)
            return _NA
        if fid == 47: return ceil(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if fid == 48: return floor(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if fid == 49:
            if args and not _is_na(args[0]):
                return 1 if float(args[0]) > 0 else (-1 if float(args[0]) < 0 else 0)
            return _NA

        # Utility
        if fid == 60:
            val = args[0] if args else _NA
            replacement = args[1] if len(args) > 1 else 0
            return replacement if _is_na(val) else val
        if fid == 61: return _is_na(args[0]) if args else _NA
        if fid == 62: return args[0] if args and not _is_na(args[0]) else 0
        if fid == 63: return int(args[0]) if args and not _is_na(args[0]) else _NA
        if fid == 64: return float(args[0]) if args and not _is_na(args[0]) else _NA

        # String
        if fid == 70:
            if len(args) >= 1: return str(args[0]) if not _is_na(args[0]) else 'NaN'
            return ''
        if fid == 71: return args[1] in args[0] if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str) else False
        if fid == 72:
            if len(args) >= 3 and isinstance(args[0], str): return args[0][int(args[1]):int(args[2])]
            return ''
        if fid == 73:
            if len(args) >= 2 and isinstance(args[0], str): return args[0].find(args[1])
            return -1
        if fid == 74: return len(args[0]) if args and isinstance(args[0], str) else 0
        if fid == 75: return args[0].strip() if args and isinstance(args[0], str) else ''

        # Array
        if fid == 80:
            size = int(args[0]) if args and not _is_na(args[0]) else 0
            fill = args[1] if len(args) > 1 else (0 if 'int' in name or 'float' in name else (False if 'bool' in name else _NA))
            return [fill] * size
        if fid == 81:
            if args and isinstance(args[0], list): args[0].append(args[1] if len(args) > 1 else _NA)
            return None
        if fid == 82: return len(args[0]) if args and isinstance(args[0], list) else 0
        if fid == 83:
            if args and isinstance(args[0], list) and len(args) > 1:
                idx = int(args[1])
                return args[0][idx] if 0 <= idx < len(args[0]) else _NA
            return _NA
        if fid == 84:
            if args and isinstance(args[0], list) and len(args) > 2:
                idx = int(args[1])
                if 0 <= idx < len(args[0]): args[0][idx] = args[2]
            return None
        if fid == 85:
            if args and isinstance(args[0], list) and len(args[0]) > 0: return args[0].pop(0)
            return _NA
        if fid == 86:
            if args and isinstance(args[0], list) and len(args[0]) > 0: return args[0].pop()
            return _NA
        if fid == 87:
            if args and isinstance(args[0], list) and len(args) > 1:
                idx = int(args[1])
                if 0 <= idx < len(args[0]): args[0].pop(idx)
            return None
        if fid == 88:
            if args and isinstance(args[0], list): args[0].clear()
            return None
        if fid == 89:
            if args and isinstance(args[0], list):
                vals = [v for v in args[0] if not _is_na(v)]
                return min(vals) if vals else _NA
            return _NA
        if fid == 90:
            if args and isinstance(args[0], list):
                vals = [v for v in args[0] if not _is_na(v)]
                return max(vals) if vals else _NA
            return _NA

        # Visual no-ops
        if name in ('plot', 'plotshape', 'plotchar', 'plotarrow', 'bgcolor',
                     'barcolor', 'hline', 'fill', 'label.new', 'label.delete',
                     'line.new', 'line.delete', 'line.set_x2', 'line.set_color', 'line.set_width',
                     'box.new', 'box.delete', 'box.set_right',
                     'table.new', 'table.cell',
                     'alertcondition', 'alert', 'color.new', 'color.rgb',
                     'log.info', 'log.warning', 'log.error', 'timestamp', 'time'):
            return None if name != 'color.new' else (args[0] if args else '#000000')

        # User-defined functions
        if name in self.user_functions:
            return self._call_user_func(name, args)

        return _NA

    # ─── Strategy functions ───────────────────────────────────────────────

    cdef object _fn_strategy(self, list args, dict kwargs):
        self.strategy_config = {
            'title': args[0] if args else kwargs.get('title', 'Strategy'),
            'overlay': kwargs.get('overlay', True),
            'initial_capital': kwargs.get('initial_capital', 10000),
            'default_qty_type': kwargs.get('default_qty_type', 'fixed'),
            'default_qty_value': kwargs.get('default_qty_value', 1),
            'commission_type': kwargs.get('commission_type', 'percent'),
            'commission_value': kwargs.get('commission_value', 0),
            'pyramiding': kwargs.get('pyramiding', 0),
            'currency': kwargs.get('currency', 'USD'),
        }
        return None

    cdef object _fn_strategy_entry(self, list args, dict kwargs):
        id_ = args[0] if args else kwargs.get('id', '')
        direction = args[1] if len(args) > 1 else kwargs.get('direction', 'long')
        qty = kwargs.get('qty', None)
        comment = kwargs.get('comment', id_)
        direction = 'long' if direction == 'long' else 'short'
        self.signals.append(Signal('entry', direction, qty, comment))
        return None

    cdef object _fn_strategy_close(self, list args, dict kwargs):
        id_ = args[0] if args else kwargs.get('id', '')
        comment = kwargs.get('comment', id_)
        direction = 'long' if 'short' not in str(id_).lower() else 'short'
        self.signals.append(Signal('close', direction, comment=comment))
        return None

    cdef object _fn_strategy_exit(self, list args, dict kwargs):
        id_ = args[0] if args else kwargs.get('id', '')
        from_entry = args[1] if len(args) > 1 else kwargs.get('from_entry', '')
        limit = kwargs.get('limit', None)
        stop = kwargs.get('stop', None)
        comment = kwargs.get('comment', id_)
        direction = 'long' if 'short' not in str(from_entry).lower() else 'short'
        self.signals.append(Signal('exit', direction, comment=comment, limit=limit, stop=stop, from_entry=from_entry))
        return None

    cdef object _fn_input(self, list args, dict kwargs):
        defval = args[0] if args else kwargs.get('defval', 0)
        title = kwargs.get('title', f'input_{len(self.inputs)}')
        if title in self.inputs:
            return self.inputs[title]
        self.inputs[title] = defval
        return defval

    cdef object _fn_request_security(self, list args, dict kwargs):
        if self._smt_row_cache is not None and len(args) >= 3:
            expr_val = args[2]
            smt_col = None
            for col in ('high', 'low', 'close', 'open', 'volume'):
                if expr_val == self.variables.get(col):
                    smt_col = col
                    break
            if smt_col is not None and smt_col in self._smt_series:
                return (<FastSeries>self._smt_series[smt_col]).get(0)
            return expr_val
        if len(args) >= 3:
            return args[2]
        return _NA

    cdef object _call_user_func(self, str name, list args):
        fdef = self.user_functions[name]
        cdef dict saved = {}
        cdef int i
        for i, p in enumerate(fdef.params):
            saved[p] = self.variables.get(p)
            self.variables[p] = args[i] if i < len(args) else _NA
        cdef object result = _NA
        for stmt in fdef.body:
            if stmt._tag == 15:
                result = self._eval(stmt.expr)
            else:
                self._exec(stmt)
        for p in fdef.params:
            if saved[p] is not None:
                self.variables[p] = saved[p]
            else:
                self.variables.pop(p, None)
        return result

    # ─── Source resolution and history ────────────────────────────────────

    @cython.wraparound(True)
    cdef str _resolve_source(self, object arg):
        if isinstance(arg, str) and arg in self.series_data:
            return arg
        cdef dict v = self.variables
        if arg == v.get('close'): return 'close'
        if arg == v.get('open'): return 'open'
        if arg == v.get('high'): return 'high'
        if arg == v.get('low'): return 'low'
        if arg == v.get('hl2'): return 'hl2'
        if arg == v.get('hlc3'): return 'hlc3'
        if arg == v.get('ohlc4'): return 'ohlc4'
        for nm, series in self.series_data.items():
            if (<FastSeries>series).data and (<FastSeries>series).data[-1] == arg:
                return nm
        return 'close'

    cdef list _get_history(self, object source, int length):
        cdef str series_name = self._resolve_source(source)
        cdef FastSeries series = <FastSeries>self.series_data.get(series_name)
        if series is None or len(series.data) == 0:
            return []
        cdef list values = []
        cdef int i
        cdef object v
        for i in range(length):
            v = series.get(i)
            values.append(None if _is_na(v) else float(v))
        values.reverse()
        return values

    # ─── Technical indicators ─────────────────────────────────────────────

    cdef object _ta_pivothigh(self, list args, dict kwargs):
        cdef str source_name
        cdef int left, right
        if len(args) == 3:
            source_name = self._resolve_source(args[0])
            left = <int>args[1]; right = <int>args[2]
        elif len(args) == 2:
            source_name = 'high'
            left = <int>args[0]; right = <int>args[1]
        else:
            source_name = 'high'; left = 5; right = 5
        cdef FastSeries series = <FastSeries>self.series_data.get(source_name)
        if series is None or len(series) < left + right + 1: return _NA
        cdef int pivot_idx = right
        cdef object pivot_val = series.get(pivot_idx)
        if _is_na(pivot_val): return _NA
        cdef int i
        cdef object v
        for i in range(1, left + 1):
            v = series.get(pivot_idx + i)
            if _is_na(v) or v >= pivot_val: return _NA
        for i in range(1, right + 1):
            v = series.get(pivot_idx - i)
            if _is_na(v) or v >= pivot_val: return _NA
        return float(pivot_val)

    cdef object _ta_pivotlow(self, list args, dict kwargs):
        cdef str source_name
        cdef int left, right
        if len(args) == 3:
            source_name = self._resolve_source(args[0])
            left = <int>args[1]; right = <int>args[2]
        elif len(args) == 2:
            source_name = 'low'
            left = <int>args[0]; right = <int>args[1]
        else:
            source_name = 'low'; left = 5; right = 5
        cdef FastSeries series = <FastSeries>self.series_data.get(source_name)
        if series is None or len(series) < left + right + 1: return _NA
        cdef int pivot_idx = right
        cdef object pivot_val = series.get(pivot_idx)
        if _is_na(pivot_val): return _NA
        cdef int i
        cdef object v
        for i in range(1, left + 1):
            v = series.get(pivot_idx + i)
            if _is_na(v) or v <= pivot_val: return _NA
        for i in range(1, right + 1):
            v = series.get(pivot_idx - i)
            if _is_na(v) or v <= pivot_val: return _NA
        return float(pivot_val)

    cdef object _ta_sma(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cdef list values = self._get_history(source, length)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < length: return _NA
        cdef double total = 0.0
        for v in valid: total += <double>v
        return total / length

    cdef object _ta_ema(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cdef str series_name = self._resolve_source(source)
        cdef str cache_key = f'_ema_{series_name}_{length}'
        cdef double current_val
        if isinstance(source, (int, float)) and not _is_na(source):
            current_val = <double>source
        else:
            cv = self.variables.get(series_name, _NA)
            if _is_na(cv): return _NA
            current_val = <double>cv
        prev_ema = self.variables.get(cache_key, _NA)
        cdef double ema
        if _is_na(prev_ema):
            values = self._get_history(source, length)
            valid = [v for v in values if v is not None]
            if len(valid) < length: return _NA
            ema = sum(valid) / len(valid)
        else:
            k = 2.0 / (length + 1)
            ema = current_val * k + <double>prev_ema * (1.0 - k)
        self.variables[cache_key] = ema
        return ema

    cdef object _ta_rsi(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cdef list values = self._get_history(source, length + 1)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < length + 1: return _NA
        cdef list gains = [max(valid[i] - valid[i-1], 0) for i in range(1, len(valid))]
        cdef list losses = [max(valid[i-1] - valid[i], 0) for i in range(1, len(valid))]
        cdef double ag = sum(gains) / len(gains)
        cdef double al = sum(losses) / len(losses)
        if al == 0.0: return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    cdef object _ta_atr(self, list args, dict kwargs):
        cdef int length = <int>(args[0] if args else kwargs.get('length', 14))
        cdef list tr_values = []
        cdef int i
        cdef double h, l, c_p, tr
        for i in range(length):
            hv = (<FastSeries>self.series_data['high']).get(i)
            lv = (<FastSeries>self.series_data['low']).get(i)
            cv = (<FastSeries>self.series_data['close']).get(i + 1)
            if _is_na(hv) or _is_na(lv): return _NA
            h = <double>hv; l = <double>lv
            if not _is_na(cv):
                c_p = <double>cv
                tr = max(h - l, fabs(h - c_p), fabs(l - c_p))
            else:
                tr = h - l
            tr_values.append(tr)
        if not tr_values: return _NA
        return sum(tr_values) / len(tr_values)

    cdef object _ta_wma(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cdef list values = self._get_history(source, length)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < length: return _NA
        cdef double ws = <double>(length * (length + 1) / 2)
        cdef double total = 0.0
        cdef int i
        for i in range(len(valid)):
            total += <double>valid[i] * (i + 1)
        return total / ws

    cdef object _ta_macd(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int fast = <int>(args[1] if len(args) > 1 else kwargs.get('fastlen', 12))
        cdef int slow = <int>(args[2] if len(args) > 2 else kwargs.get('slowlen', 26))
        fast_ema = self._ta_ema([source, fast], {})
        slow_ema = self._ta_ema([source, slow], {})
        if _is_na(fast_ema) or _is_na(slow_ema): return (_NA, _NA, _NA)
        return (<double>fast_ema - <double>slow_ema, _NA, _NA)

    cdef object _ta_crossover(self, list args):
        if len(args) < 2: return False
        a_now = args[0]; b_now = args[1]
        if _is_na(a_now) or _is_na(b_now): return False
        cdef str a_name = self._resolve_source(a_now)
        cdef str b_name = self._resolve_source(b_now)
        if a_name in self.series_data and b_name in self.series_data:
            a_prev = (<FastSeries>self.series_data[a_name]).get(1)
            b_prev = (<FastSeries>self.series_data[b_name]).get(1)
            if not _is_na(a_prev) and not _is_na(b_prev):
                return float(a_now) > float(b_now) and float(a_prev) <= float(b_prev)
        return False

    cdef object _ta_crossunder(self, list args):
        if len(args) < 2: return False
        a_now = args[0]; b_now = args[1]
        if _is_na(a_now) or _is_na(b_now): return False
        cdef str a_name = self._resolve_source(a_now)
        cdef str b_name = self._resolve_source(b_now)
        if a_name in self.series_data and b_name in self.series_data:
            a_prev = (<FastSeries>self.series_data[a_name]).get(1)
            b_prev = (<FastSeries>self.series_data[b_name]).get(1)
            if not _is_na(a_prev) and not _is_na(b_prev):
                return float(a_now) < float(b_now) and float(a_prev) >= float(b_prev)
        return False

    cdef object _ta_highest(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('high', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cdef list values = self._get_history(source, length)
        cdef list valid = [v for v in values if v is not None]
        return max(valid) if valid else _NA

    cdef object _ta_lowest(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('low', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cdef list values = self._get_history(source, length)
        cdef list valid = [v for v in values if v is not None]
        return min(valid) if valid else _NA

    cdef object _ta_change(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 1))
        cdef str series_name = self._resolve_source(source)
        cdef FastSeries series = <FastSeries>self.series_data.get(series_name)
        if series is None: return _NA
        current = series.get(0); prev = series.get(length)
        if not _is_na(current) and not _is_na(prev):
            return <double>current - <double>prev
        return _NA

    cdef object _ta_stoch(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[3] if len(args) > 3 else kwargs.get('length', 14))
        cdef list highs = self._get_history(self.variables.get('high', 0), length)
        cdef list lows = self._get_history(self.variables.get('low', 0), length)
        cdef list vh = [v for v in highs if v is not None]
        cdef list vl = [v for v in lows if v is not None]
        if len(vh) < length or len(vl) < length: return _NA
        cdef double highest = max(vh)
        cdef double lowest = min(vl)
        cdef double current
        if isinstance(source, (int, float)):
            current = <double>source
        else:
            current = <double>self.variables.get('close', 0)
        if highest != lowest:
            return 100.0 * (current - lowest) / (highest - lowest)
        return 50.0

    cdef object _ta_bb(self, list args, dict kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        cdef int length = <int>(args[1] if len(args) > 1 else kwargs.get('length', 20))
        cdef double mult = <double>(args[2] if len(args) > 2 else kwargs.get('mult', 2.0))
        cdef list values = self._get_history(source, length)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < length: return (_NA, _NA, _NA)
        cdef double basis = sum(valid) / len(valid)
        cdef double std_val = sqrt(sum((<double>v - basis) ** 2 for v in valid) / len(valid))
        return (basis, basis + mult * std_val, basis - mult * std_val)
