# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-optimized MQL5 interpreter.
Drop-in replacement for MQL5Interpreter from mql5_parser.py.
"""
cimport cython
from libc.math cimport sqrt, log, log10, exp, ceil, floor, pow as cpow, fabs, isnan, isinf, NAN, M_PI, sin, cos, tan
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject

import math

from mql5_parser import (
    NumberLit, StringLit, BoolLit, NullLit, Identifier, ArrayLit,
    BinOp, UnaryOp, ArrayAccess, FuncCall, DotAccess, Ternary,
    Assignment, IfStmt, ForStmt, ExprStmt, ContinueStmt, BreakStmt,
    SwitchStmt, FuncDef, Program, WhileStmt, DoWhileStmt, ReturnStmt,
    VarDecl, InputDecl, PreprocessorDir, EnumDecl, CastExpr, PostfixOp,
    MQL5NA, NA, Series, Signal,
    LoopBreak, LoopContinue, FuncReturn,
    MQL5_CONSTANTS, is_na,
)

# ---- Module-level constants ------------------------------------------------

cdef object _NA = NA
cdef object _LoopBreak = LoopBreak
cdef object _LoopContinue = LoopContinue
cdef object _FuncReturn = FuncReturn

# ---- Fast NA check ----------------------------------------------------------

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

# ---- FastSeries -------------------------------------------------------------

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


# ---- FastMQL5Interpreter ---------------------------------------------------

cdef class FastMQL5Interpreter:
    cdef public object ast
    cdef public dict variables
    cdef public dict series_data
    cdef public int bar_index
    cdef public object bars
    cdef public list signals
    cdef public list all_signals
    cdef public dict strategy_config
    cdef public dict inputs
    cdef public dict user_functions
    cdef public bint _first_bar

    # MQL5-specific
    cdef public dict _indicator_handles
    cdef public int _next_handle
    cdef public bint _ctrade_available
    cdef public int _position_type
    cdef public double _position_volume
    cdef public double _position_price_open
    cdef public double _position_sl
    cdef public double _position_tp
    cdef public double _position_profit
    cdef public double _account_balance
    cdef public double _account_equity
    cdef public dict _array_series_flags
    cdef public double _contract_size
    cdef public bint _is_jpy_pair
    cdef public double _pip_value

    # OnInit / OnTick / OnDeinit
    cdef public object _on_init
    cdef public object _on_tick
    cdef public object _on_deinit
    cdef public list _global_stmts

    # SMT data
    cdef public list _smt_row_cache
    cdef public dict _smt_series

    # Row cache and timestamps
    cdef public list _row_cache
    cdef public dict _ema_cache
    cdef public list _timestamps

    # C arrays for OHLCV fast access
    cdef double* _c_open
    cdef double* _c_high
    cdef double* _c_low
    cdef double* _c_close
    cdef double* _c_vol
    cdef int _bar_count

    # SMT C arrays
    cdef double* _c_smt_open
    cdef double* _c_smt_high
    cdef double* _c_smt_low
    cdef double* _c_smt_close
    cdef double* _c_smt_vol
    cdef int _smt_bar_count

    def __init__(self, object ast):
        self.ast = ast
        self.variables = {}
        self.series_data = {}
        self.bar_index = 0
        self.bars = None
        self.signals = []
        self.all_signals = []
        self.strategy_config = {}
        self.inputs = {}
        self.user_functions = {}
        self._first_bar = True

        # MQL5-specific
        self._indicator_handles = {}
        self._next_handle = 1
        self._ctrade_available = False
        self._position_type = -1
        self._position_volume = 0.0
        self._position_price_open = 0.0
        self._position_sl = 0.0
        self._position_tp = 0.0
        self._position_profit = 0.0
        self._account_balance = 10000.0
        self._account_equity = 10000.0
        self._array_series_flags = {}
        self._contract_size = 100000.0
        self._is_jpy_pair = False
        self._pip_value = 1.0

        self._on_init = None
        self._on_tick = None
        self._on_deinit = None
        self._global_stmts = []

        self._smt_row_cache = None
        self._smt_series = {}
        self._row_cache = []
        self._ema_cache = {}
        self._timestamps = []

        # C arrays
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

        # Load constants BEFORE prescan
        self.variables.update(MQL5_CONSTANTS)

        # Pre-scan AST
        self._prescan_ast()

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

    # ---- Prescan AST --------------------------------------------------------

    cdef void _prescan_ast(self):
        """Separate top-level statements into globals, functions, and OnXxx handlers."""
        for stmt in self.ast.stmts:
            if isinstance(stmt, FuncDef):
                if stmt.name == 'OnInit':
                    self._on_init = stmt.body
                elif stmt.name == 'OnTick':
                    self._on_tick = stmt.body
                elif stmt.name == 'OnDeinit':
                    self._on_deinit = stmt.body
                else:
                    self.user_functions[stmt.name] = stmt
            elif isinstance(stmt, PreprocessorDir):
                if 'Trade.mqh' in stmt.directive or 'Trade/Trade' in stmt.directive or 'Trade\\Trade' in stmt.directive:
                    self._ctrade_available = True
                if stmt.directive.startswith('define '):
                    parts = stmt.directive.split(None, 2)
                    if len(parts) >= 3:
                        dname = parts[1]
                        try:
                            dval = float(parts[2]) if '.' in parts[2] else int(parts[2])
                        except ValueError:
                            dval = parts[2]
                        self.variables[dname] = dval
                    elif len(parts) == 2:
                        self.variables[parts[1]] = 1
            elif isinstance(stmt, InputDecl):
                val = self._eval(stmt.value) if stmt.value else self._default_for_type(stmt.type_name)
                self.inputs[stmt.name] = val
                self.variables[stmt.name] = val
            elif isinstance(stmt, EnumDecl):
                for vname, vval in stmt.values.items():
                    self.variables[vname] = vval
                    MQL5_CONSTANTS[vname] = vval
            else:
                self._global_stmts.append(stmt)

    cdef object _default_for_type(self, str type_name):
        if type_name in ('int', 'long', 'short', 'char', 'uchar', 'ushort', 'uint', 'ulong'):
            return 0
        if type_name in ('double', 'float'):
            return 0.0
        if type_name == 'string':
            return ''
        if type_name == 'bool':
            return False
        return 0

    # ---- Setup --------------------------------------------------------------

    cpdef setup(self, object bars):
        self.bars = bars
        self.bar_index = 0
        self._first_bar = True
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self.series_data[col] = FastSeries()

        cdef int n = len(bars)
        self._bar_count = n

        # Allocate C arrays
        if self._c_open != NULL: free(self._c_open)
        self._c_open = <double*>malloc(n * sizeof(double))
        self._c_high = <double*>malloc(n * sizeof(double))
        self._c_low = <double*>malloc(n * sizeof(double))
        self._c_close = <double*>malloc(n * sizeof(double))
        self._c_vol = <double*>malloc(n * sizeof(double))

        # Precompute row data and timestamps
        self._row_cache = []
        self._timestamps = []
        date_col = None
        for col in ('datetime', 'date', 'Date', 'Datetime'):
            if col in bars.columns:
                date_col = col
                break

        cdef int i
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

            # Parse real timestamp
            ts = 0
            if date_col:
                try:
                    import pandas as pd
                    dt = pd.Timestamp(r[date_col])
                    ts = int(dt.timestamp())
                except Exception:
                    ts = 1700000000 + i * 300
            else:
                ts = 1700000000 + i * 300
            self._timestamps.append(ts)

        # Detect instrument type from price levels
        cdef double avg_price
        if n > 0:
            avg_price = bars['close'].iloc[:min(100, n)].mean()
        else:
            avg_price = 1.0

        if avg_price > 10:  # JPY pair
            self._is_jpy_pair = True
            self.variables['_Point'] = 0.001
            self.variables['_Digits'] = 3
            self._pip_value = self._contract_size * 0.01 / avg_price
        elif avg_price > 1:  # Standard forex
            self._is_jpy_pair = False
            self.variables['_Point'] = 0.00001
            self.variables['_Digits'] = 5
            self._pip_value = self._contract_size * 0.0001
        else:  # Crypto or fractional
            self._is_jpy_pair = False
            self.variables['_Point'] = 0.01
            self.variables['_Digits'] = 2
            self._pip_value = 1.0

        # Set initial constants
        self.variables['_Symbol'] = 'SYMBOL'
        self.variables['_Period'] = 0
        self.variables['_LastError'] = 0
        self.variables['Ask'] = 0.0
        self.variables['Bid'] = 0.0
        self.variables['Bars'] = n
        self.variables['INIT_SUCCEEDED'] = 0

        # Load MQL5 constants
        self.variables.update(MQL5_CONSTANTS)

        # Execute global statements
        for stmt in self._global_stmts:
            self._exec(stmt)

        # Execute OnInit
        if self._on_init:
            try:
                for stmt in self._on_init:
                    self._exec(stmt)
            except FuncReturn:
                pass

        # Build strategy config
        self.strategy_config = {
            'title': 'MQL5 Expert Advisor',
            'overlay': True,
        }

    cpdef setup_secondary(self, object bars):
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

    # ---- Run bar ------------------------------------------------------------

    cpdef run_bar(self, int idx):
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
        v['Ask'] = c; v['Bid'] = c

        # Execute OnTick
        if self._on_tick:
            try:
                for stmt in self._on_tick:
                    self._exec(stmt)
            except FuncReturn:
                pass

        if self.signals:
            self.all_signals.extend([(idx, s) for s in self.signals])
        self._first_bar = False

    cpdef list run_all_bars(self):
        self.all_signals = []
        cdef int i
        for i in range(self._bar_count):
            self.run_bar(i)
        return self.all_signals

    # ---- _exec (statement execution) ----------------------------------------

    cdef void _exec(self, object node):
        cdef int tag = node._tag

        if tag == 12:  # Assignment
            self._exec_assign(node)
        elif tag == 24:  # VarDecl
            self._exec_vardecl(node)
        elif tag == 25:  # InputDecl
            if node.name not in self.variables:
                val = self._eval(node.value) if node.value else self._default_for_type(node.type_name)
                self.variables[node.name] = val
                self.inputs[node.name] = val
        elif tag == 13:  # IfStmt
            if _truthy(self._eval(node.cond)):
                for s in node.body: self._exec(s)
            elif node.else_body:
                for s in node.else_body: self._exec(s)
        elif tag == 14:  # ForStmt
            self._exec_for(node)
        elif tag == 21:  # WhileStmt
            self._exec_while(node)
        elif tag == 22:  # DoWhileStmt
            self._exec_dowhile(node)
        elif tag == 18:  # SwitchStmt
            self._exec_switch(node)
        elif tag == 23:  # ReturnStmt
            val = self._eval(node.value) if node.value else None
            raise _FuncReturn(val)
        elif tag == 16:  # ContinueStmt
            raise _LoopContinue()
        elif tag == 17:  # BreakStmt
            raise _LoopBreak()
        elif tag == 15:  # ExprStmt
            self._eval(node.expr)
        elif tag == 19:  # FuncDef
            self.user_functions[node.name] = node
        elif tag == 26:  # PreprocessorDir
            pass  # Already handled in prescan
        elif tag == 27:  # EnumDecl
            for vname, vval in node.values.items():
                self.variables[vname] = vval

    cdef void _exec_assign(self, object node):
        cdef object val = self._eval(node.value)
        cdef str name = node.name
        cdef str op = node.op

        if op == '=':
            self.variables[name] = val
        elif op == '+=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '+')
        elif op == '-=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '-')
        elif op == '*=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '*')
        elif op == '/=':
            self.variables[name] = self._arith(self.variables.get(name, 0), val, '/')

    cdef void _exec_vardecl(self, object node):
        if node.is_array:
            if node.value is not None:
                self.variables[node.name] = self._eval(node.value)
            else:
                self.variables[node.name] = []
        else:
            if node.value is not None:
                self.variables[node.name] = self._eval(node.value)
            elif node.name not in self.variables:
                self.variables[node.name] = self._default_for_type(node.type_name)

    cdef void _exec_for(self, object node):
        if node.init:
            if isinstance(node.init, (Assignment, VarDecl, ExprStmt)):
                self._exec(node.init)
            else:
                self._eval(node.init)
        cdef int max_iter = 100000
        cdef int count = 0
        while count < max_iter:
            if node.cond:
                if not _truthy(self._eval(node.cond)):
                    break
            try:
                for stmt in node.body:
                    self._exec(stmt)
            except LoopContinue:
                pass
            except LoopBreak:
                break
            if node.step:
                self._eval(node.step)
            count += 1

    cdef void _exec_while(self, object node):
        cdef int max_iter = 100000
        cdef int count = 0
        while count < max_iter and _truthy(self._eval(node.cond)):
            try:
                for stmt in node.body:
                    self._exec(stmt)
            except LoopContinue:
                count += 1
                continue
            except LoopBreak:
                break
            count += 1

    cdef void _exec_dowhile(self, object node):
        cdef int max_iter = 100000
        cdef int count = 0
        while count < max_iter:
            try:
                for stmt in node.body:
                    self._exec(stmt)
            except LoopContinue:
                pass
            except LoopBreak:
                break
            if not _truthy(self._eval(node.cond)):
                break
            count += 1

    cdef void _exec_switch(self, object node):
        cdef object val = self._eval(node.expr)
        cdef bint matched = False
        for cond, body in node.cases:
            cv = self._eval(cond)
            if matched or val == cv:
                matched = True
                try:
                    for stmt in body:
                        self._exec(stmt)
                except LoopBreak:
                    matched = False
                    break

        if not matched and node.default:
            try:
                for stmt in node.default:
                    self._exec(stmt)
            except LoopBreak:
                pass

    # ---- _eval (expression evaluation) --------------------------------------

    cdef object _eval(self, object node):
        if node is None:
            return _NA
        cdef int tag = node._tag

        if tag == 0:  # NumberLit
            return node.value
        elif tag == 1:  # StringLit
            return node.value
        elif tag == 2:  # BoolLit
            return node.value
        elif tag == 3:  # NullLit
            return _NA
        elif tag == 5:  # ArrayLit
            return [self._eval(e) for e in node.elements]
        elif tag == 4:  # Identifier
            return self._eval_identifier(node)
        elif tag == 6:  # BinOp
            return self._eval_binop(node)
        elif tag == 7:  # UnaryOp
            return self._eval_unaryop(node)
        elif tag == 29:  # PostfixOp
            return self._eval_postfixop(node)
        elif tag == 11:  # Ternary
            return self._eval(node.true_val) if _truthy(self._eval(node.cond)) else self._eval(node.false_val)
        elif tag == 8:  # ArrayAccess
            return self._eval_array_access(node)
        elif tag == 28:  # CastExpr
            return self._eval_cast(node)
        elif tag == 9:  # FuncCall
            return self._call(node)
        elif tag == 10:  # DotAccess
            return self._eval_dot_access(node)
        elif tag == 13:  # IfStmt as expression
            return self._eval_if_expr(node)
        return _NA

    cdef object _eval_identifier(self, object node):
        cdef str name = node.name
        val = self.variables.get(name)
        if val is not None:
            return val
        val = MQL5_CONSTANTS.get(name)
        if val is not None:
            return val
        if name in self.series_data:
            return (<FastSeries>self.series_data[name]).get(0)
        return _NA

    cdef object _eval_unaryop(self, object node):
        cdef str op = node.op
        if op == '++pre':
            name = node.operand.name if isinstance(node.operand, Identifier) else None
            if name:
                self.variables[name] = self.variables.get(name, 0) + 1
                return self.variables[name]
            return _NA
        if op == '--pre':
            name = node.operand.name if isinstance(node.operand, Identifier) else None
            if name:
                self.variables[name] = self.variables.get(name, 0) - 1
                return self.variables[name]
            return _NA
        cdef object val = self._eval(node.operand)
        if op == '-':
            return -val if not _is_na(val) else _NA
        if op == '!':
            return not _truthy(val)
        return val

    cdef object _eval_postfixop(self, object node):
        name = node.operand.name if isinstance(node.operand, Identifier) else None
        if name:
            old = self.variables.get(name, 0)
            if node.op == '++':
                self.variables[name] = old + 1
            elif node.op == '--':
                self.variables[name] = old - 1
            return old
        return _NA

    cdef object _eval_cast(self, object node):
        cdef object val = self._eval(node.expr)
        if _is_na(val):
            return _NA
        cdef str tt = node.target_type
        if tt in ('int', 'long', 'short', 'char', 'uchar', 'ushort', 'uint', 'ulong'):
            return int(val)
        if tt in ('double', 'float'):
            return float(val)
        if tt == 'string':
            return str(val)
        if tt == 'bool':
            return bool(val)
        return val

    cdef object _eval_dot_access(self, object node):
        cdef str full = self._dot_name(node)
        if full:
            val = self.variables.get(full)
            if val is not None:
                return val
            val = MQL5_CONSTANTS.get(full)
            if val is not None:
                return val
        return _NA

    cdef object _eval_if_expr(self, object node):
        cdef object cond = self._eval(node.cond)
        cdef object result
        if _truthy(cond):
            result = _NA
            for stmt in node.body:
                if stmt._tag == 15:
                    result = self._eval(stmt.expr)
                else:
                    self._exec(stmt)
            return result
        elif node.else_body:
            result = _NA
            for stmt in node.else_body:
                if stmt._tag == 15:
                    result = self._eval(stmt.expr)
                else:
                    self._exec(stmt)
            return result
        return _NA

    @cython.wraparound(True)
    cdef object _eval_array_access(self, object node):
        """Handle Close[0], buffer[i], etc."""
        cdef str name = self._node_name(node.expr)
        cdef object index_val = self._eval(node.index)
        if _is_na(index_val):
            return _NA
        cdef int index = int(index_val)

        # Built-in timeseries arrays
        cdef str mapped = None
        if name == 'Close' or name == 'close':
            mapped = 'close'
        elif name == 'Open' or name == 'open':
            mapped = 'open'
        elif name == 'High' or name == 'high':
            mapped = 'high'
        elif name == 'Low' or name == 'low':
            mapped = 'low'
        elif name == 'Volume' or name == 'volume':
            mapped = 'volume'

        if mapped is not None:
            series = self.series_data.get(mapped)
            if series:
                return (<FastSeries>series).get(index)
            return _NA

        # Regular array variable
        arr = self.variables.get(name) if name else self._eval(node.expr)
        if isinstance(arr, list):
            is_series = self._array_series_flags.get(id(arr), False)
            if is_series:
                idx = len(arr) - 1 - index
            else:
                idx = index
            if 0 <= idx < len(arr):
                return arr[idx]
            return _NA
        return _NA

    cdef object _eval_binop(self, object node):
        cdef object left = self._eval(node.left)
        cdef object right = self._eval(node.right)
        cdef int op_id = node.op_id

        if op_id == 11:  # &&
            return _truthy(left) and _truthy(right)
        if op_id == 12:  # ||
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
        if isinstance(node, DotAccess):
            p = self._dot_name(node.obj)
            if p is not None:
                return p + '.' + node.attr
            return None
        if isinstance(node, Identifier):
            return node.name
        return None

    cdef str _node_name(self, object node):
        if isinstance(node, Identifier):
            return node.name
        if isinstance(node, DotAccess):
            return self._dot_name(node)
        return None

    # ---- _call (built-in function dispatch) ---------------------------------

    @cython.wraparound(True)
    cdef object _call(self, object node):
        cdef str name = node.name
        cdef list args = [self._eval(a) for a in node.args]

        # ---- CTrade methods -- match any object.Buy/Sell/etc. ----
        cdef str short_name = name.split('.')[-1] if '.' in name else name

        if short_name == 'Buy':
            return self._call_buy(args)

        if short_name == 'Sell':
            return self._call_sell(args)

        if short_name == 'PositionOpen':
            return self._call_position_open(args)

        if short_name in ('PositionClose', 'PositionCloseBy'):
            self.signals.append(Signal('close', 'all'))
            return True

        if short_name == 'PositionModify':
            sl = args[1] if len(args) > 1 and not _is_na(args[1]) and args[1] != 0 else None
            tp = args[2] if len(args) > 2 and not _is_na(args[2]) and args[2] != 0 else None
            direction = 'long' if self._position_type == 0 else 'short'
            self.signals.append(Signal('exit', direction, comment='modify', stop=sl, limit=tp))
            return True

        # CTrade no-ops
        if (short_name.startswith('Set') and '.' in name) or short_name.startswith('Request') or short_name.startswith('Check') or short_name.startswith('Result'):
            return True

        # ---- Position queries ----
        if name == 'PositionSelect':
            return self._position_type >= 0

        if name == 'PositionGetDouble':
            return self._call_position_get_double(args)

        if name == 'PositionGetInteger':
            prop = args[0] if args else 0
            if prop == 106: return self._position_type
            return 0

        if name == 'PositionGetString':
            prop = args[0] if args else 0
            if prop == 200: return 'SYMBOL'
            return ''

        if name == 'PositionsTotal':
            return 1 if self._position_type >= 0 else 0

        if name == 'PositionGetTicket':
            return 1 if self._position_type >= 0 else 0

        # ---- OrderSend ----
        if name == 'OrderSend':
            return True

        # ---- Symbol info ----
        if name == 'SymbolInfoDouble':
            return self._call_symbol_info_double(args)

        if name == 'SymbolInfoInteger':
            prop = args[1] if len(args) > 1 else 0
            if prop == 203: return self.variables.get('_Digits', 5)
            return 0

        if name == 'SymbolInfoString':
            return self._call_symbol_info_string(args)

        if name == 'SymbolInfoTick':
            return True

        if name == 'PeriodSeconds':
            return self._call_period_seconds(args)

        if name == 'TimeToStruct':
            return self._call_time_to_struct(node, args)

        # ---- Account info ----
        if name == 'AccountInfoDouble':
            prop = args[0] if args else 0
            if prop == 300: return self._account_balance
            if prop == 301: return self._account_equity
            if prop == 302: return self._position_profit
            return 0.0

        # ---- Indicator handles ----
        if name == 'iMA':
            return self._create_indicator('MA', args)
        if name == 'iRSI':
            return self._create_indicator('RSI', args)
        if name == 'iMACD':
            return self._create_indicator('MACD', args)
        if name == 'iATR':
            return self._create_indicator('ATR', args)
        if name == 'iBands':
            return self._create_indicator('BB', args)
        if name == 'iStochastic':
            return self._create_indicator('STOCH', args)
        if name == 'iCCI':
            return self._create_indicator('CCI', args)
        if name == 'iMomentum':
            return self._create_indicator('MOM', args)
        if name == 'iADX':
            return self._create_indicator('ADX', args)
        if name == 'iCustom':
            return self._create_indicator('CUSTOM', args)

        # ---- CopyBuffer ----
        if name == 'CopyBuffer':
            return self._copy_buffer(args)

        # ---- Array functions ----
        if name == 'ArrayResize':
            return self._call_array_resize(args)
        if name == 'ArraySetAsSeries':
            arr = args[0] if args else []
            flag = bool(args[1]) if len(args) > 1 else True
            if isinstance(arr, list):
                self._array_series_flags[id(arr)] = flag
            return True
        if name == 'ArraySize':
            arr = args[0] if args else []
            return len(arr) if isinstance(arr, list) else 0
        if name == 'ArrayCopy':
            return self._call_array_copy(args)
        if name == 'ArrayInitialize':
            arr = args[0] if args else []
            val = args[1] if len(args) > 1 else 0.0
            if isinstance(arr, list):
                for i in range(len(arr)): arr[i] = val
            return None
        if name == 'ArrayFree':
            arr = args[0] if args else []
            if isinstance(arr, list): arr.clear()
            return None
        if name == 'ArrayMaximum':
            arr = args[0] if args else []
            if isinstance(arr, list) and arr:
                return arr.index(max(arr))
            return -1
        if name == 'ArrayMinimum':
            arr = args[0] if args else []
            if isinstance(arr, list) and arr:
                return arr.index(min(arr))
            return -1

        # ---- iClose, iOpen, iHigh, iLow, iVolume ----
        if name in ('iClose', 'iOpen', 'iHigh', 'iLow', 'iVolume'):
            return self._call_i_series(name, args)

        if name == 'iTime':
            return self._call_itime(args)

        if name == 'iBarShift':
            return 0

        # ---- Math functions ----
        if name in ('MathSqrt', 'sqrt'):
            return sqrt(<double>args[0]) if args and not _is_na(args[0]) and float(args[0]) >= 0 else _NA
        if name in ('MathAbs', 'fabs'):
            return fabs(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if name in ('MathMax', 'fmax'):
            if len(args) >= 2 and not _is_na(args[0]) and not _is_na(args[1]):
                return max(float(args[0]), float(args[1]))
            return _NA
        if name in ('MathMin', 'fmin'):
            if len(args) >= 2 and not _is_na(args[0]) and not _is_na(args[1]):
                return min(float(args[0]), float(args[1]))
            return _NA
        if name in ('MathPow', 'pow'):
            return cpow(<double>args[0], <double>args[1]) if len(args) >= 2 and not _is_na(args[0]) and not _is_na(args[1]) else _NA
        if name in ('MathLog', 'log'):
            return log(<double>args[0]) if args and not _is_na(args[0]) and float(args[0]) > 0 else _NA
        if name == 'MathLog10':
            return log10(<double>args[0]) if args and not _is_na(args[0]) and float(args[0]) > 0 else _NA
        if name in ('MathExp', 'exp'):
            return exp(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if name in ('MathFloor', 'floor'):
            return floor(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if name in ('MathCeil', 'ceil'):
            return ceil(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if name in ('MathRound', 'round'):
            return round(args[0]) if args and not _is_na(args[0]) else _NA
        if name == 'NormalizeDouble':
            if len(args) >= 2 and not _is_na(args[0]):
                return round(float(args[0]), int(args[1]))
            return _NA
        if name in ('MathSin', 'sin'):
            return sin(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if name in ('MathCos', 'cos'):
            return cos(<double>args[0]) if args and not _is_na(args[0]) else _NA
        if name in ('MathTan', 'tan'):
            return tan(<double>args[0]) if args and not _is_na(args[0]) else _NA

        # ---- String functions ----
        if name == 'StringLen':
            return len(args[0]) if args and isinstance(args[0], str) else 0
        if name == 'StringFind':
            if len(args) >= 2 and isinstance(args[0], str):
                start = int(args[2]) if len(args) > 2 else 0
                return args[0].find(args[1], start)
            return -1
        if name == 'StringSubstr':
            if len(args) >= 2 and isinstance(args[0], str):
                start = int(args[1])
                count = int(args[2]) if len(args) > 2 else len(args[0])
                return args[0][start:start+count]
            return ''
        if name == 'IntegerToString':
            return str(int(args[0])) if args and not _is_na(args[0]) else ''
        if name == 'DoubleToString':
            if args and not _is_na(args[0]):
                digits = int(args[1]) if len(args) > 1 else 8
                return f'{float(args[0]):.{digits}f}'
            return ''
        if name == 'StringConcatenate':
            return ''.join(str(a) for a in args)
        if name == 'StringFormat':
            if args:
                fmt = str(args[0])
                try:
                    return fmt % tuple(args[1:]) if len(args) > 1 else fmt
                except (TypeError, ValueError):
                    return fmt
            return ''

        # ---- Output (no-ops) ----
        if name in ('Print', 'Comment', 'Alert', 'PrintFormat', 'printf'):
            return None

        # ---- Bars/time ----
        if name == 'Bars' or name == 'iBars':
            return self.bar_index + 1 if self.bars is not None else 0

        if name == 'TimeCurrent':
            if self.bar_index < len(self._timestamps):
                return self._timestamps[self.bar_index]
            return 0

        if name == 'TimeLocal':
            if self.bar_index < len(self._timestamps):
                return self._timestamps[self.bar_index]
            return 0

        # ---- Indicator release ----
        if name == 'IndicatorRelease':
            handle = int(args[0]) if args else 0
            self._indicator_handles.pop(handle, None)
            return True

        # ---- No-ops ----
        if name in ('ObjectCreate', 'ObjectDelete', 'ObjectSetInteger', 'ObjectSetDouble',
                     'ObjectSetString', 'ObjectFind', 'ChartRedraw', 'ChartSetInteger',
                     'ChartIndicatorAdd', 'EventSetTimer', 'EventKillTimer',
                     'GlobalVariableSet', 'GlobalVariableGet', 'GlobalVariableCheck',
                     'Sleep', 'GetLastError', 'ResetLastError',
                     'MarketInfo', 'RefreshRates',
                     'StringToTime', 'TimeToString', 'TimeToStruct',
                     'MathIsValidNumber', 'MathRand', 'MathSrand'):
            return 0

        # ---- __array_set__ ----
        if name == '__array_set__':
            return self._call_array_set(node, args)

        # ---- User-defined functions ----
        if name in self.user_functions:
            return self._call_user_func(name, args)

        return _NA

    # ---- CTrade call helpers ------------------------------------------------

    cdef object _call_buy(self, list args):
        cdef double lots = args[0] if args else 1.0
        if _is_na(lots) or lots <= 0:
            lots = 1.0
        cdef double units = lots * self._contract_size
        comment = args[5] if len(args) > 5 else ''
        sl = args[3] if len(args) > 3 and not _is_na(args[3]) and args[3] != 0 else None
        tp = args[4] if len(args) > 4 and not _is_na(args[4]) and args[4] != 0 else None
        self.signals.append(Signal('entry', 'long', units, str(comment)))
        if sl or tp:
            self.signals.append(Signal('exit', 'long', comment=str(comment), stop=sl, limit=tp, from_entry=str(comment)))
        return True

    cdef object _call_sell(self, list args):
        cdef double lots = args[0] if args else 1.0
        if _is_na(lots) or lots <= 0:
            lots = 1.0
        cdef double units = lots * self._contract_size
        comment = args[5] if len(args) > 5 else ''
        sl = args[3] if len(args) > 3 and not _is_na(args[3]) and args[3] != 0 else None
        tp = args[4] if len(args) > 4 and not _is_na(args[4]) and args[4] != 0 else None
        self.signals.append(Signal('entry', 'short', units, str(comment)))
        if sl or tp:
            self.signals.append(Signal('exit', 'short', comment=str(comment), stop=sl, limit=tp, from_entry=str(comment)))
        return True

    cdef object _call_position_open(self, list args):
        if len(args) >= 3:
            order_type = args[1]
            vol = args[2]
            sl = args[4] if len(args) > 4 and not _is_na(args[4]) and args[4] != 0 else None
            tp = args[5] if len(args) > 5 and not _is_na(args[5]) and args[5] != 0 else None
            comment = args[6] if len(args) > 6 else ''
            direction = 'long' if order_type == 0 else 'short'
            self.signals.append(Signal('entry', direction, vol, str(comment)))
            if sl or tp:
                self.signals.append(Signal('exit', direction, comment=str(comment), stop=sl, limit=tp, from_entry=str(comment)))
        return True

    cdef object _call_position_get_double(self, list args):
        prop = args[0] if args else 0
        if prop == 100: return self._position_price_open
        if prop == 101: return self._position_sl
        if prop == 102: return self._position_tp
        if prop == 103: return self._position_profit
        if prop == 104: return self._position_volume
        if prop == 105: return self.variables.get('close', 0)
        return 0.0

    cdef object _call_symbol_info_double(self, list args):
        prop = args[1] if len(args) > 1 else 0
        if prop == 200: return self.variables.get('Bid', 0)
        if prop == 201: return self.variables.get('Ask', 0)
        if prop == 202: return self.variables.get('_Point', 0.00001)
        if prop == 205:
            if self._is_jpy_pair:
                price = self.variables.get('close', 100)
                return self._contract_size * 0.001 / price if price > 0 else 1.0
            return 10.0
        if prop == 206: return self._contract_size
        if prop == 207: return 0.01
        if prop == 208: return 100.0
        if prop == 209: return 0.01
        if prop == 210:
            return 0.001 if self._is_jpy_pair else 0.00001
        return 0.0

    cdef object _call_symbol_info_string(self, list args):
        prop = args[1] if len(args) > 1 else 0
        if prop == 300:
            return 'JPY' if self._is_jpy_pair else 'USD'
        if prop == 301: return 'NZD'
        return ''

    cdef object _call_period_seconds(self, list args):
        cdef int tf = int(args[0]) if args else 0
        if tf == 0: return 300
        if tf == 1: return 60
        if tf == 5: return 300
        if tf == 15: return 900
        if tf == 30: return 1800
        if tf == 60: return 3600
        if tf == 240: return 14400
        if tf == 1440: return 86400
        if tf == 10080: return 604800
        if tf == 43200: return 2592000
        return 300

    cdef object _call_time_to_struct(self, object node, list args):
        cdef int timestamp = int(args[0]) if args and not _is_na(args[0]) else 0
        if timestamp > 0:
            import datetime as _dt
            try:
                t = _dt.datetime.utcfromtimestamp(timestamp)
                if len(node.args) >= 2:
                    struct_name = self._node_name(node.args[1])
                    if struct_name:
                        self.variables[f'{struct_name}.hour'] = t.hour
                        self.variables[f'{struct_name}.min'] = t.minute
                        self.variables[f'{struct_name}.sec'] = t.second
                        self.variables[f'{struct_name}.day'] = t.day
                        self.variables[f'{struct_name}.mon'] = t.month
                        self.variables[f'{struct_name}.year'] = t.year
                        self.variables[f'{struct_name}.day_of_week'] = t.weekday()
                        self.variables[f'{struct_name}.day_of_year'] = t.timetuple().tm_yday
            except Exception:
                pass
        return True

    # ---- iClose/iOpen/iHigh/iLow/iVolume -----------------------------------

    @cython.wraparound(True)
    cdef object _call_i_series(self, str name, list args):
        symbol = str(args[0]) if args else 'SYMBOL'
        cdef int tf = int(args[1]) if len(args) > 1 else 0
        cdef int shift = int(args[2]) if len(args) > 2 else 0
        cdef int base_tf = 5
        cdef int ratio

        cdef str col
        if name == 'iClose': col = 'close'
        elif name == 'iOpen': col = 'open'
        elif name == 'iHigh': col = 'high'
        elif name == 'iLow': col = 'low'
        else: col = 'volume'

        # Route to SMT data if symbol differs
        cdef object series
        if symbol != 'SYMBOL' and symbol != self.variables.get('_Symbol', 'SYMBOL') and self._smt_series:
            series = self._smt_series.get(col)
        else:
            series = self.series_data.get(col)
        if series is None:
            return _NA

        # Multi-timeframe simulation
        if tf > 0:
            ratio = max(1, tf // base_tf)
            if ratio > 1 and name == 'iHigh':
                start = shift * ratio
                best = _NA
                for k in range(ratio):
                    v = (<FastSeries>series).get(start + k)
                    if not _is_na(v) and (_is_na(best) or v > best):
                        best = v
                return best
            elif ratio > 1 and name == 'iLow':
                start = shift * ratio
                best = _NA
                for k in range(ratio):
                    v = (<FastSeries>series).get(start + k)
                    if not _is_na(v) and (_is_na(best) or v < best):
                        best = v
                return best
            elif ratio > 1:
                if name == 'iOpen':
                    return (<FastSeries>series).get(shift * ratio + ratio - 1)
                else:
                    return (<FastSeries>series).get(shift * ratio)
        return (<FastSeries>series).get(shift)

    @cython.wraparound(True)
    cdef object _call_itime(self, list args):
        cdef int tf = int(args[1]) if len(args) > 1 else 0
        cdef int shift = int(args[2]) if len(args) > 2 else 0
        cdef int effective_bar = max(0, self.bar_index - shift)
        cdef int timestamp
        if effective_bar < len(self._timestamps):
            timestamp = self._timestamps[effective_bar]
        elif self._timestamps:
            timestamp = self._timestamps[-1]
        else:
            timestamp = 0
        if tf > 0:
            htf_seconds = tf * 60
            timestamp = (timestamp // htf_seconds) * htf_seconds
        return timestamp

    # ---- Array function helpers ---------------------------------------------

    cdef object _call_array_resize(self, list args):
        arr = args[0] if args else []
        cdef int new_size = int(args[1]) if len(args) > 1 else 0
        if isinstance(arr, list):
            while len(arr) < new_size: arr.append(0.0)
            while len(arr) > new_size: arr.pop()
        return len(arr) if isinstance(arr, list) else 0

    cdef object _call_array_copy(self, list args):
        dst = args[0] if args else []
        src = args[1] if len(args) > 1 else []
        cdef int dst_start = int(args[2]) if len(args) > 2 else 0
        cdef int src_start = int(args[3]) if len(args) > 3 else 0
        cdef int count = int(args[4]) if len(args) > 4 else len(src) - src_start
        if isinstance(dst, list) and isinstance(src, list):
            while len(dst) < dst_start + count:
                dst.append(0.0)
            for i in range(count):
                if src_start + i < len(src):
                    dst[dst_start + i] = src[src_start + i]
        return count

    @cython.wraparound(True)
    cdef object _call_array_set(self, object node, list args):
        arr_name = self._node_name(node.args[0]) if len(node.args) > 0 else None
        arr = args[0] if args else []
        cdef int idx = int(args[1]) if len(args) > 1 else 0
        val = args[2] if len(args) > 2 else _NA
        if isinstance(arr, list):
            is_series = self._array_series_flags.get(id(arr), False)
            if is_series:
                actual_idx = len(arr) - 1 - idx
            else:
                actual_idx = idx
            if 0 <= actual_idx < len(arr):
                arr[actual_idx] = val
        return None

    # ---- User-defined function calls ----------------------------------------

    cdef object _call_user_func(self, str name, list args):
        fdef = self.user_functions[name]
        cdef dict saved = {}
        cdef int i
        for i, (pname, pdefault) in enumerate(fdef.params):
            saved[pname] = self.variables.get(pname)
            if i < len(args):
                self.variables[pname] = args[i]
            elif pdefault is not None:
                self.variables[pname] = self._eval(pdefault)
            else:
                self.variables[pname] = _NA
        cdef object result = _NA
        try:
            for stmt in fdef.body:
                self._exec(stmt)
        except FuncReturn as ret:
            result = ret.value
        for pname, _ in fdef.params:
            if saved[pname] is not None:
                self.variables[pname] = saved[pname]
            else:
                self.variables.pop(pname, None)
        return result

    # ---- Indicator Support --------------------------------------------------

    cdef object _create_indicator(self, str ind_type, list args):
        cdef int handle = self._next_handle
        self._next_handle += 1
        self._indicator_handles[handle] = {
            'type': ind_type,
            'args': args,
            'cache': {},
        }
        return handle

    @cython.wraparound(True)
    cdef object _copy_buffer(self, list args):
        if len(args) < 5:
            return -1
        cdef int handle = int(args[0])
        cdef int buffer_idx = int(args[1])
        cdef int start_pos = int(args[2])
        cdef int count = int(args[3])
        arr = args[4]
        if not isinstance(arr, list):
            return -1

        ind = self._indicator_handles.get(handle)
        if ind is None:
            return -1

        values = []
        cdef int shift
        for shift in range(start_pos, start_pos + count):
            val = self._compute_indicator(ind, buffer_idx, shift)
            values.append(val)

        is_series = self._array_series_flags.get(id(arr), False)
        while len(arr) < count:
            arr.append(0.0)

        cdef int i
        if is_series:
            for i in range(len(values)):
                arr[i] = values[i] if not _is_na(values[i]) else 0.0
        else:
            for i in range(len(values)):
                arr[i] = values[i] if not _is_na(values[i]) else 0.0

        return count

    cdef object _compute_indicator(self, object ind, int buffer_idx, int shift):
        cdef str ind_type = ind['type']
        ind_args = ind['args']

        if ind_type == 'MA':
            period = int(ind_args[2]) if len(ind_args) > 2 else 14
            method = int(ind_args[4]) if len(ind_args) > 4 else 0
            price_type = int(ind_args[5]) if len(ind_args) > 5 else 0
            source_name = self._price_type_to_series(price_type)
            if method == 0:
                return self._calc_sma(source_name, period, shift)
            elif method == 1:
                return self._calc_ema(source_name, period, shift)
            elif method == 2:
                return self._calc_sma(source_name, period, shift)
            elif method == 3:
                return self._calc_wma(source_name, period, shift)
            return _NA

        if ind_type == 'RSI':
            period = int(ind_args[2]) if len(ind_args) > 2 else 14
            price_type = int(ind_args[3]) if len(ind_args) > 3 else 0
            source_name = self._price_type_to_series(price_type)
            return self._calc_rsi(source_name, period, shift)

        if ind_type == 'MACD':
            fast = int(ind_args[2]) if len(ind_args) > 2 else 12
            slow = int(ind_args[3]) if len(ind_args) > 3 else 26
            signal = int(ind_args[4]) if len(ind_args) > 4 else 9
            price_type = int(ind_args[5]) if len(ind_args) > 5 else 0
            source_name = self._price_type_to_series(price_type)
            if buffer_idx == 0:
                fast_ema = self._calc_ema(source_name, fast, shift)
                slow_ema = self._calc_ema(source_name, slow, shift)
                if _is_na(fast_ema) or _is_na(slow_ema):
                    return _NA
                return <double>fast_ema - <double>slow_ema
            return _NA

        if ind_type == 'ATR':
            period = int(ind_args[2]) if len(ind_args) > 2 else 14
            return self._calc_atr(period, shift)

        if ind_type == 'BB':
            period = int(ind_args[2]) if len(ind_args) > 2 else 20
            deviation = float(ind_args[4]) if len(ind_args) > 4 else 2.0
            price_type = int(ind_args[5]) if len(ind_args) > 5 else 0
            source_name = self._price_type_to_series(price_type)
            basis = self._calc_sma(source_name, period, shift)
            if _is_na(basis):
                return _NA
            values = self._get_history(source_name, period, shift)
            valid = [v for v in values if v is not None]
            if len(valid) < period:
                return _NA
            std_val = sqrt(sum((<double>v - <double>basis) ** 2 for v in valid) / len(valid))
            if buffer_idx == 0: return basis
            if buffer_idx == 1: return <double>basis + deviation * std_val
            if buffer_idx == 2: return <double>basis - deviation * std_val
            return _NA

        if ind_type == 'STOCH':
            k_period = int(ind_args[2]) if len(ind_args) > 2 else 14
            highs = self._get_history('high', k_period, shift)
            lows = self._get_history('low', k_period, shift)
            vh = [v for v in highs if v is not None]
            vl = [v for v in lows if v is not None]
            if len(vh) < k_period or len(vl) < k_period:
                return _NA
            highest = max(vh); lowest = min(vl)
            close_val = (<FastSeries>self.series_data['close']).get(shift)
            if _is_na(close_val):
                return _NA
            if highest == lowest:
                return 50.0
            return 100.0 * (<double>close_val - lowest) / (highest - lowest)

        return _NA

    cdef str _price_type_to_series(self, int price_type):
        if price_type == 0: return 'close'
        if price_type == 1: return 'open'
        if price_type == 2: return 'high'
        if price_type == 3: return 'low'
        return 'close'

    # ---- History and indicator computation ----------------------------------

    cdef list _get_history(self, str source_name, int length, int extra_shift=0):
        series = self.series_data.get(source_name)
        if series is None or len((<FastSeries>series).data) == 0:
            return []
        cdef list values = []
        cdef int i
        cdef object v
        for i in range(length):
            v = (<FastSeries>series).get(i + extra_shift)
            values.append(None if _is_na(v) else float(v))
        values.reverse()
        return values

    cdef object _calc_sma(self, str source_name, int period, int shift=0):
        cdef list values = self._get_history(source_name, period, shift)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < period:
            return _NA
        cdef double total = 0.0
        for v in valid:
            total += <double>v
        return total / period

    cdef object _calc_ema(self, str source_name, int period, int shift=0):
        cdef str cache_key = f'_ema_{source_name}_{period}'
        series = self.series_data.get(source_name)
        if series is None:
            return _NA
        current_val = (<FastSeries>series).get(shift)
        if _is_na(current_val):
            return _NA

        if shift > 0:
            values = self._get_history(source_name, period + shift, 0)
            valid = [v for v in values if v is not None]
            if len(valid) < period:
                return _NA
            return sum(valid[:period]) / period

        prev_ema = self._ema_cache.get(cache_key, _NA)
        cdef double ema
        if _is_na(prev_ema):
            values = self._get_history(source_name, period, shift)
            valid = [v for v in values if v is not None]
            if len(valid) < period:
                return _NA
            ema = sum(valid) / len(valid)
        else:
            k = 2.0 / (period + 1)
            ema = <double>current_val * k + <double>prev_ema * (1.0 - k)
        self._ema_cache[cache_key] = ema
        return ema

    cdef object _calc_wma(self, str source_name, int period, int shift=0):
        cdef list values = self._get_history(source_name, period, shift)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < period:
            return _NA
        cdef double ws = <double>(period * (period + 1)) / 2.0
        cdef double total = 0.0
        cdef int i
        for i in range(len(valid)):
            total += <double>valid[i] * (i + 1)
        return total / ws

    cdef object _calc_rsi(self, str source_name, int period, int shift=0):
        cdef list values = self._get_history(source_name, period + 1, shift)
        cdef list valid = [v for v in values if v is not None]
        if len(valid) < period + 1:
            return _NA
        cdef list gains = [max(valid[i] - valid[i-1], 0) for i in range(1, len(valid))]
        cdef list losses = [max(valid[i-1] - valid[i], 0) for i in range(1, len(valid))]
        cdef double ag = sum(gains) / len(gains)
        cdef double al = sum(losses) / len(losses)
        if al == 0.0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    cdef object _calc_atr(self, int period, int shift=0):
        cdef list tr_values = []
        cdef int i
        cdef double h_val, l_val, c_p, tr
        for i in range(period):
            hv = (<FastSeries>self.series_data['high']).get(i + shift)
            lv = (<FastSeries>self.series_data['low']).get(i + shift)
            cv = (<FastSeries>self.series_data['close']).get(i + shift + 1)
            if _is_na(hv) or _is_na(lv):
                return _NA
            h_val = <double>hv
            l_val = <double>lv
            if not _is_na(cv):
                c_p = <double>cv
                tr = max(h_val - l_val, fabs(h_val - c_p), fabs(l_val - c_p))
            else:
                tr = h_val - l_val
            tr_values.append(tr)
        if not tr_values:
            return _NA
        return sum(tr_values) / len(tr_values)
