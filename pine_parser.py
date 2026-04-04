"""
Pine Script Tokenizer, Parser, and Interpreter.
Supports a practical subset of Pine Script v5 for backtesting strategies.
"""

import re
import math
import numpy as np
from enum import Enum, auto
from typing import Any

# ─── Token types ───────────────────────────────────────────────────────────────

class TT(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    BOOL = auto()
    NA = auto()
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto(); PERCENT = auto()
    EQ = auto(); NEQ = auto(); LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    ASSIGN = auto(); COLONEQ = auto()
    AND = auto(); OR = auto(); NOT = auto()
    QUESTION = auto(); COLON = auto()
    DOT = auto(); COMMA = auto()
    LPAREN = auto(); RPAREN = auto()
    LBRACKET = auto(); RBRACKET = auto()
    IF = auto(); ELSE = auto(); FOR = auto(); WHILE = auto()
    VAR = auto(); VARIP = auto()
    TRUE = auto(); FALSE = auto()
    CONTINUE = auto(); BREAK = auto(); SWITCH = auto()
    NEWLINE = auto(); INDENT = auto(); DEDENT = auto(); EOF = auto()
    PLUSEQ = auto(); MINUSEQ = auto(); STAREQ = auto(); SLASHEQ = auto()
    ARROW = auto()  # =>
    HASH = auto()   # for color literals #rrggbb


class Token:
    __slots__ = ('type', 'value', 'line')
    def __init__(self, type: TT, value: Any, line: int = 0):
        self.type = type; self.value = value; self.line = line
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


KEYWORDS = {
    'if': TT.IF, 'else': TT.ELSE, 'for': TT.FOR, 'while': TT.WHILE,
    'var': TT.VAR, 'varip': TT.VARIP,
    'true': TT.TRUE, 'false': TT.FALSE,
    'and': TT.AND, 'or': TT.OR, 'not': TT.NOT,
    'na': TT.NA, 'continue': TT.CONTINUE, 'break': TT.BREAK, 'switch': TT.SWITCH,
}

TWO_CHAR_OPS = {
    '==': TT.EQ, '!=': TT.NEQ, '<=': TT.LTE, '>=': TT.GTE,
    ':=': TT.COLONEQ, '+=': TT.PLUSEQ, '-=': TT.MINUSEQ,
    '*=': TT.STAREQ, '/=': TT.SLASHEQ, '=>': TT.ARROW,
}

ONE_CHAR_OPS = {
    '+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR, '/': TT.SLASH, '%': TT.PERCENT,
    '<': TT.LT, '>': TT.GT, '=': TT.ASSIGN,
    '?': TT.QUESTION, ':': TT.COLON, '.': TT.DOT, ',': TT.COMMA,
    '(': TT.LPAREN, ')': TT.RPAREN, '[': TT.LBRACKET, ']': TT.RBRACKET,
    '#': TT.HASH,
}


def tokenize(source: str) -> list[Token]:
    lines = source.split('\n')
    cleaned = []
    in_block = False
    for line in lines:
        while '/*' in line or in_block:
            if in_block:
                idx = line.find('*/')
                if idx == -1:
                    line = ''; break
                else:
                    line = line[idx+2:]; in_block = False
            else:
                idx = line.find('/*')
                end = line.find('*/', idx+2)
                if end == -1:
                    line = line[:idx]; in_block = True
                else:
                    line = line[:idx] + line[end+2:]
        ci = line.find('//')
        if ci != -1:
            line = line[:ci]
        cleaned.append(line)

    tokens = []
    indent_stack = [0]
    line_num = 0

    for raw_line in cleaned:
        line_num += 1
        stripped = raw_line.rstrip()
        if not stripped:
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token(TT.INDENT, indent, line_num))
        else:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TT.DEDENT, indent, line_num))

        i = 0
        content = stripped
        while i < len(content):
            ch = content[i]
            if ch in ' \t':
                i += 1; continue

            # Color literal #rrggbb
            if ch == '#' and i+1 < len(content) and content[i+1] in '0123456789abcdefABCDEF':
                j = i + 1
                while j < len(content) and content[j] in '0123456789abcdefABCDEF':
                    j += 1
                tokens.append(Token(TT.STRING, content[i:j], line_num))
                i = j; continue

            if ch.isdigit() or (ch == '.' and i+1 < len(content) and content[i+1].isdigit()):
                j = i; has_dot = False
                while j < len(content) and (content[j].isdigit() or (content[j] == '.' and not has_dot)):
                    if content[j] == '.': has_dot = True
                    j += 1
                tokens.append(Token(TT.NUMBER, float(content[i:j]), line_num))
                i = j; continue

            if ch in ('"', "'"):
                j = i + 1
                while j < len(content) and content[j] != ch:
                    if content[j] == '\\': j += 1
                    j += 1
                tokens.append(Token(TT.STRING, content[i+1:j], line_num))
                i = j + 1; continue

            if ch.isalpha() or ch == '_':
                j = i
                while j < len(content) and (content[j].isalnum() or content[j] == '_'):
                    j += 1
                word = content[i:j]
                if word in KEYWORDS:
                    tokens.append(Token(KEYWORDS[word], word, line_num))
                else:
                    tokens.append(Token(TT.IDENT, word, line_num))
                i = j; continue

            if i + 1 < len(content):
                two = content[i:i+2]
                if two in TWO_CHAR_OPS:
                    tokens.append(Token(TWO_CHAR_OPS[two], two, line_num))
                    i += 2; continue

            if ch in ONE_CHAR_OPS:
                tokens.append(Token(ONE_CHAR_OPS[ch], ch, line_num))
                i += 1; continue

            i += 1

        tokens.append(Token(TT.NEWLINE, '\\n', line_num))

    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TT.DEDENT, 0, line_num))

    tokens.append(Token(TT.EOF, None, line_num))
    return tokens


# ─── AST Nodes ─────────────────────────────────────────────────────────────────
# _tag integers enable fast dispatch in Cython (avoids isinstance chains)

_OP_ID = {'+': 0, '-': 1, '*': 2, '/': 3, '%': 4, '<': 5, '>': 6, '<=': 7, '>=': 8, '==': 9, '!=': 10, 'and': 11, 'or': 12}

class ASTNode:
    _tag = -1
class NumberLit(ASTNode):
    _tag = 0
    def __init__(self, value): self.value = value
class StringLit(ASTNode):
    _tag = 1
    def __init__(self, value): self.value = value
class BoolLit(ASTNode):
    _tag = 2
    def __init__(self, value): self.value = value
class NALit(ASTNode):
    _tag = 3
class Identifier(ASTNode):
    _tag = 4
    def __init__(self, name): self.name = name
class ArrayLit(ASTNode):
    _tag = 5
    def __init__(self, elements): self.elements = elements
class BinOp(ASTNode):
    _tag = 6
    def __init__(self, op, left, right): self.op = op; self.left = left; self.right = right; self.op_id = _OP_ID.get(op, -1)
class UnaryOp(ASTNode):
    _tag = 7
    def __init__(self, op, operand): self.op = op; self.operand = operand
class HistoryRef(ASTNode):
    _tag = 8
    def __init__(self, expr, offset): self.expr = expr; self.offset = offset
class FuncCall(ASTNode):
    _tag = 9
    def __init__(self, name, args, kwargs): self.name = name; self.args = args; self.kwargs = kwargs
class DotAccess(ASTNode):
    _tag = 10
    def __init__(self, obj, attr): self.obj = obj; self.attr = attr
class Ternary(ASTNode):
    _tag = 11
    def __init__(self, cond, true_val, false_val): self.cond = cond; self.true_val = true_val; self.false_val = false_val
class Assignment(ASTNode):
    _tag = 12
    def __init__(self, name, value, op='=', is_var=False, type_hint=None):
        self.name = name; self.value = value; self.op = op; self.is_var = is_var; self.type_hint = type_hint; self.op_id = _OP_ID.get(op, -1)
class IfStmt(ASTNode):
    _tag = 13
    def __init__(self, cond, body, else_body=None): self.cond = cond; self.body = body; self.else_body = else_body
class ForStmt(ASTNode):
    _tag = 14
    def __init__(self, var, start, end, step, body): self.var = var; self.start = start; self.end = end; self.step = step; self.body = body
class ExprStmt(ASTNode):
    _tag = 15
    def __init__(self, expr): self.expr = expr
class ContinueStmt(ASTNode):
    _tag = 16
class BreakStmt(ASTNode):
    _tag = 17
class SwitchExpr(ASTNode):
    _tag = 18
    def __init__(self, expr, cases, default):
        self.expr = expr; self.cases = cases; self.default = default
class FuncDef(ASTNode):
    _tag = 19
    def __init__(self, name, params, body):
        self.name = name; self.params = params; self.body = body
class Program(ASTNode):
    _tag = 20
    def __init__(self, stmts): self.stmts = stmts


# ─── Parser ────────────────────────────────────────────────────────────────────

# Pine Script type keywords that can appear in declarations: float x = ..., int[] arr = ...
TYPE_KEYWORDS = {'float', 'int', 'bool', 'string', 'color', 'line', 'label', 'box', 'table', 'series'}


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self._paren_depth = 0

    def peek(self) -> Token:
        if self._paren_depth > 0:
            while self.pos < len(self.tokens) and self.tokens[self.pos].type in (TT.NEWLINE, TT.INDENT, TT.DEDENT):
                self.pos += 1
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TT.EOF, None)

    def advance(self) -> Token:
        t = self.peek()  # skip whitespace if inside parens
        self.pos += 1
        if t.type == TT.LPAREN: self._paren_depth += 1
        elif t.type == TT.RPAREN: self._paren_depth = max(0, self._paren_depth - 1)
        return t

    def expect(self, tt: TT) -> Token:
        t = self.advance()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt}, got {t.type} ({t.value!r}) at line {t.line}")
        return t

    def skip_newlines(self):
        while self.peek().type == TT.NEWLINE:
            self.advance()

    def skip_whitespace(self):
        while self.peek().type in (TT.NEWLINE, TT.INDENT, TT.DEDENT):
            self.advance()

    def _skip_inside_parens(self):
        while self.peek().type in (TT.NEWLINE, TT.INDENT, TT.DEDENT):
            self.advance()

    def parse(self) -> Program:
        stmts = []
        self.skip_whitespace()
        while self.peek().type != TT.EOF:
            stmt = self.parse_stmt()
            if stmt: stmts.append(stmt)
            self.skip_whitespace()
        return Program(stmts)

    def parse_stmt(self):
        while self.peek().type in (TT.NEWLINE, TT.DEDENT):
            self.advance()
        tok = self.peek()
        if tok.type == TT.IF: return self.parse_if()
        if tok.type == TT.FOR: return self.parse_for()
        if tok.type == TT.SWITCH: return ExprStmt(self.parse_switch())
        if tok.type == TT.CONTINUE: self.advance(); return ContinueStmt()
        if tok.type == TT.BREAK: self.advance(); return BreakStmt()
        if tok.type in (TT.VAR, TT.VARIP): return self.parse_var_decl()
        # Type-annotated declaration: float x = ..., int[] arr = ...
        if tok.type == TT.IDENT and tok.value in TYPE_KEYWORDS:
            return self._try_typed_decl()
        return self.parse_assignment_or_expr()

    def _try_typed_decl(self):
        """Try to parse: type_name [identifier] = expr  or  type_name[] identifier = expr"""
        save = self.pos
        self.advance()  # type keyword
        # Skip [] if present
        if self.peek().type == TT.LBRACKET:
            self.advance()
            if self.peek().type == TT.RBRACKET:
                self.advance()
        # Must be an identifier followed by = or :=
        if self.peek().type == TT.IDENT:
            next_after = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else Token(TT.EOF, None)
            if next_after.type in (TT.ASSIGN, TT.COLONEQ):
                name = self.advance().value
                self.advance()  # = or :=
                value = self.parse_expr()
                return Assignment(name, value, '=')
        # Not a typed decl, restore and parse as expression
        self.pos = save
        return self.parse_assignment_or_expr()

    def parse_var_decl(self):
        self.advance()  # var/varip
        # Skip optional type annotation: var float x = ..., var float[] x = ...
        if self.peek().type == TT.IDENT and self.peek().value in TYPE_KEYWORDS:
            self.advance()
            if self.peek().type == TT.LBRACKET:
                self.advance()
                if self.peek().type == TT.RBRACKET:
                    self.advance()
        name_tok = self.expect(TT.IDENT)
        self.expect(TT.ASSIGN)
        value = self.parse_expr()
        return Assignment(name_tok.value, value, '=', is_var=True)

    def parse_assignment_or_expr(self):
        expr = self.parse_expr()
        tok = self.peek()
        # Function definition: myFunc(a, b) => body
        if tok.type == TT.ARROW and isinstance(expr, FuncCall):
            self.advance()  # =>
            name = expr.name
            params = [a.name if isinstance(a, Identifier) else str(a) for a in expr.args]
            self.skip_newlines()
            if self.peek().type == TT.INDENT:
                body = self.parse_block()
            else:
                body = [ExprStmt(self.parse_expr())]
            return FuncDef(name, params, body)
        if tok.type in (TT.ASSIGN, TT.COLONEQ, TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ):
            if isinstance(expr, Identifier):
                op = self.advance()
                value = self.parse_expr()
                return Assignment(expr.name, value, op.value)
            else:
                raise SyntaxError(f"Invalid assignment target at line {tok.line}")
        return ExprStmt(expr)

    def parse_if(self):
        self.advance()  # if
        cond = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block()
        else_body = None
        self.skip_newlines()
        if self.peek().type == TT.ELSE:
            self.advance()
            self.skip_newlines()
            if self.peek().type == TT.IF:
                else_body = [self.parse_if()]
            else:
                else_body = self.parse_block()
        return IfStmt(cond, body, else_body)

    def parse_for(self):
        self.advance()  # for
        var_tok = self.expect(TT.IDENT)
        self.expect(TT.ASSIGN)
        start = self.parse_expr()
        to_tok = self.expect(TT.IDENT)
        if to_tok.value != 'to':
            raise SyntaxError(f"Expected 'to' in for loop, got '{to_tok.value}'")
        end = self.parse_expr()
        step = None
        if self.peek().type == TT.IDENT and self.peek().value == 'by':
            self.advance(); step = self.parse_expr()
        self.skip_newlines()
        body = self.parse_block()
        return ForStmt(var_tok.value, start, end, step, body)

    def parse_switch(self):
        self.advance()  # switch
        # Optional expression after switch (before newline)
        expr = None
        if self.peek().type not in (TT.NEWLINE, TT.INDENT):
            expr = self.parse_expr()
        self.skip_newlines()
        cases = []
        default = None
        if self.peek().type == TT.INDENT:
            self.advance()
            while self.peek().type not in (TT.DEDENT, TT.EOF):
                while self.peek().type == TT.NEWLINE: self.advance()
                if self.peek().type in (TT.DEDENT, TT.EOF): break
                # Default case: => value
                if self.peek().type == TT.ARROW:
                    self.advance()
                    self.skip_newlines()
                    if self.peek().type == TT.INDENT:
                        default = self.parse_block()
                    else:
                        default = [ExprStmt(self.parse_expr())]
                else:
                    cond = self.parse_expr()
                    self.expect(TT.ARROW)
                    self.skip_newlines()
                    if self.peek().type == TT.INDENT:
                        body = self.parse_block()
                    else:
                        body = [ExprStmt(self.parse_expr())]
                    cases.append((cond, body))
                while self.peek().type == TT.NEWLINE: self.advance()
            if self.peek().type == TT.DEDENT: self.advance()
        return SwitchExpr(expr, cases, default)

    def parse_block(self):
        stmts = []
        if self.peek().type == TT.INDENT:
            self.advance()
            while self.peek().type not in (TT.DEDENT, TT.EOF):
                while self.peek().type == TT.NEWLINE: self.advance()
                if self.peek().type in (TT.DEDENT, TT.EOF): break
                stmt = self._parse_block_stmt()
                if stmt: stmts.append(stmt)
            if self.peek().type == TT.DEDENT: self.advance()
        else:
            stmt = self._parse_block_stmt()
            if stmt: stmts.append(stmt)
        return stmts

    def _parse_block_stmt(self):
        while self.peek().type == TT.NEWLINE: self.advance()
        tok = self.peek()
        if tok.type == TT.IF: return self.parse_if()
        if tok.type == TT.FOR: return self.parse_for()
        if tok.type == TT.SWITCH: return ExprStmt(self.parse_switch())
        if tok.type == TT.CONTINUE: self.advance(); return ContinueStmt()
        if tok.type == TT.BREAK: self.advance(); return BreakStmt()
        if tok.type in (TT.VAR, TT.VARIP): return self.parse_var_decl()
        if tok.type == TT.IDENT and tok.value in TYPE_KEYWORDS:
            return self._try_typed_decl()
        return self.parse_assignment_or_expr()

    # ─── Expressions ───────────────────────────────────────────────────────

    def parse_expr(self):
        return self.parse_ternary()

    def _skip_continuation(self):
        """Skip newlines after operators for line continuation."""
        while self.pos < len(self.tokens) and self.tokens[self.pos].type in (TT.NEWLINE, TT.INDENT, TT.DEDENT):
            self.pos += 1

    def parse_ternary(self):
        expr = self.parse_or()
        # Check for ? on same line or continuation line
        save = self.pos
        self._skip_continuation()
        if self.peek().type == TT.QUESTION:
            self.advance(); self._skip_continuation()
            true_val = self.parse_or()
            self._skip_continuation(); self.expect(TT.COLON); self._skip_continuation()
            false_val = self.parse_ternary()
            return Ternary(expr, true_val, false_val)
        self.pos = save
        return expr

    def parse_or(self):
        left = self.parse_and()
        while self.peek().type == TT.OR:
            self.advance(); self._skip_continuation(); left = BinOp('or', left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek().type == TT.AND:
            self.advance(); self._skip_continuation(); left = BinOp('and', left, self.parse_not())
        return left

    def parse_not(self):
        if self.peek().type == TT.NOT:
            self.advance(); return UnaryOp('not', self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_addition()
        while self.peek().type in (TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE):
            op = self.advance(); self._skip_continuation(); left = BinOp(op.value, left, self.parse_addition())
        return left

    def parse_addition(self):
        left = self.parse_multiplication()
        while self.peek().type in (TT.PLUS, TT.MINUS):
            op = self.advance(); self._skip_continuation(); left = BinOp(op.value, left, self.parse_multiplication())
        return left

    def parse_multiplication(self):
        left = self.parse_unary()
        while self.peek().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance(); self._skip_continuation(); left = BinOp(op.value, left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.peek().type == TT.MINUS:
            self.advance(); return UnaryOp('-', self.parse_unary())
        if self.peek().type == TT.PLUS:
            self.advance(); return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if self.peek().type == TT.DOT:
                self.advance(); attr = self.expect(TT.IDENT); expr = DotAccess(expr, attr.value)
            elif self.peek().type == TT.LPAREN:
                expr = self.parse_call(expr)
            elif self.peek().type == TT.LBRACKET:
                self.advance(); index = self.parse_expr(); self.expect(TT.RBRACKET)
                expr = HistoryRef(expr, index)
            else:
                break
        return expr

    def parse_call(self, callee) -> FuncCall:
        self.advance()  # (
        args = []; kwargs = {}
        while self.peek().type != TT.RPAREN:
            self._skip_inside_parens()
            if self.peek().type == TT.RPAREN: break
            if (self.peek().type == TT.IDENT and
                self.pos + 1 < len(self.tokens) and
                self.tokens[self.pos + 1].type == TT.ASSIGN):
                name = self.advance().value; self.advance()
                val = self.parse_expr(); kwargs[name] = val
            else:
                args.append(self.parse_expr())
            self._skip_inside_parens()
            if self.peek().type == TT.COMMA: self.advance()
        self.expect(TT.RPAREN)
        name = self._node_to_name(callee)
        return FuncCall(name, args, kwargs)

    def _node_to_name(self, node) -> str:
        if isinstance(node, Identifier): return node.name
        if isinstance(node, DotAccess): return self._node_to_name(node.obj) + '.' + node.attr
        return '<expr>'

    def parse_primary(self):
        tok = self.peek()
        if tok.type == TT.NUMBER: self.advance(); return NumberLit(tok.value)
        if tok.type == TT.STRING: self.advance(); return StringLit(tok.value)
        if tok.type in (TT.TRUE, TT.FALSE): self.advance(); return BoolLit(tok.value == 'true')
        if tok.type == TT.NA:
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TT.LPAREN:
                self.advance(); return Identifier('na')
            self.advance(); return NALit()
        if tok.type == TT.IF: return self.parse_if()
        if tok.type == TT.SWITCH: return self.parse_switch()
        if tok.type == TT.IDENT: self.advance(); return Identifier(tok.value)
        if tok.type == TT.LPAREN:
            self.advance(); expr = self.parse_expr(); self.expect(TT.RPAREN); return expr
        # Array literal [a, b, c]
        if tok.type == TT.LBRACKET:
            self.advance()
            elements = []
            while self.peek().type != TT.RBRACKET:
                self._skip_inside_parens()
                if self.peek().type == TT.RBRACKET: break
                elements.append(self.parse_expr())
                self._skip_inside_parens()
                if self.peek().type == TT.COMMA: self.advance()
            self.expect(TT.RBRACKET)
            return ArrayLit(elements)
        raise SyntaxError(f"Unexpected token {tok.type} ({tok.value!r}) at line {tok.line}")


# ─── Interpreter ───────────────────────────────────────────────────────────────

class PineNA:
    _instance = None
    def __new__(cls):
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self): return 'na'
    def __bool__(self): return False

NA = PineNA()

def is_na(v):
    return v is NA or v is None or (isinstance(v, float) and math.isnan(v))

class Series:
    def __init__(self, max_lookback=500):
        self.data = []
        self._start = 0
        self._max = max_lookback
    def append(self, val):
        self.data.append(val)
        if len(self.data) > self._max:
            self.data.pop(0)
            self._start += 1
    def get(self, offset=0):
        idx = len(self.data) - 1 - offset
        if idx < 0: return NA
        return self.data[idx]
    def __len__(self):
        return self._start + len(self.data)

class Signal:
    def __init__(self, action, direction, qty=None, comment='', limit=None, stop=None, from_entry='', order_type='market', entry_price=None):
        self.action = action; self.direction = direction; self.qty = qty
        self.comment = comment; self.limit = limit; self.stop = stop
        self.from_entry = from_entry; self.order_type = order_type; self.entry_price = entry_price

class LoopBreak(Exception): pass
class LoopContinue(Exception): pass


class PineInterpreter:
    def __init__(self, ast: Program):
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
        self._series_vars = set()  # filled by setup() via AST pre-scan
        self._bar_cache = {}
        # Secondary (SMT) data
        self._smt_row_cache = None
        self._smt_series = {}

    def _collect_history_vars(self, stmts):
        """Walk AST once to find all variable names used in history references (x[n]).
        Only those variables need full series tracking."""
        names = set()
        def walk(node):
            if node is None: return
            if isinstance(node, HistoryRef):
                n = self._node_name(node.expr)
                if n: names.add(n)
                walk(node.offset)
            elif isinstance(node, BinOp): walk(node.left); walk(node.right)
            elif isinstance(node, UnaryOp): walk(node.operand)
            elif isinstance(node, Ternary): walk(node.cond); walk(node.true_val); walk(node.false_val)
            elif isinstance(node, FuncCall):
                for a in node.args: walk(a)
                for v in node.kwargs.values(): walk(v)
            elif isinstance(node, Assignment): walk(node.value)
            elif isinstance(node, IfStmt):
                walk(node.cond)
                for s in node.body: walk(s)
                if node.else_body:
                    for s in node.else_body: walk(s)
            elif isinstance(node, ForStmt):
                walk(node.start); walk(node.end)
                if node.step: walk(node.step)
                for s in node.body: walk(s)
            elif isinstance(node, ExprStmt): walk(node.expr)
            elif isinstance(node, SwitchExpr):
                if node.expr: walk(node.expr)
                for cond, body in node.cases:
                    walk(cond)
                    for s in body: walk(s)
                if node.default:
                    for s in node.default: walk(s)
            elif isinstance(node, FuncDef):
                for s in node.body: walk(s)
        for stmt in stmts:
            walk(stmt)
        # Always include OHLCV and common derived series
        names.update(['open', 'high', 'low', 'close', 'volume', 'hl2', 'hlc3', 'ohlc4'])
        return names

    def setup(self, bars):
        self.bars = bars
        self.bar_index = 0
        self._first_bar = True
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self.series_data[col] = Series()
        # Pre-scan AST: only track series for variables used in history refs
        self._series_vars = self._collect_history_vars(self.ast.stmts)
        # Precompute row data as dicts for fast access
        self._row_cache = []
        for i in range(len(bars)):
            r = bars.iloc[i]
            self._row_cache.append({
                'open': r.get('open', 0), 'high': r.get('high', 0),
                'low': r.get('low', 0), 'close': r.get('close', 0),
                'volume': r.get('volume', 0),
            })

        # Auto-detect instrument type from price levels for syminfo
        avg_price = bars['close'].iloc[:min(100, len(bars))].mean() if len(bars) > 0 else 1.0
        if avg_price > 10:  # JPY pair
            self.variables['_syminfo_currency'] = 'JPY'
            self.variables['_syminfo_pointvalue'] = 0.01 / avg_price  # ~0.000108 for NZDJPY@92
        else:
            self.variables['_syminfo_currency'] = 'USD'
            self.variables['_syminfo_pointvalue'] = 1.0

    def setup_secondary(self, bars):
        """Load a secondary (SMT) symbol's OHLCV data for request.security()."""
        self._smt_series = {}
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self._smt_series[col] = Series()
        self._smt_row_cache = []
        for i in range(len(bars)):
            r = bars.iloc[i]
            self._smt_row_cache.append({
                'open': r.get('open', 0), 'high': r.get('high', 0),
                'low': r.get('low', 0), 'close': r.get('close', 0),
                'volume': r.get('volume', 0),
            })

    def run_all_bars(self):
        self.all_signals = []
        for i in range(len(self.bars)):
            self.run_bar(i)
        return self.all_signals

    def run_bar(self, idx):
        self.bar_index = idx
        self._bar_cache = {}
        self.signals = []

        row = self._row_cache[idx]
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self.series_data[col].append(row[col])

        # Feed secondary (SMT) data if available
        if self._smt_row_cache and idx < len(self._smt_row_cache):
            smt_row = self._smt_row_cache[idx]
            for col in ('open', 'high', 'low', 'close', 'volume'):
                self._smt_series[col].append(smt_row[col])

        o = row['open']; h = row['high']; l = row['low']; c = row['close']
        v = self.variables
        v['bar_index'] = idx; v['close'] = c; v['open'] = o
        v['high'] = h; v['low'] = l; v['volume'] = row['volume']
        v['hl2'] = (h + l) / 2; v['hlc3'] = (h + l + c) / 3; v['ohlc4'] = (o + h + l + c) / 4

        for stmt in self.ast.stmts:
            self._exec(stmt)

        self.all_signals.extend([(idx, s) for s in self.signals])
        self._first_bar = False

    def _exec(self, node):
        if isinstance(node, Assignment):
            val = self._eval(node.value)
            if node.is_var:
                if node.name not in self.var_declared:
                    self.var_declared.add(node.name)
                    self.variables[node.name] = val
                # else keep existing value (var only inits on first bar)
                return
            if node.op in (':=', '='):
                self.variables[node.name] = val
            elif node.op == '+=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '+')
            elif node.op == '-=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '-')
            elif node.op == '*=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '*')
            elif node.op == '/=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '/')
            # Update series (only for vars that are history-referenced)
            if node.name in self.series_data:
                s = self.series_data[node.name]
                if len(s) <= self.bar_index:
                    s.append(self.variables[node.name])
                else:
                    rel = self.bar_index - s._start
                    if 0 <= rel < len(s.data):
                        s.data[rel] = self.variables[node.name]
            elif node.name in self._series_vars:
                s = Series()
                s._start = self.bar_index  # no NA padding, just record start
                s.append(self.variables.get(node.name, NA))
                self.series_data[node.name] = s

        elif isinstance(node, IfStmt):
            cond = self._eval(node.cond)
            if self._truthy(cond):
                for stmt in node.body: self._exec(stmt)
            elif node.else_body:
                for stmt in node.else_body: self._exec(stmt)

        elif isinstance(node, ForStmt):
            start = int(self._eval(node.start))
            end = int(self._eval(node.end))
            step = int(self._eval(node.step)) if node.step else (1 if end >= start else -1)
            if step == 0: step = 1
            if step > 0:
                rng = range(start, end + 1, step)
            else:
                rng = range(start, end - 1, step)
            for val in rng:
                self.variables[node.var] = val
                try:
                    for stmt in node.body: self._exec(stmt)
                except LoopContinue:
                    continue
                except LoopBreak:
                    break

        elif isinstance(node, ContinueStmt):
            raise LoopContinue()
        elif isinstance(node, BreakStmt):
            raise LoopBreak()
        elif isinstance(node, ExprStmt):
            self._eval(node.expr)
        elif isinstance(node, FuncDef):
            self.user_functions[node.name] = node

    def _eval(self, node) -> Any:
        if isinstance(node, NumberLit): return node.value
        if isinstance(node, StringLit): return node.value
        if isinstance(node, BoolLit): return node.value
        if isinstance(node, NALit): return NA
        if isinstance(node, ArrayLit): return [self._eval(e) for e in node.elements]

        if isinstance(node, Identifier):
            name = node.name
            if name in self.variables: return self.variables[name]
            if name in self.series_data: return self.series_data[name].get(0)
            if name == 'strategy': return '<strategy>'
            return NA

        if isinstance(node, BinOp): return self._eval_binop(node)

        if isinstance(node, UnaryOp):
            val = self._eval(node.operand)
            if node.op == '-': return -val if not is_na(val) else NA
            if node.op == 'not': return not self._truthy(val)
            return val

        if isinstance(node, Ternary):
            return self._eval(node.true_val) if self._truthy(self._eval(node.cond)) else self._eval(node.false_val)

        if isinstance(node, HistoryRef):
            offset = int(self._eval(node.offset))
            name = self._node_name(node.expr)
            if name and name in self.series_data:
                return self.series_data[name].get(offset)
            return NA

        if isinstance(node, SwitchExpr): return self._eval_switch(node)
        if isinstance(node, IfStmt): return self._eval_if_expr(node)
        if isinstance(node, FuncCall): return self._call(node)

        if isinstance(node, DotAccess):
            full = self._dot_name(node)
            if full: return self._resolve_constant(full)
            return NA

        return NA

    def _eval_switch(self, node):
        if node.expr is not None:
            val = self._eval(node.expr)
            for cond, body in node.cases:
                cv = self._eval(cond)
                if val == cv or (is_na(val) and is_na(cv)):
                    result = NA
                    for stmt in body:
                        if isinstance(stmt, ExprStmt): result = self._eval(stmt.expr)
                        else: self._exec(stmt)
                    return result
        else:
            for cond, body in node.cases:
                if self._truthy(self._eval(cond)):
                    result = NA
                    for stmt in body:
                        if isinstance(stmt, ExprStmt): result = self._eval(stmt.expr)
                        else: self._exec(stmt)
                    return result
        if node.default:
            result = NA
            for stmt in node.default:
                if isinstance(stmt, ExprStmt): result = self._eval(stmt.expr)
                else: self._exec(stmt)
            return result
        return NA

    def _eval_if_expr(self, node):
        cond = self._eval(node.cond)
        if self._truthy(cond):
            result = NA
            for stmt in node.body:
                if isinstance(stmt, ExprStmt): result = self._eval(stmt.expr)
                else: self._exec(stmt)
            return result
        elif node.else_body:
            result = NA
            for stmt in node.else_body:
                if isinstance(stmt, ExprStmt): result = self._eval(stmt.expr)
                else: self._exec(stmt)
            return result
        return NA

    def _dot_name(self, node):
        if isinstance(node, DotAccess):
            p = self._dot_name(node.obj)
            return p + '.' + node.attr if p else None
        if isinstance(node, Identifier): return node.name
        return None

    def _resolve_constant(self, name):
        constants = {
            'strategy.long': 'long', 'strategy.short': 'short',
            'strategy.fixed': 'fixed', 'strategy.cash': 'cash',
            'strategy.percent_of_equity': 'percent_of_equity',
            'strategy.commission.percent': 'percent',
            'strategy.commission.cash_per_contract': 'cash_per_contract',
            'currency.USD': 'USD', 'currency.EUR': 'EUR', 'currency.GBP': 'GBP',
            'currency.JPY': 'JPY', 'currency.AUD': 'AUD', 'currency.NZD': 'NZD',
            'math.pi': math.pi, 'math.e': math.e,
            'barstate.islast': self.bar_index == len(self.bars) - 1 if self.bars is not None else False,
            'barstate.isfirst': self.bar_index == 0,
            'barstate.isconfirmed': True,
            'syminfo.tickerid': 'SYMBOL', 'syminfo.ticker': 'SYMBOL',
            'syminfo.currency': self.variables.get('_syminfo_currency', 'USD'),
            'syminfo.pointvalue': self.variables.get('_syminfo_pointvalue', 1.0),
            'timeframe.period': '60',
            'barmerge.lookahead_on': True, 'barmerge.lookahead_off': False,
            'label.style_label_down': 'label_down', 'label.style_label_up': 'label_up',
            'label.style_label_left': 'label_left',
            'line.style_dashed': 'dashed', 'line.style_solid': 'solid',
            'plot.style_cross': 'cross', 'plot.style_line': 'line',
            'size.tiny': 'tiny', 'size.small': 'small', 'size.normal': 'normal',
            'position.top_right': 'top_right', 'position.top_left': 'top_left',
            'position.bottom_right': 'bottom_right', 'position.bottom_left': 'bottom_left',
            'format.mintick': '#.#####',
            'order.ascending': 'asc', 'order.descending': 'desc',
            'alert.freq_once_per_bar_close': 'once_per_bar_close',
            'strategy.position_size': self.variables.get('_position_size', 0),
            'strategy.equity': self.variables.get('_equity', 10000),
            'strategy.openprofit': self.variables.get('_open_profit', 0),
        }
        # strategy.opentrades.entry_price(0) is handled as a function call
        if name in constants: return constants[name]
        # Color constants
        if name.startswith('color.'): return name
        return NA

    def _node_name(self, node):
        if isinstance(node, Identifier): return node.name
        if isinstance(node, DotAccess): return self._dot_name(node)
        return None

    def _eval_binop(self, node):
        left = self._eval(node.left)
        right = self._eval(node.right)
        if node.op == 'and': return self._truthy(left) and self._truthy(right)
        if node.op == 'or': return self._truthy(left) or self._truthy(right)
        if node.op in ('==', '!='):
            if is_na(left) and is_na(right): return node.op == '=='
            if is_na(left) or is_na(right): return node.op == '!='
            return (left == right) if node.op == '==' else (left != right)
        if is_na(left) or is_na(right): return NA
        return self._arith(left, right, node.op)

    def _arith(self, left, right, op):
        if is_na(left) or is_na(right): return NA
        try: left = float(left); right = float(right)
        except (TypeError, ValueError): return NA
        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/': return left / right if right != 0 else NA
        if op == '%': return left % right if right != 0 else NA
        if op == '<': return left < right
        if op == '>': return left > right
        if op == '<=': return left <= right
        if op == '>=': return left >= right
        return NA

    def _truthy(self, val):
        if is_na(val): return False
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return val != 0
        return bool(val)

    # ─── Built-in Functions ────────────────────────────────────────────────

    def _call(self, node: FuncCall):
        name = node.name
        args = [self._eval(a) for a in node.args]
        kwargs = {k: self._eval(v) for k, v in node.kwargs.items()}

        # Strategy declaration
        if name == 'strategy':
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

        # Strategy actions
        if name == 'strategy.entry':
            id_ = args[0] if args else kwargs.get('id', '')
            direction = args[1] if len(args) > 1 else kwargs.get('direction', 'long')
            qty = kwargs.get('qty', None)
            comment = kwargs.get('comment', id_)
            direction = 'long' if direction == 'long' else 'short'
            limit_price = kwargs.get('limit', None)
            stop_price = kwargs.get('stop', None)
            order_type = 'market'
            entry_price = None
            if limit_price and not is_na(limit_price):
                order_type = 'limit'; entry_price = limit_price
            elif stop_price and not is_na(stop_price):
                order_type = 'stop'; entry_price = stop_price
            self.signals.append(Signal('entry', direction, qty, comment, order_type=order_type, entry_price=entry_price))
            return None

        if name == 'strategy.close':
            id_ = args[0] if args else kwargs.get('id', '')
            comment = kwargs.get('comment', id_)
            direction = 'long' if 'short' not in str(id_).lower() else 'short'
            self.signals.append(Signal('close', direction, comment=comment))
            return None

        if name == 'strategy.exit':
            id_ = args[0] if args else kwargs.get('id', '')
            from_entry = args[1] if len(args) > 1 else kwargs.get('from_entry', '')
            limit = kwargs.get('limit', None)
            stop = kwargs.get('stop', None)
            comment = kwargs.get('comment', id_)
            direction = 'long' if 'short' not in str(from_entry).lower() else 'short'
            self.signals.append(Signal('exit', direction, comment=comment, limit=limit, stop=stop, from_entry=from_entry))
            return None

        if name == 'strategy.close_all':
            self.signals.append(Signal('close', 'all'))
            return None

        if name == 'strategy.opentrades.entry_price':
            return self.variables.get('_entry_price', NA)

        # Input functions
        if name.startswith('input'):
            defval = args[0] if args else kwargs.get('defval', 0)
            title = kwargs.get('title', f'input_{len(self.inputs)}')
            if title in self.inputs: return self.inputs[title]
            self.inputs[title] = defval
            return defval

        # request.security - use MTF engine or secondary (SMT) data
        if name == 'request.security':
            if len(args) >= 3:
                symbol = args[0]
                timeframe = args[1]
                expr_val = args[2]
                # Resolve which OHLCV field
                expr_col = None
                for col in ('high', 'low', 'close', 'open', 'volume'):
                    if expr_val == self.variables.get(col):
                        expr_col = col; break
                # Try MTF engine for higher timeframes
                if self.variables.get('__mtf__') is not None and timeframe:
                    tf_str = self.variables.get('__mtf__').resolve_tf(str(timeframe))
                    if tf_str and self.variables.get('__mtf__').has_tf(tf_str) and expr_col:
                        val = self.variables.get('__mtf__').get(tf_str, expr_col, self.bar_index, 0)
                        return val if val is not None else NA
                # Try SMT data
                if self._smt_row_cache and expr_col and expr_col in self._smt_series:
                    return self._smt_series[expr_col].get(0)
                return expr_val
            return NA

        # Technical Indicators
        if name in ('ta.pivothigh', 'pivothigh'):
            return self._ta_pivothigh(args, kwargs)
        if name in ('ta.pivotlow', 'pivotlow'):
            return self._ta_pivotlow(args, kwargs)
        if name in ('ta.sma', 'sma'):
            return self._ta_sma(args, kwargs)
        if name in ('ta.ema', 'ema'):
            return self._ta_ema(args, kwargs)
        if name in ('ta.rsi', 'rsi'):
            return self._ta_rsi(args, kwargs)
        if name in ('ta.atr', 'atr'):
            return self._ta_atr(args, kwargs)
        if name in ('ta.wma', 'wma'):
            return self._ta_wma(args, kwargs)
        if name in ('ta.macd',):
            return self._ta_macd(args, kwargs)
        if name in ('ta.crossover', 'crossover'):
            return self._ta_crossover(args)
        if name in ('ta.crossunder', 'crossunder'):
            return self._ta_crossunder(args)
        if name in ('ta.highest', 'highest'):
            return self._ta_highest(args, kwargs)
        if name in ('ta.lowest', 'lowest'):
            return self._ta_lowest(args, kwargs)
        if name in ('ta.change', 'change'):
            return self._ta_change(args, kwargs)
        if name in ('ta.tr',):
            h = self.variables.get('high', 0); l = self.variables.get('low', 0)
            c_prev = self.series_data['close'].get(1)
            if is_na(c_prev): return h - l
            return max(h - l, abs(h - c_prev), abs(l - c_prev))
        if name in ('ta.stoch',):
            return self._ta_stoch(args, kwargs)
        if name in ('ta.bb',):
            return self._ta_bb(args, kwargs)
        if name in ('ta.cum',):
            source = args[0] if args else 0
            if is_na(source): return self.variables.get('_cum_total', 0)
            total = self.variables.get('_cum_total', 0) + float(source)
            self.variables['_cum_total'] = total; return total
        if name in ('ta.rising', 'ta.falling', 'ta.pivothigh', 'ta.pivotlow',
                     'ta.valuewhen', 'ta.barssince', 'ta.swma', 'ta.percentrank',
                     'ta.vwma', 'ta.mom'):
            return NA

        # Math
        if name in ('math.abs', 'abs'):
            return abs(args[0]) if args and not is_na(args[0]) else NA
        if name in ('math.max', 'max'):
            vals = [v for v in args if not is_na(v)]
            return max(vals) if vals else NA
        if name in ('math.min', 'min'):
            vals = [v for v in args if not is_na(v)]
            return min(vals) if vals else NA
        if name in ('math.sqrt',):
            return math.sqrt(args[0]) if args and not is_na(args[0]) and args[0] >= 0 else NA
        if name in ('math.pow',):
            return math.pow(args[0], args[1]) if len(args) >= 2 and not is_na(args[0]) and not is_na(args[1]) else NA
        if name in ('math.log',):
            return math.log(args[0]) if args and not is_na(args[0]) and args[0] > 0 else NA
        if name in ('math.round', 'round'):
            if args and not is_na(args[0]):
                return round(args[0], int(args[1]) if len(args) > 1 else 0)
            return NA
        if name in ('math.ceil',): return math.ceil(args[0]) if args and not is_na(args[0]) else NA
        if name in ('math.floor',): return math.floor(args[0]) if args and not is_na(args[0]) else NA
        if name in ('math.sign',):
            if args and not is_na(args[0]):
                return 1 if args[0] > 0 else (-1 if args[0] < 0 else 0)
            return NA

        # Platform SMT Divergence (any script can use these)
        if name in ('smt.divergence', 'smt.bull', 'SMTDivergence'):
            direction = int(args[0]) if args else 0
            if direction == 1 or name == 'smt.bull':
                return self.variables.get('_smt_bull_active', False)
            elif direction == -1:
                return self.variables.get('_smt_bear_active', False)
            else:
                return self.variables.get('_smt_bull_active', False) or self.variables.get('_smt_bear_active', False)
        if name in ('smt.bear',):
            return self.variables.get('_smt_bear_active', False)
        if name in ('smt.available', 'IsSMTAvailable'):
            return self.variables.get('_smt_available', False)

        # Utility
        if name == 'nz':
            val = args[0] if args else NA
            replacement = args[1] if len(args) > 1 else 0
            return replacement if is_na(val) else val
        if name == 'na':
            return is_na(args[0]) if args else NA
        if name == 'fixnan':
            return args[0] if args and not is_na(args[0]) else 0
        if name == 'int':
            return int(args[0]) if args and not is_na(args[0]) else NA
        if name == 'float':
            return float(args[0]) if args and not is_na(args[0]) else NA

        # String
        if name == 'str.tostring':
            if len(args) >= 1: return str(args[0]) if not is_na(args[0]) else 'NaN'
            return ''
        if name == 'str.contains':
            return args[1] in args[0] if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str) else False
        if name == 'str.substring':
            if len(args) >= 3 and isinstance(args[0], str): return args[0][int(args[1]):int(args[2])]
            return ''
        if name == 'str.pos':
            if len(args) >= 2 and isinstance(args[0], str): return args[0].find(args[1])
            return -1
        if name == 'str.length':
            return len(args[0]) if args and isinstance(args[0], str) else 0
        if name == 'str.trim':
            return args[0].strip() if args and isinstance(args[0], str) else ''

        # Array operations
        if name.startswith('array.new'):
            size = int(args[0]) if args and not is_na(args[0]) else 0
            fill = args[1] if len(args) > 1 else (0 if 'int' in name or 'float' in name else (False if 'bool' in name else NA))
            return [fill] * size
        if name == 'array.push':
            if args and isinstance(args[0], list): args[0].append(args[1] if len(args) > 1 else NA)
            return None
        if name == 'array.size':
            return len(args[0]) if args and isinstance(args[0], list) else 0
        if name == 'array.get':
            if args and isinstance(args[0], list) and len(args) > 1:
                idx = int(args[1])
                return args[0][idx] if 0 <= idx < len(args[0]) else NA
            return NA
        if name == 'array.set':
            if args and isinstance(args[0], list) and len(args) > 2:
                idx = int(args[1])
                if 0 <= idx < len(args[0]): args[0][idx] = args[2]
            return None
        if name == 'array.shift':
            if args and isinstance(args[0], list) and len(args[0]) > 0:
                return args[0].pop(0)
            return NA
        if name == 'array.pop':
            if args and isinstance(args[0], list) and len(args[0]) > 0:
                return args[0].pop()
            return NA
        if name == 'array.remove':
            if args and isinstance(args[0], list) and len(args) > 1:
                idx = int(args[1])
                if 0 <= idx < len(args[0]): args[0].pop(idx)
            return None
        if name == 'array.clear':
            if args and isinstance(args[0], list): args[0].clear()
            return None
        if name == 'array.min':
            if args and isinstance(args[0], list):
                vals = [v for v in args[0] if not is_na(v)]
                return min(vals) if vals else NA
            return NA
        if name == 'array.max':
            if args and isinstance(args[0], list):
                vals = [v for v in args[0] if not is_na(v)]
                return max(vals) if vals else NA
            return NA

        # Visual no-ops
        if name in ('plot', 'plotshape', 'plotchar', 'plotarrow', 'bgcolor',
                     'barcolor', 'hline', 'fill', 'label.new', 'label.delete',
                     'line.new', 'line.delete', 'line.set_x2', 'line.set_color', 'line.set_width',
                     'box.new', 'box.delete', 'box.set_right',
                     'table.new', 'table.cell',
                     'alertcondition', 'alert', 'color.new', 'color.rgb',
                     'log.info', 'log.warning', 'log.error', 'timestamp',
                     'time'):
            return None if name != 'color.new' else (args[0] if args else '#000000')

        # User-defined functions
        if name in self.user_functions:
            fdef = self.user_functions[name]
            saved = {}
            for i, p in enumerate(fdef.params):
                saved[p] = self.variables.get(p)
                self.variables[p] = args[i] if i < len(args) else NA
            result = NA
            for stmt in fdef.body:
                if isinstance(stmt, ExprStmt):
                    result = self._eval(stmt.expr)
                else:
                    self._exec(stmt)
            for p in fdef.params:
                if saved[p] is not None:
                    self.variables[p] = saved[p]
                else:
                    self.variables.pop(p, None)
            return result

        return NA

    # ─── Technical Indicators ──────────────────────────────────────────────

    def _resolve_source(self, arg):
        if isinstance(arg, str) and arg in self.series_data: return arg
        # Fast path: check OHLCV and common derived values first (O(1))
        v = self.variables
        if arg == v.get('close'): return 'close'
        if arg == v.get('open'): return 'open'
        if arg == v.get('high'): return 'high'
        if arg == v.get('low'): return 'low'
        if arg == v.get('hl2'): return 'hl2'
        if arg == v.get('hlc3'): return 'hlc3'
        if arg == v.get('ohlc4'): return 'ohlc4'
        # Slow path: scan series_data (only reached for custom series sources)
        for name, series in self.series_data.items():
            if series.data and series.data[-1] == arg: return name
        return 'close'

    def _get_history(self, source, length):
        series_name = self._resolve_source(source)
        series = self.series_data.get(series_name)
        if series is None or len(series) == 0: return []
        values = []
        for i in range(length):
            v = series.get(i)
            values.append(None if is_na(v) else float(v))
        values.reverse()
        return values

    def _ta_pivothigh(self, args, kwargs):
        if len(args) == 3:
            source_name = self._resolve_source(args[0])
            left = int(args[1]); right = int(args[2])
        elif len(args) == 2:
            source_name = 'high'
            left = int(args[0]); right = int(args[1])
        else:
            source_name = 'high'; left = 5; right = 5

        series = self.series_data.get(source_name)
        if series is None or len(series) < left + right + 1: return NA

        pivot_idx = right  # offset from current bar
        pivot_val = series.get(pivot_idx)
        if is_na(pivot_val): return NA

        for i in range(1, left + 1):
            v = series.get(pivot_idx + i)
            if is_na(v) or v >= pivot_val: return NA
        for i in range(1, right + 1):
            v = series.get(pivot_idx - i)
            if is_na(v) or v >= pivot_val: return NA

        return float(pivot_val)

    def _ta_pivotlow(self, args, kwargs):
        if len(args) == 3:
            source_name = self._resolve_source(args[0])
            left = int(args[1]); right = int(args[2])
        elif len(args) == 2:
            source_name = 'low'
            left = int(args[0]); right = int(args[1])
        else:
            source_name = 'low'; left = 5; right = 5

        series = self.series_data.get(source_name)
        if series is None or len(series) < left + right + 1: return NA

        pivot_idx = right
        pivot_val = series.get(pivot_idx)
        if is_na(pivot_val): return NA

        for i in range(1, left + 1):
            v = series.get(pivot_idx + i)
            if is_na(v) or v <= pivot_val: return NA
        for i in range(1, right + 1):
            v = series.get(pivot_idx - i)
            if is_na(v) or v <= pivot_val: return NA

        return float(pivot_val)

    def _ta_sma(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cache_key = ('sma', self._resolve_source(source), length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source, length)
        valid = [v for v in values if v is not None]
        result = sum(valid) / len(valid) if len(valid) >= length else NA
        self._bar_cache[cache_key] = result
        return result

    def _ta_ema(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 14))
        series_name = self._resolve_source(source)
        cache_key = f'_ema_{series_name}_{length}'
        current_val = source if isinstance(source, (int, float)) and not is_na(source) else self.variables.get(series_name, NA)
        if is_na(current_val): return NA
        prev_ema = self.variables.get(cache_key, NA)
        if is_na(prev_ema):
            values = self._get_history(source, length)
            valid = [v for v in values if v is not None]
            if len(valid) < length: return NA
            ema = sum(valid) / len(valid)
        else:
            k = 2.0 / (length + 1)
            ema = current_val * k + prev_ema * (1 - k)
        self.variables[cache_key] = ema
        return ema

    def _ta_rsi(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cache_key = ('rsi', self._resolve_source(source), length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source, length + 1)
        valid = [v for v in values if v is not None]
        if len(valid) < length + 1:
            self._bar_cache[cache_key] = NA; return NA
        arr = np.array(valid)
        diffs = np.diff(arr)
        gains = np.maximum(diffs, 0)
        losses = np.maximum(-diffs, 0)
        ag = float(np.mean(gains)); al = float(np.mean(losses))
        result = 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + ag / al))
        self._bar_cache[cache_key] = result
        return result

    def _ta_atr(self, args, kwargs):
        length = int(args[0] if args else kwargs.get('length', 14))
        cache_key = ('atr', length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        tr_values = []
        for i in range(length):
            h = self.series_data['high'].get(i); l = self.series_data['low'].get(i)
            c_prev = self.series_data['close'].get(i + 1)
            if is_na(h) or is_na(l):
                self._bar_cache[cache_key] = NA; return NA
            tr_values.append(max(h - l, abs(h - c_prev), abs(l - c_prev)) if not is_na(c_prev) else h - l)
        result = sum(tr_values) / len(tr_values) if tr_values else NA
        self._bar_cache[cache_key] = result
        return result

    def _ta_wma(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cache_key = ('wma', self._resolve_source(source), length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source, length)
        valid = [v for v in values if v is not None]
        if len(valid) < length:
            self._bar_cache[cache_key] = NA; return NA
        ws = sum(range(1, length + 1))
        result = sum(v * (i + 1) for i, v in enumerate(valid)) / ws
        self._bar_cache[cache_key] = result
        return result

    def _ta_macd(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        fast = int(args[1] if len(args) > 1 else kwargs.get('fastlen', 12))
        slow = int(args[2] if len(args) > 2 else kwargs.get('slowlen', 26))
        fast_ema = self._ta_ema([source, fast], {})
        slow_ema = self._ta_ema([source, slow], {})
        if is_na(fast_ema) or is_na(slow_ema): return (NA, NA, NA)
        return (fast_ema - slow_ema, NA, NA)

    def _ta_crossover(self, args):
        if len(args) < 2: return False
        a_now, b_now = args[0], args[1]
        if is_na(a_now) or is_na(b_now): return False
        a_name = self._resolve_source(a_now)
        b_name = self._resolve_source(b_now)
        if a_name and b_name and a_name in self.series_data and b_name in self.series_data:
            a_prev = self.series_data[a_name].get(1); b_prev = self.series_data[b_name].get(1)
            if not is_na(a_prev) and not is_na(b_prev):
                return float(a_now) > float(b_now) and float(a_prev) <= float(b_prev)
        return False

    def _ta_crossunder(self, args):
        if len(args) < 2: return False
        a_now, b_now = args[0], args[1]
        if is_na(a_now) or is_na(b_now): return False
        a_name = self._resolve_source(a_now)
        b_name = self._resolve_source(b_now)
        if a_name and b_name and a_name in self.series_data and b_name in self.series_data:
            a_prev = self.series_data[a_name].get(1); b_prev = self.series_data[b_name].get(1)
            if not is_na(a_prev) and not is_na(b_prev):
                return float(a_now) < float(b_now) and float(a_prev) >= float(b_prev)
        return False

    def _ta_highest(self, args, kwargs):
        source = args[0] if args else self.variables.get('high', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cache_key = ('highest', self._resolve_source(source), length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source, length)
        valid = [v for v in values if v is not None]
        result = max(valid) if valid else NA
        self._bar_cache[cache_key] = result
        return result

    def _ta_lowest(self, args, kwargs):
        source = args[0] if args else self.variables.get('low', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 14))
        cache_key = ('lowest', self._resolve_source(source), length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source, length)
        valid = [v for v in values if v is not None]
        result = min(valid) if valid else NA
        self._bar_cache[cache_key] = result
        return result

    def _ta_change(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 1))
        series_name = self._resolve_source(source)
        cache_key = ('change', series_name, length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        series = self.series_data.get(series_name)
        if series is None:
            self._bar_cache[cache_key] = NA; return NA
        current = series.get(0); prev = series.get(length)
        result = current - prev if not is_na(current) and not is_na(prev) else NA
        self._bar_cache[cache_key] = result
        return result

    def _ta_stoch(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[3] if len(args) > 3 else kwargs.get('length', 14))
        cache_key = ('stoch', self._resolve_source(source), length)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        highs = self._get_history(self.variables.get('high', 0), length)
        lows = self._get_history(self.variables.get('low', 0), length)
        vh = [v for v in highs if v is not None]; vl = [v for v in lows if v is not None]
        if len(vh) < length or len(vl) < length:
            self._bar_cache[cache_key] = NA; return NA
        highest = max(vh); lowest = min(vl)
        current = source if isinstance(source, (int, float)) else self.variables.get('close', 0)
        result = 100.0 * (current - lowest) / (highest - lowest) if highest != lowest else 50.0
        self._bar_cache[cache_key] = result
        return result

    def _ta_bb(self, args, kwargs):
        source = args[0] if args else self.variables.get('close', 0)
        length = int(args[1] if len(args) > 1 else kwargs.get('length', 20))
        mult = float(args[2] if len(args) > 2 else kwargs.get('mult', 2.0))
        cache_key = ('bb', self._resolve_source(source), length, mult)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source, length)
        valid = [v for v in values if v is not None]
        if len(valid) < length:
            self._bar_cache[cache_key] = (NA, NA, NA); return (NA, NA, NA)
        basis = sum(valid) / len(valid)
        std = math.sqrt(sum((v - basis) ** 2 for v in valid) / len(valid))
        result = (basis, basis + mult * std, basis - mult * std)
        self._bar_cache[cache_key] = result
        return result


def parse_pine(source: str) -> Program:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()

def create_interpreter(source: str):
    ast = parse_pine(source)
    try:
        from pine_fast import FastPineInterpreter
        return FastPineInterpreter(ast)
    except ImportError:
        return PineInterpreter(ast)
