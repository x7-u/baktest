"""
MQL5 Expert Advisor Tokenizer, Parser, and Interpreter.
Supports a practical subset of MQL5 for backtesting Expert Advisors.
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
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto(); PERCENT = auto()
    EQ = auto(); NEQ = auto(); LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    ASSIGN = auto()
    AMPAMP = auto()   # &&
    PIPEPIPE = auto() # ||
    EXCL = auto()     # !
    QUESTION = auto(); COLON = auto()
    DOT = auto(); COMMA = auto(); SEMICOLON = auto()
    LPAREN = auto(); RPAREN = auto()
    LBRACKET = auto(); RBRACKET = auto()
    LBRACE = auto(); RBRACE = auto()
    PLUSEQ = auto(); MINUSEQ = auto(); STAREQ = auto(); SLASHEQ = auto()
    INCREMENT = auto(); DECREMENT = auto()  # ++ --
    SCOPE = auto()    # ::
    HASH = auto()     # #
    AMP = auto()      # & (for references)
    EOF = auto()


class Token:
    __slots__ = ('type', 'value', 'line')
    def __init__(self, type: TT, value: Any, line: int = 0):
        self.type = type; self.value = value; self.line = line
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


KEYWORDS = {
    'if': 'IF', 'else': 'ELSE', 'for': 'FOR', 'while': 'WHILE', 'do': 'DO',
    'switch': 'SWITCH', 'case': 'CASE', 'default': 'DEFAULT',
    'break': 'BREAK', 'continue': 'CONTINUE', 'return': 'RETURN',
    'true': 'TRUE', 'false': 'FALSE',
    'void': 'TYPE', 'int': 'TYPE', 'double': 'TYPE', 'float': 'TYPE',
    'string': 'TYPE', 'bool': 'TYPE', 'datetime': 'TYPE', 'color': 'TYPE',
    'long': 'TYPE', 'ulong': 'TYPE', 'short': 'TYPE', 'ushort': 'TYPE',
    'char': 'TYPE', 'uchar': 'TYPE', 'uint': 'TYPE',
    'input': 'INPUT', 'sinput': 'INPUT',
    'static': 'STATIC', 'const': 'CONST', 'extern': 'EXTERN',
    'enum': 'ENUM', 'struct': 'STRUCT', 'class': 'CLASS',
    'new': 'NEW', 'delete': 'DELETE',
    'NULL': 'NULL',
}

TYPE_NAMES = {
    'void', 'int', 'double', 'float', 'string', 'bool', 'datetime', 'color',
    'long', 'ulong', 'short', 'ushort', 'char', 'uchar', 'uint',
}

TWO_CHAR_OPS = {
    '==': TT.EQ, '!=': TT.NEQ, '<=': TT.LTE, '>=': TT.GTE,
    '+=': TT.PLUSEQ, '-=': TT.MINUSEQ, '*=': TT.STAREQ, '/=': TT.SLASHEQ,
    '&&': TT.AMPAMP, '||': TT.PIPEPIPE, '++': TT.INCREMENT, '--': TT.DECREMENT,
    '::': TT.SCOPE,
}

ONE_CHAR_OPS = {
    '+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR, '/': TT.SLASH, '%': TT.PERCENT,
    '<': TT.LT, '>': TT.GT, '=': TT.ASSIGN,
    '?': TT.QUESTION, ':': TT.COLON, '.': TT.DOT, ',': TT.COMMA, ';': TT.SEMICOLON,
    '(': TT.LPAREN, ')': TT.RPAREN, '[': TT.LBRACKET, ']': TT.RBRACKET,
    '{': TT.LBRACE, '}': TT.RBRACE,
    '#': TT.HASH, '!': TT.EXCL, '&': TT.AMP,
}


def tokenize(source: str) -> list[Token]:
    """Tokenize MQL5 source code."""
    # Strip comments
    lines = source.split('\n')
    cleaned_lines = []
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
        cleaned_lines.append(line)

    source = '\n'.join(cleaned_lines)
    tokens = []
    i = 0
    line_num = 1

    while i < len(source):
        ch = source[i]

        # Newlines (track line number, skip)
        if ch == '\n':
            line_num += 1; i += 1; continue
        if ch in ' \t\r':
            i += 1; continue

        # Preprocessor directives: capture entire remaining line
        if ch == '#':
            j = i + 1
            while j < len(source) and source[j] in ' \t':
                j += 1
            k = j
            while k < len(source) and source[k] not in '\n':
                k += 1
            directive = source[j:k].strip()
            tokens.append(Token(TT.HASH, directive, line_num))
            i = k; continue

        # Numbers
        if ch.isdigit() or (ch == '.' and i+1 < len(source) and source[i+1].isdigit()):
            j = i; has_dot = False
            while j < len(source) and (source[j].isdigit() or (source[j] == '.' and not has_dot)):
                if source[j] == '.': has_dot = True
                j += 1
            tokens.append(Token(TT.NUMBER, float(source[i:j]), line_num))
            i = j; continue

        # Strings
        if ch in ('"', "'"):
            j = i + 1
            while j < len(source) and source[j] != ch:
                if source[j] == '\\': j += 1
                j += 1
            tokens.append(Token(TT.STRING, source[i+1:j], line_num))
            i = j + 1; continue

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            if word in KEYWORDS:
                kw = KEYWORDS[word]
                if kw == 'TRUE':
                    tokens.append(Token(TT.BOOL, True, line_num))
                elif kw == 'FALSE':
                    tokens.append(Token(TT.BOOL, False, line_num))
                elif kw == 'NULL':
                    tokens.append(Token(TT.IDENT, 'NULL', line_num))
                else:
                    tokens.append(Token(TT.IDENT, word, line_num))
            else:
                tokens.append(Token(TT.IDENT, word, line_num))
            i = j; continue

        # Two-char operators
        if i + 1 < len(source):
            two = source[i:i+2]
            if two in TWO_CHAR_OPS:
                tokens.append(Token(TWO_CHAR_OPS[two], two, line_num))
                i += 2; continue

        # One-char operators
        if ch in ONE_CHAR_OPS:
            tokens.append(Token(ONE_CHAR_OPS[ch], ch, line_num))
            i += 1; continue

        i += 1  # skip unknown

    tokens.append(Token(TT.EOF, None, line_num))
    return tokens


# ─── AST Nodes ─────────────────────────────────────────────────────────────────
# _tag integers enable fast dispatch in Cython (avoids isinstance chains)

_OP_ID = {
    '+': 0, '-': 1, '*': 2, '/': 3, '%': 4,
    '<': 5, '>': 6, '<=': 7, '>=': 8, '==': 9, '!=': 10,
    '&&': 11, '||': 12, 'and': 11, 'or': 12,
}

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
class NullLit(ASTNode):
    _tag = 3
class Identifier(ASTNode):
    _tag = 4
    def __init__(self, name): self.name = name
class ArrayLit(ASTNode):
    _tag = 5
    def __init__(self, elements): self.elements = elements
class BinOp(ASTNode):
    _tag = 6
    def __init__(self, op, left, right):
        self.op = op; self.left = left; self.right = right
        self.op_id = _OP_ID.get(op, -1)
class UnaryOp(ASTNode):
    _tag = 7
    def __init__(self, op, operand): self.op = op; self.operand = operand
class ArrayAccess(ASTNode):
    _tag = 8
    def __init__(self, expr, index): self.expr = expr; self.index = index
class FuncCall(ASTNode):
    _tag = 9
    def __init__(self, name, args): self.name = name; self.args = args
class DotAccess(ASTNode):
    _tag = 10
    def __init__(self, obj, attr): self.obj = obj; self.attr = attr
class Ternary(ASTNode):
    _tag = 11
    def __init__(self, cond, true_val, false_val):
        self.cond = cond; self.true_val = true_val; self.false_val = false_val
class Assignment(ASTNode):
    _tag = 12
    def __init__(self, name, value, op='='):
        self.name = name; self.value = value; self.op = op
        self.op_id = _OP_ID.get(op, -1)
class IfStmt(ASTNode):
    _tag = 13
    def __init__(self, cond, body, else_body=None):
        self.cond = cond; self.body = body; self.else_body = else_body
class ForStmt(ASTNode):
    _tag = 14
    def __init__(self, init, cond, step, body):
        self.init = init; self.cond = cond; self.step = step; self.body = body
class ExprStmt(ASTNode):
    _tag = 15
    def __init__(self, expr): self.expr = expr
class ContinueStmt(ASTNode):
    _tag = 16
class BreakStmt(ASTNode):
    _tag = 17
class SwitchStmt(ASTNode):
    _tag = 18
    def __init__(self, expr, cases, default):
        self.expr = expr; self.cases = cases; self.default = default
class FuncDef(ASTNode):
    _tag = 19
    def __init__(self, name, return_type, params, body):
        self.name = name; self.return_type = return_type
        self.params = params; self.body = body
class Program(ASTNode):
    _tag = 20
    def __init__(self, stmts): self.stmts = stmts
class WhileStmt(ASTNode):
    _tag = 21
    def __init__(self, cond, body): self.cond = cond; self.body = body
class DoWhileStmt(ASTNode):
    _tag = 22
    def __init__(self, body, cond): self.body = body; self.cond = cond
class ReturnStmt(ASTNode):
    _tag = 23
    def __init__(self, value): self.value = value
class VarDecl(ASTNode):
    _tag = 24
    def __init__(self, type_name, name, value=None, is_array=False):
        self.type_name = type_name; self.name = name; self.value = value; self.is_array = is_array
class InputDecl(ASTNode):
    _tag = 25
    def __init__(self, type_name, name, value=None, comment=''):
        self.type_name = type_name; self.name = name; self.value = value; self.comment = comment
class PreprocessorDir(ASTNode):
    _tag = 26
    def __init__(self, directive): self.directive = directive
class EnumDecl(ASTNode):
    _tag = 27
    def __init__(self, name, values): self.name = name; self.values = values
class CastExpr(ASTNode):
    _tag = 28
    def __init__(self, target_type, expr): self.target_type = target_type; self.expr = expr
class PostfixOp(ASTNode):
    _tag = 29
    def __init__(self, operand, op): self.operand = operand; self.op = op


# ─── Parser ────────────────────────────────────────────────────────────────────

class FuncReturn(Exception):
    def __init__(self, value): self.value = value

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0) -> Token:
        p = self.pos + offset
        return self.tokens[p] if p < len(self.tokens) else Token(TT.EOF, None)

    def advance(self) -> Token:
        t = self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TT.EOF, None)
        self.pos += 1
        return t

    def expect(self, tt: TT) -> Token:
        t = self.advance()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt}, got {t.type} ({t.value!r}) at line {t.line}")
        return t

    def match(self, tt: TT) -> bool:
        if self.peek().type == tt:
            self.advance(); return True
        return False

    def _is_type(self, tok=None):
        """Check if token is a type keyword."""
        t = tok or self.peek()
        return t.type == TT.IDENT and t.value in TYPE_NAMES

    def _is_class_type(self, tok=None):
        """Check if token looks like a class name (CTrade, MqlTradeRequest etc.)."""
        t = tok or self.peek()
        return t.type == TT.IDENT and t.value[0:1].isupper() and t.value not in KEYWORDS

    def parse(self) -> Program:
        stmts = []
        while self.peek().type != TT.EOF:
            stmt = self.parse_top_level()
            if stmt is not None:
                if isinstance(stmt, list):
                    stmts.extend(stmt)
                else:
                    stmts.append(stmt)
        return Program(stmts)

    def parse_top_level(self):
        tok = self.peek()

        # Preprocessor
        if tok.type == TT.HASH:
            return self.parse_preprocessor()

        # input declaration
        if tok.type == TT.IDENT and tok.value in ('input', 'sinput'):
            return self.parse_input_decl()

        # enum
        if tok.type == TT.IDENT and tok.value == 'enum':
            return self.parse_enum_decl()

        # static, const, extern — modifiers before type
        if tok.type == TT.IDENT and tok.value in ('static', 'const', 'extern'):
            self.advance()  # skip modifier
            return self.parse_top_level()

        # Type-prefixed: could be function def or variable decl
        if self._is_type() or self._is_class_type():
            return self._parse_typed_decl_or_func()

        # struct/class
        if tok.type == TT.IDENT and tok.value in ('struct', 'class'):
            return self._skip_struct_class()

        # Semicolons
        if tok.type == TT.SEMICOLON:
            self.advance(); return None

        # Fallback: expression statement
        return self.parse_stmt()

    def parse_preprocessor(self):
        tok = self.advance()  # HASH token, value is the directive content
        directive = tok.value
        return PreprocessorDir(directive)

    def parse_input_decl(self):
        self.advance()  # 'input' or 'sinput'
        type_tok = self.advance()  # type or 'group'
        type_name = type_tok.value

        # input group "Label" — section header, skip
        if type_name == 'group':
            if self.peek().type == TT.STRING:
                self.advance()
            self.match(TT.SEMICOLON)
            return None

        # Skip array brackets
        if self.peek().type == TT.LBRACKET:
            self.advance(); self.advance()  # []

        # The next token should be the variable name
        if self.peek().type == TT.IDENT:
            name = self.advance().value
        elif self.peek().type == TT.STRING:
            # Handle unusual case: skip to semicolon
            while self.peek().type != TT.SEMICOLON and self.peek().type != TT.EOF:
                self.advance()
            self.match(TT.SEMICOLON)
            return None
        else:
            # Skip to semicolon
            while self.peek().type != TT.SEMICOLON and self.peek().type != TT.EOF:
                self.advance()
            self.match(TT.SEMICOLON)
            return None

        value = None
        if self.match(TT.ASSIGN):
            value = self.parse_expr()
        self.match(TT.SEMICOLON)
        return InputDecl(type_name, name, value, comment=name)

    def parse_enum_decl(self):
        self.advance()  # 'enum'
        name = self.expect(TT.IDENT).value
        self.expect(TT.LBRACE)
        values = {}
        counter = 0
        while self.peek().type != TT.RBRACE and self.peek().type != TT.EOF:
            vname = self.expect(TT.IDENT).value
            if self.match(TT.ASSIGN):
                counter = int(self.parse_expr().value) if isinstance(self.peek(), NumberLit) else counter
                # re-parse: just get the number
                val_node = self.parse_expr() if self.tokens[self.pos-1].type == TT.ASSIGN else None
                # Simpler: we already consumed ASSIGN, parse value
            values[vname] = counter
            counter += 1
            self.match(TT.COMMA)
        self.expect(TT.RBRACE)
        self.match(TT.SEMICOLON)
        return EnumDecl(name, values)

    def _skip_struct_class(self):
        """Skip struct/class definitions — extract nothing."""
        self.advance()  # struct/class
        if self.peek().type == TT.IDENT:
            self.advance()  # name
        if self.peek().type == TT.COLON:
            # inheritance: skip until {
            while self.peek().type not in (TT.LBRACE, TT.EOF):
                self.advance()
        if self.peek().type == TT.LBRACE:
            self._skip_braces()
        self.match(TT.SEMICOLON)
        return None

    def _skip_braces(self):
        """Skip matched {} block."""
        self.expect(TT.LBRACE)
        depth = 1
        while depth > 0 and self.peek().type != TT.EOF:
            t = self.advance()
            if t.type == TT.LBRACE: depth += 1
            elif t.type == TT.RBRACE: depth -= 1

    def _parse_typed_decl_or_func(self):
        """Parse: type name(...) { body } OR type name = expr;"""
        save = self.pos
        type_name = self.advance().value  # type or class name

        # Handle pointer: type* name
        if self.peek().type == TT.STAR:
            self.advance()

        # Handle array return: type[]
        is_array = False
        if self.peek().type == TT.LBRACKET:
            self.advance()
            if self.peek().type == TT.RBRACKET:
                self.advance()
            is_array = True

        if self.peek().type != TT.IDENT:
            self.pos = save
            return self.parse_stmt()

        name = self.advance().value

        # Function definition: type name(params) { body }
        if self.peek().type == TT.LPAREN:
            params = self._parse_param_list()
            # Optional const qualifier
            if self.peek().type == TT.IDENT and self.peek().value == 'const':
                self.advance()
            if self.peek().type == TT.LBRACE:
                body = self.parse_block()
                return FuncDef(name, type_name, params, body)
            # Forward declaration: type name(params);
            self.match(TT.SEMICOLON)
            return None

        # Array declaration: type name[]
        if self.peek().type == TT.LBRACKET:
            self.advance()
            if self.peek().type == TT.RBRACKET:
                self.advance()
            else:
                self.parse_expr()  # size
                self.expect(TT.RBRACKET)
            is_array = True
            value = None
            if self.match(TT.ASSIGN):
                value = self.parse_expr()
        else:
            # Variable declaration: type name = expr;
            value = None
            if self.match(TT.ASSIGN):
                value = self.parse_expr()

        # Multiple declarations: type name1 = v1, name2 = v2;
        decls = [VarDecl(type_name, name, value, is_array)]
        while self.match(TT.COMMA):
            n2 = self.expect(TT.IDENT).value
            v2 = None
            is_arr2 = False
            if self.peek().type == TT.LBRACKET:
                self.advance()
                is_arr2 = True
                if self.peek().type == TT.RBRACKET:
                    self.advance()
                else:
                    self.parse_expr(); self.expect(TT.RBRACKET)
            if self.match(TT.ASSIGN):
                v2 = self.parse_expr()
            decls.append(VarDecl(type_name, n2, v2, is_array=is_arr2))
        self.match(TT.SEMICOLON)
        return decls if len(decls) > 1 else decls[0]

    def _parse_param_list(self):
        """Parse function parameter list: (type name, type name = default, ...)"""
        self.expect(TT.LPAREN)
        params = []
        while self.peek().type != TT.RPAREN and self.peek().type != TT.EOF:
            # Optional const, static
            while self.peek().type == TT.IDENT and self.peek().value in ('const', 'static'):
                self.advance()
            if self._is_type() or self._is_class_type():
                self.advance()  # type
            # Handle &
            if self.peek().type == TT.AMP:
                self.advance()
            # Handle array params type name[]
            if self.peek().type == TT.LBRACKET:
                self.advance()
                if self.peek().type == TT.RBRACKET:
                    self.advance()
            if self.peek().type == TT.IDENT:
                pname = self.advance().value
                # Array param
                if self.peek().type == TT.LBRACKET:
                    self.advance()
                    if self.peek().type == TT.RBRACKET: self.advance()
                default = None
                if self.match(TT.ASSIGN):
                    default = self.parse_expr()
                params.append((pname, default))
            if not self.match(TT.COMMA):
                break
        self.expect(TT.RPAREN)
        return params

    def parse_block(self):
        """Parse { stmt; stmt; ... }"""
        self.expect(TT.LBRACE)
        stmts = []
        while self.peek().type != TT.RBRACE and self.peek().type != TT.EOF:
            s = self.parse_stmt()
            if s is not None:
                if isinstance(s, list):
                    stmts.extend(s)
                else:
                    stmts.append(s)
        self.expect(TT.RBRACE)
        return stmts

    def parse_stmt(self):
        tok = self.peek()

        if tok.type == TT.SEMICOLON:
            self.advance(); return None

        if tok.type == TT.LBRACE:
            return self._parse_block_as_stmts()

        if tok.type == TT.IDENT:
            if tok.value == 'if': return self.parse_if()
            if tok.value == 'for': return self.parse_for()
            if tok.value == 'while': return self.parse_while()
            if tok.value == 'do': return self.parse_do_while()
            if tok.value == 'switch': return self.parse_switch()
            if tok.value == 'return': return self.parse_return()
            if tok.value == 'break': self.advance(); self.match(TT.SEMICOLON); return BreakStmt()
            if tok.value == 'continue': self.advance(); self.match(TT.SEMICOLON); return ContinueStmt()
            if tok.value in ('static', 'const', 'extern'):
                self.advance()
                return self.parse_stmt()
            if tok.value in ('input', 'sinput'):
                return self.parse_input_decl()
            if tok.value == 'enum':
                return self.parse_enum_decl()

            # Type-prefixed declaration inside function body
            if self._is_type() or self._is_class_type():
                # Lookahead: is this type ident = ... or type ident(... ?
                save = self.pos
                self.advance()  # type
                if self.peek().type == TT.STAR: self.advance()
                if self.peek().type == TT.LBRACKET:
                    # skip []
                    self.advance()
                    if self.peek().type == TT.RBRACKET: self.advance()
                if self.peek().type == TT.IDENT:
                    self.pos = save
                    return self._parse_typed_decl_or_func()
                self.pos = save

        # Expression statement (including assignments)
        return self.parse_expr_stmt()

    def _parse_block_as_stmts(self):
        """Parse { stmts } returning a list but wrap in a single IfStmt body etc."""
        stmts = self.parse_block()
        # If only one statement, return it directly
        if len(stmts) == 1: return stmts[0]
        # Return as a pseudo-block: use IfStmt(True, stmts)
        return IfStmt(BoolLit(True), stmts)

    def parse_if(self):
        self.advance()  # 'if'
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        if self.peek().type == TT.LBRACE:
            body = self.parse_block()
        else:
            s = self.parse_stmt()
            body = [s] if s else []
        else_body = None
        if self.peek().type == TT.IDENT and self.peek().value == 'else':
            self.advance()
            if self.peek().type == TT.IDENT and self.peek().value == 'if':
                else_body = [self.parse_if()]
            elif self.peek().type == TT.LBRACE:
                else_body = self.parse_block()
            else:
                s = self.parse_stmt()
                else_body = [s] if s else []
        return IfStmt(cond, body, else_body)

    def parse_for(self):
        self.advance()  # 'for'
        self.expect(TT.LPAREN)
        # Init
        init = None
        if self.peek().type != TT.SEMICOLON:
            if self._is_type():
                init = self._parse_typed_decl_or_func()
                # The typed decl consumed the semicolon already — but we need it for the for loop
                # Actually _parse_typed_decl_or_func calls match(SEMICOLON), so it may have consumed it
                # Undo: check
            else:
                init = self.parse_expr()
                self.match(TT.SEMICOLON)
        else:
            self.advance()  # ;

        # Condition
        cond = None
        if self.peek().type != TT.SEMICOLON:
            cond = self.parse_expr()
        self.expect(TT.SEMICOLON)

        # Step
        step = None
        if self.peek().type != TT.RPAREN:
            step = self.parse_expr()
        self.expect(TT.RPAREN)

        if self.peek().type == TT.LBRACE:
            body = self.parse_block()
        else:
            s = self.parse_stmt()
            body = [s] if s else []

        return ForStmt(init, cond, step, body)

    def parse_while(self):
        self.advance()  # 'while'
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        if self.peek().type == TT.LBRACE:
            body = self.parse_block()
        else:
            s = self.parse_stmt()
            body = [s] if s else []
        return WhileStmt(cond, body)

    def parse_do_while(self):
        self.advance()  # 'do'
        body = self.parse_block()
        if self.peek().type == TT.IDENT and self.peek().value == 'while':
            self.advance()
        self.expect(TT.LPAREN)
        cond = self.parse_expr()
        self.expect(TT.RPAREN)
        self.match(TT.SEMICOLON)
        return DoWhileStmt(body, cond)

    def parse_switch(self):
        self.advance()  # 'switch'
        self.expect(TT.LPAREN)
        expr = self.parse_expr()
        self.expect(TT.RPAREN)
        self.expect(TT.LBRACE)
        cases = []
        default = None
        while self.peek().type != TT.RBRACE and self.peek().type != TT.EOF:
            if self.peek().type == TT.IDENT and self.peek().value == 'case':
                self.advance()  # 'case'
                val = self.parse_expr()
                self.expect(TT.COLON)
                body = []
                while (self.peek().type != TT.IDENT or self.peek().value not in ('case', 'default')) \
                        and self.peek().type != TT.RBRACE and self.peek().type != TT.EOF:
                    s = self.parse_stmt()
                    if s: body.append(s)
                cases.append((val, body))
            elif self.peek().type == TT.IDENT and self.peek().value == 'default':
                self.advance(); self.expect(TT.COLON)
                default = []
                while self.peek().type != TT.RBRACE and self.peek().type != TT.EOF:
                    s = self.parse_stmt()
                    if s: default.append(s)
            else:
                self.advance()  # skip unexpected
        self.expect(TT.RBRACE)
        return SwitchStmt(expr, cases, default)

    def parse_return(self):
        self.advance()  # 'return'
        value = None
        if self.peek().type != TT.SEMICOLON and self.peek().type != TT.RPAREN:
            value = self.parse_expr()
        # Handle return(expr); syntax
        self.match(TT.SEMICOLON)
        return ReturnStmt(value)

    def parse_expr_stmt(self):
        expr = self.parse_expr()
        # Check for assignment operators
        tok = self.peek()
        if tok.type in (TT.ASSIGN, TT.PLUSEQ, TT.MINUSEQ, TT.STAREQ, TT.SLASHEQ):
            if isinstance(expr, Identifier):
                op = self.advance()
                value = self.parse_expr()
                self.match(TT.SEMICOLON)
                return Assignment(expr.name, value, op.value)
            elif isinstance(expr, ArrayAccess):
                op = self.advance()
                value = self.parse_expr()
                self.match(TT.SEMICOLON)
                # Array element assignment: arr[i] = val
                return ExprStmt(FuncCall('__array_set__', [expr.expr, expr.index, value]))
            elif isinstance(expr, DotAccess):
                op = self.advance()
                value = self.parse_expr()
                self.match(TT.SEMICOLON)
                full_name = self._dot_to_name(expr)
                if full_name:
                    return Assignment(full_name, value, op.value)
        self.match(TT.SEMICOLON)
        return ExprStmt(expr)

    def _dot_to_name(self, node):
        if isinstance(node, Identifier): return node.name
        if isinstance(node, DotAccess):
            p = self._dot_to_name(node.obj)
            return p + '.' + node.attr if p else None
        return None

    # ─── Expressions ───────────────────────────────────────────────────────

    def parse_expr(self):
        return self.parse_ternary()

    def parse_ternary(self):
        expr = self.parse_or()
        if self.peek().type == TT.QUESTION:
            self.advance()
            true_val = self.parse_expr()
            self.expect(TT.COLON)
            false_val = self.parse_ternary()
            return Ternary(expr, true_val, false_val)
        return expr

    def parse_or(self):
        left = self.parse_and()
        while self.peek().type == TT.PIPEPIPE:
            self.advance(); left = BinOp('||', left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek().type == TT.AMPAMP:
            self.advance(); left = BinOp('&&', left, self.parse_not())
        return left

    def parse_not(self):
        if self.peek().type == TT.EXCL:
            self.advance(); return UnaryOp('!', self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_addition()
        while self.peek().type in (TT.EQ, TT.NEQ, TT.LT, TT.GT, TT.LTE, TT.GTE):
            op = self.advance(); left = BinOp(op.value, left, self.parse_addition())
        return left

    def parse_addition(self):
        left = self.parse_multiplication()
        while self.peek().type in (TT.PLUS, TT.MINUS):
            op = self.advance(); left = BinOp(op.value, left, self.parse_multiplication())
        return left

    def parse_multiplication(self):
        left = self.parse_unary()
        while self.peek().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            op = self.advance(); left = BinOp(op.value, left, self.parse_unary())
        return left

    def parse_unary(self):
        if self.peek().type == TT.MINUS:
            self.advance(); return UnaryOp('-', self.parse_unary())
        if self.peek().type == TT.PLUS:
            self.advance(); return self.parse_unary()
        # Prefix ++ / --
        if self.peek().type == TT.INCREMENT:
            self.advance()
            operand = self.parse_unary()
            return UnaryOp('++pre', operand)
        if self.peek().type == TT.DECREMENT:
            self.advance()
            operand = self.parse_unary()
            return UnaryOp('--pre', operand)
        # Cast: (type)expr
        if self.peek().type == TT.LPAREN and self.peek(1).type == TT.IDENT and self.peek(1).value in TYPE_NAMES:
            save = self.pos
            self.advance()  # (
            type_name = self.advance().value
            if self.peek().type == TT.RPAREN:
                self.advance()  # )
                expr = self.parse_unary()
                return CastExpr(type_name, expr)
            self.pos = save
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if self.peek().type == TT.DOT:
                self.advance()
                attr = self.expect(TT.IDENT).value
                # Method call: obj.method(args)
                if self.peek().type == TT.LPAREN:
                    args = self._parse_arg_list()
                    name = self._node_to_name(expr) + '.' + attr
                    expr = FuncCall(name, args)
                else:
                    expr = DotAccess(expr, attr)
            elif self.peek().type == TT.LPAREN:
                name = self._node_to_name(expr)
                args = self._parse_arg_list()
                expr = FuncCall(name, args)
            elif self.peek().type == TT.LBRACKET:
                self.advance()
                index = self.parse_expr()
                self.expect(TT.RBRACKET)
                expr = ArrayAccess(expr, index)
            elif self.peek().type == TT.INCREMENT:
                self.advance()
                expr = PostfixOp(expr, '++')
            elif self.peek().type == TT.DECREMENT:
                self.advance()
                expr = PostfixOp(expr, '--')
            elif self.peek().type == TT.SCOPE:
                self.advance()
                attr = self.expect(TT.IDENT).value
                expr = DotAccess(expr, attr)
            else:
                break
        return expr

    def _parse_arg_list(self):
        self.expect(TT.LPAREN)
        args = []
        while self.peek().type != TT.RPAREN and self.peek().type != TT.EOF:
            args.append(self.parse_expr())
            if not self.match(TT.COMMA):
                break
        self.expect(TT.RPAREN)
        return args

    def _node_to_name(self, node) -> str:
        if isinstance(node, Identifier): return node.name
        if isinstance(node, DotAccess): return self._node_to_name(node.obj) + '.' + node.attr
        return '<expr>'

    def parse_primary(self):
        tok = self.peek()
        if tok.type == TT.NUMBER: self.advance(); return NumberLit(tok.value)
        if tok.type == TT.STRING: self.advance(); return StringLit(tok.value)
        if tok.type == TT.BOOL: self.advance(); return BoolLit(tok.value)
        if tok.type == TT.IDENT:
            # return(expr) syntax
            if tok.value == 'return':
                self.advance()
                if self.peek().type == TT.LPAREN:
                    self.advance()
                    val = self.parse_expr()
                    self.expect(TT.RPAREN)
                    return val  # Used in expression context
                return NullLit()
            self.advance()
            return Identifier(tok.value)
        if tok.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TT.RPAREN)
            return expr
        if tok.type == TT.LBRACE:
            # Array initializer {1, 2, 3}
            self.advance()
            elements = []
            while self.peek().type != TT.RBRACE and self.peek().type != TT.EOF:
                elements.append(self.parse_expr())
                self.match(TT.COMMA)
            self.expect(TT.RBRACE)
            return ArrayLit(elements)
        raise SyntaxError(f"Unexpected token {tok.type} ({tok.value!r}) at line {tok.line}")


# ─── Interpreter ───────────────────────────────────────────────────────────────

class MQL5NA:
    _instance = None
    def __new__(cls):
        if cls._instance is None: cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self): return 'NULL'
    def __bool__(self): return False

NA = MQL5NA()
PineNA = MQL5NA  # Alias for compatibility with app.py sanitizer

def is_na(v):
    return v is NA or v is None or (isinstance(v, float) and math.isnan(v))


class Series:
    def __init__(self, max_lookback=5000):
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


# ─── MQL5 Constants ────────────────────────────────────────────────────────────

MQL5_CONSTANTS = {
    # Order types
    'ORDER_TYPE_BUY': 0, 'ORDER_TYPE_SELL': 1,
    'ORDER_TYPE_BUY_LIMIT': 2, 'ORDER_TYPE_SELL_LIMIT': 3,
    'ORDER_TYPE_BUY_STOP': 4, 'ORDER_TYPE_SELL_STOP': 5,
    # Position properties
    'POSITION_TYPE_BUY': 0, 'POSITION_TYPE_SELL': 1,
    'POSITION_PRICE_OPEN': 100, 'POSITION_SL': 101, 'POSITION_TP': 102,
    'POSITION_PROFIT': 103, 'POSITION_VOLUME': 104, 'POSITION_PRICE_CURRENT': 105,
    'POSITION_TYPE': 106, 'POSITION_TIME': 107, 'POSITION_MAGIC': 108,
    'POSITION_SWAP': 109, 'POSITION_COMMISSION': 110,
    # Symbol info
    'SYMBOL_BID': 200, 'SYMBOL_ASK': 201, 'SYMBOL_POINT': 202,
    'SYMBOL_DIGITS': 203, 'SYMBOL_SPREAD': 204,
    'SYMBOL_TRADE_TICK_VALUE': 205, 'SYMBOL_TRADE_CONTRACT_SIZE': 206,
    # MA methods
    'MODE_SMA': 0, 'MODE_EMA': 1, 'MODE_SMMA': 2, 'MODE_LWMA': 3,
    # Applied price
    'PRICE_CLOSE': 0, 'PRICE_OPEN': 1, 'PRICE_HIGH': 2, 'PRICE_LOW': 3,
    'PRICE_MEDIAN': 4, 'PRICE_TYPICAL': 5, 'PRICE_WEIGHTED': 6,
    # Timeframes
    'PERIOD_CURRENT': 0, 'PERIOD_M1': 1, 'PERIOD_M5': 5, 'PERIOD_M15': 15,
    'PERIOD_M30': 30, 'PERIOD_H1': 60, 'PERIOD_H4': 240, 'PERIOD_D1': 1440,
    'PERIOD_W1': 10080, 'PERIOD_MN1': 43200,
    # Init return
    'INIT_SUCCEEDED': 0, 'INIT_FAILED': -1,
    # Trade return codes
    'TRADE_RETCODE_DONE': 10009,
    # Type filling
    'ORDER_FILLING_FOK': 0, 'ORDER_FILLING_IOC': 1, 'ORDER_FILLING_RETURN': 2,
    # Predefined
    'EMPTY_VALUE': float('nan'), 'WRONG_VALUE': -1, 'INVALID_HANDLE': -1,
    'CLR_NONE': 0, 'clrNONE': 0,
    'CHARTS_MAX': 100,
    # Account info
    'ACCOUNT_BALANCE': 300, 'ACCOUNT_EQUITY': 301, 'ACCOUNT_PROFIT': 302,
    'ACCOUNT_MARGIN': 303, 'ACCOUNT_MARGIN_FREE': 304,
    # STO price
    'STO_LOWHIGH': 0, 'STO_CLOSECLOSE': 1,
    # Position string properties
    'POSITION_SYMBOL': 200, 'POSITION_COMMENT': 201,
    # Symbol string properties
    'SYMBOL_CURRENCY_PROFIT': 300, 'SYMBOL_CURRENCY_BASE': 301,
    'SYMBOL_DESCRIPTION': 302,
    # Symbol double properties (additional)
    'SYMBOL_VOLUME_MIN': 207, 'SYMBOL_VOLUME_MAX': 208, 'SYMBOL_VOLUME_STEP': 209,
    'SYMBOL_TRADE_TICK_SIZE': 210, 'SYMBOL_SWAP_LONG': 211, 'SYMBOL_SWAP_SHORT': 212,
    # Timeframe enum values
    'ENUM_TIMEFRAMES': 0,
}


class MQL5Interpreter:
    def __init__(self, ast: Program):
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
        self._series_vars = set()

        # MQL5-specific
        self._indicator_handles = {}  # handle_id -> {type, params, cache}
        self._next_handle = 1
        self._ctrade_available = False
        self._position_type = -1   # -1=none, 0=BUY, 1=SELL
        self._position_volume = 0.0
        self._position_price_open = 0.0
        self._position_sl = 0.0
        self._position_tp = 0.0
        self._position_profit = 0.0
        self._account_balance = 10000.0
        self._account_equity = 10000.0
        self._array_series_flags = {}  # array id -> bool (timeseries mode)
        self._contract_size = 100000.0  # standard forex lot
        self._is_jpy_pair = False       # detected in setup() from price levels
        self._pip_value = 1.0           # USD per pip per unit

        # OnInit / OnTick / OnDeinit function bodies
        self._on_init = None
        self._on_tick = None
        self._on_deinit = None
        self._global_stmts = []  # top-level non-function statements

        # Load constants BEFORE prescan so input defaults can reference them
        self.variables.update(MQL5_CONSTANTS)

        # Pre-scan AST
        self._prescan_ast()

        # SMT data
        self._smt_row_cache = None
        self._smt_series = {}

        # Row cache
        self._row_cache = []

        # EMA cache
        self._ema_cache = {}
        self._bar_cache = {}
        self._max_exec_count = 0
        self._max_exec_limit = 500000  # max AST evaluations per bar

    def _prescan_ast(self):
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
                # Handle #define NAME VALUE
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

    def _default_for_type(self, type_name):
        if type_name in ('int', 'long', 'short', 'char', 'uchar', 'ushort', 'uint', 'ulong'):
            return 0
        if type_name in ('double', 'float'):
            return 0.0
        if type_name == 'string':
            return ''
        if type_name == 'bool':
            return False
        return 0

    def setup(self, bars):
        self.bars = bars
        self.bar_index = 0
        self._first_bar = True
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self.series_data[col] = Series()

        # Precompute row data and timestamps
        self._row_cache = []
        self._timestamps = []  # real Unix timestamps from CSV datetime
        date_col = None
        for col in ('datetime', 'date', 'Date', 'Datetime'):
            if col in bars.columns:
                date_col = col; break

        for i in range(len(bars)):
            r = bars.iloc[i]
            self._row_cache.append({
                'open': r.get('open', 0), 'high': r.get('high', 0),
                'low': r.get('low', 0), 'close': r.get('close', 0),
                'volume': r.get('volume', 0),
            })
            # Parse real timestamp
            ts = 0
            if date_col:
                try:
                    import pandas as pd
                    dt = pd.Timestamp(r[date_col])
                    ts = int(dt.timestamp())
                except Exception:
                    ts = 1700000000 + i * 300  # fallback: 5min intervals
            else:
                ts = 1700000000 + i * 300
            self._timestamps.append(ts)

        # Detect instrument type from price levels
        avg_price = bars['close'].iloc[:min(100, len(bars))].mean() if len(bars) > 0 else 1.0
        if avg_price > 10:  # JPY pair (prices like 87, 150)
            self._is_jpy_pair = True
            self.variables['_Point'] = 0.001
            self.variables['_Digits'] = 3
            # For JPY pairs: 1 pip = 0.01, tick value depends on USDJPY rate
            # Approximate: 1 standard lot moves ~$6.67 per pip at USDJPY=150
            self._pip_value = self._contract_size * 0.01 / avg_price  # ≈ $6.67 for USDJPY≈150
        elif avg_price > 1:  # Standard forex (e.g., EURUSD ~1.10)
            self._is_jpy_pair = False
            self.variables['_Point'] = 0.00001
            self.variables['_Digits'] = 5
            self._pip_value = self._contract_size * 0.0001  # $10 per pip per standard lot
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
        self.variables['Bars'] = len(bars)
        self.variables['INIT_SUCCEEDED'] = 0

        # Load MQL5 constants
        self.variables.update(MQL5_CONSTANTS)

        # Execute global statements (var declarations, class instantiations)
        for stmt in self._global_stmts:
            self._exec(stmt)

        # Execute OnInit
        if self._on_init:
            try:
                for stmt in self._on_init:
                    self._exec(stmt)
            except FuncReturn:
                pass

        # Build strategy config from inputs
        self.strategy_config = {
            'title': 'MQL5 Expert Advisor',
            'overlay': True,
        }

    def setup_secondary(self, bars):
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

    def run_bar(self, idx):
        self.bar_index = idx
        self._bar_cache = {}
        self._max_exec_count = 0
        self.signals = []

        row = self._row_cache[idx]
        for col in ('open', 'high', 'low', 'close', 'volume'):
            self.series_data[col].append(row[col])

        # Feed SMT data
        if self._smt_row_cache and idx < len(self._smt_row_cache):
            smt_row = self._smt_row_cache[idx]
            for col in ('open', 'high', 'low', 'close', 'volume'):
                self._smt_series[col].append(smt_row[col])

        o = row['open']; h = row['high']; l = row['low']; c = row['close']
        v = self.variables
        v['bar_index'] = idx
        v['close'] = c; v['open'] = o; v['high'] = h; v['low'] = l
        v['volume'] = row['volume']
        v['Ask'] = c; v['Bid'] = c  # simplified: no spread

        # Execute OnTick
        if self._on_tick:
            try:
                for stmt in self._on_tick:
                    self._exec(stmt)
            except FuncReturn:
                pass

        self.all_signals.extend([(idx, s) for s in self.signals])
        self._first_bar = False

    # ─── Execution ────────────────────────────────────────────────────────

    def _exec(self, node):
        if isinstance(node, Assignment):
            val = self._eval(node.value)
            if node.op == '=':
                self.variables[node.name] = val
            elif node.op == '+=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '+')
            elif node.op == '-=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '-')
            elif node.op == '*=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '*')
            elif node.op == '/=':
                self.variables[node.name] = self._arith(self.variables.get(node.name, 0), val, '/')

        elif isinstance(node, VarDecl):
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

        elif isinstance(node, InputDecl):
            if node.name not in self.variables:
                val = self._eval(node.value) if node.value else self._default_for_type(node.type_name)
                self.variables[node.name] = val
                self.inputs[node.name] = val

        elif isinstance(node, IfStmt):
            cond = self._eval(node.cond)
            if self._truthy(cond):
                for stmt in node.body: self._exec(stmt)
            elif node.else_body:
                for stmt in node.else_body: self._exec(stmt)

        elif isinstance(node, ForStmt):
            if node.init:
                self._exec(node.init) if isinstance(node.init, (Assignment, VarDecl, ExprStmt)) else self._eval(node.init)
            max_iter = 100000
            count = 0
            while count < max_iter:
                if node.cond:
                    if not self._truthy(self._eval(node.cond)):
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

        elif isinstance(node, WhileStmt):
            max_iter = 100000
            count = 0
            while count < max_iter and self._truthy(self._eval(node.cond)):
                try:
                    for stmt in node.body:
                        self._exec(stmt)
                except LoopContinue:
                    continue
                except LoopBreak:
                    break
                count += 1

        elif isinstance(node, DoWhileStmt):
            max_iter = 100000
            count = 0
            while count < max_iter:
                try:
                    for stmt in node.body:
                        self._exec(stmt)
                except LoopContinue:
                    pass
                except LoopBreak:
                    break
                if not self._truthy(self._eval(node.cond)):
                    break
                count += 1

        elif isinstance(node, SwitchStmt):
            val = self._eval(node.expr)
            matched = False
            for cond, body in node.cases:
                cv = self._eval(cond)
                if matched or val == cv:
                    matched = True
                    try:
                        for stmt in body:
                            self._exec(stmt)
                    except LoopBreak:
                        matched = False; break

            if not matched and node.default:
                try:
                    for stmt in node.default:
                        self._exec(stmt)
                except LoopBreak:
                    pass

        elif isinstance(node, ReturnStmt):
            val = self._eval(node.value) if node.value else None
            raise FuncReturn(val)

        elif isinstance(node, ContinueStmt):
            raise LoopContinue()
        elif isinstance(node, BreakStmt):
            raise LoopBreak()
        elif isinstance(node, ExprStmt):
            self._eval(node.expr)
        elif isinstance(node, FuncDef):
            self.user_functions[node.name] = node
        elif isinstance(node, PreprocessorDir):
            pass  # Already handled in prescan
        elif isinstance(node, EnumDecl):
            for vname, vval in node.values.items():
                self.variables[vname] = vval

    # ─── Evaluation ───────────────────────────────────────────────────────

    def _eval(self, node) -> Any:
        self._max_exec_count += 1
        if self._max_exec_count > self._max_exec_limit:
            raise RuntimeError(f'Execution limit exceeded on bar {self.bar_index} — possible infinite loop')
        if node is None: return NA
        if isinstance(node, NumberLit): return node.value
        if isinstance(node, StringLit): return node.value
        if isinstance(node, BoolLit): return node.value
        if isinstance(node, NullLit): return NA
        if isinstance(node, ArrayLit): return [self._eval(e) for e in node.elements]

        if isinstance(node, Identifier):
            name = node.name
            if name in self.variables: return self.variables[name]
            if name in MQL5_CONSTANTS: return MQL5_CONSTANTS[name]
            if name in self.series_data: return self.series_data[name].get(0)
            return NA

        if isinstance(node, BinOp): return self._eval_binop(node)

        if isinstance(node, UnaryOp):
            if node.op == '++pre':
                name = node.operand.name if isinstance(node.operand, Identifier) else None
                if name:
                    self.variables[name] = self.variables.get(name, 0) + 1
                    return self.variables[name]
                return NA
            if node.op == '--pre':
                name = node.operand.name if isinstance(node.operand, Identifier) else None
                if name:
                    self.variables[name] = self.variables.get(name, 0) - 1
                    return self.variables[name]
                return NA
            val = self._eval(node.operand)
            if node.op == '-': return -val if not is_na(val) else NA
            if node.op == '!': return not self._truthy(val)
            return val

        if isinstance(node, PostfixOp):
            name = node.operand.name if isinstance(node.operand, Identifier) else None
            if name:
                old = self.variables.get(name, 0)
                if node.op == '++':
                    self.variables[name] = old + 1
                elif node.op == '--':
                    self.variables[name] = old - 1
                return old  # postfix returns old value
            return NA

        if isinstance(node, Ternary):
            return self._eval(node.true_val) if self._truthy(self._eval(node.cond)) else self._eval(node.false_val)

        if isinstance(node, ArrayAccess):
            return self._eval_array_access(node)

        if isinstance(node, CastExpr):
            val = self._eval(node.expr)
            if is_na(val): return NA
            if node.target_type in ('int', 'long', 'short', 'char', 'uchar', 'ushort', 'uint', 'ulong'):
                return int(val)
            if node.target_type in ('double', 'float'):
                return float(val)
            if node.target_type == 'string':
                return str(val)
            if node.target_type == 'bool':
                return bool(val)
            return val

        if isinstance(node, FuncCall): return self._call(node)

        if isinstance(node, DotAccess):
            full = self._dot_name(node)
            if full:
                if full in self.variables: return self.variables[full]
                if full in MQL5_CONSTANTS: return MQL5_CONSTANTS[full]
            return NA

        if isinstance(node, IfStmt):
            # If used as expression
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

        return NA

    def _eval_array_access(self, node):
        """Handle Close[0], buffer[i], etc."""
        name = self._node_name(node.expr)
        index = self._eval(node.index)
        if is_na(index): return NA
        index = int(index)

        # Built-in timeseries arrays: Close, Open, High, Low, Volume
        ts_map = {'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume',
                   'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'}
        if name in ts_map:
            series = self.series_data.get(ts_map[name])
            if series: return series.get(index)
            return NA

        # Regular array variable
        arr = self.variables.get(name) if name else self._eval(node.expr)
        if isinstance(arr, list):
            # Check if array is set as series (reversed indexing)
            is_series = self._array_series_flags.get(id(arr), False)
            if is_series:
                idx = len(arr) - 1 - index
            else:
                idx = index
            if 0 <= idx < len(arr): return arr[idx]
            return NA

        return NA

    def _dot_name(self, node):
        if isinstance(node, DotAccess):
            p = self._dot_name(node.obj)
            return p + '.' + node.attr if p else None
        if isinstance(node, Identifier): return node.name
        return None

    def _node_name(self, node):
        if isinstance(node, Identifier): return node.name
        if isinstance(node, DotAccess): return self._dot_name(node)
        return None

    def _eval_binop(self, node):
        left = self._eval(node.left)
        right = self._eval(node.right)
        op = node.op
        if op in ('&&', 'and'): return self._truthy(left) and self._truthy(right)
        if op in ('||', 'or'): return self._truthy(left) or self._truthy(right)
        if op in ('==', '!='):
            if is_na(left) and is_na(right): return op == '=='
            if is_na(left) or is_na(right): return op == '!='
            return (left == right) if op == '==' else (left != right)
        if is_na(left) or is_na(right): return NA
        return self._arith(left, right, op)

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

        # ── CTrade methods — match any object.Buy/Sell/etc. ──
        short_name = name.split('.')[-1] if '.' in name else name

        if short_name == 'Buy':
            lots = args[0] if args else 1.0
            if is_na(lots) or lots <= 0: lots = 1.0
            # Convert lots to units (lots * contract_size) for P&L calculation
            units = float(lots) * self._contract_size
            comment = args[5] if len(args) > 5 else ''
            sl = args[3] if len(args) > 3 and not is_na(args[3]) and args[3] != 0 else None
            tp = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
            self.signals.append(Signal('entry', 'long', units, str(comment)))
            if sl or tp:
                self.signals.append(Signal('exit', 'long', comment=str(comment), stop=sl, limit=tp, from_entry=str(comment)))
            return True

        if short_name == 'Sell':
            lots = args[0] if args else 1.0
            if is_na(lots) or lots <= 0: lots = 1.0
            units = float(lots) * self._contract_size
            comment = args[5] if len(args) > 5 else ''
            sl = args[3] if len(args) > 3 and not is_na(args[3]) and args[3] != 0 else None
            tp = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
            self.signals.append(Signal('entry', 'short', units, str(comment)))
            if sl or tp:
                self.signals.append(Signal('exit', 'short', comment=str(comment), stop=sl, limit=tp, from_entry=str(comment)))
            return True

        if short_name == 'BuyLimit':
            # BuyLimit(lots, price, symbol, sl, tp, comment)
            lots = args[0] if args else 1.0
            if is_na(lots) or lots <= 0: lots = 1.0
            units = float(lots) * self._contract_size
            price = args[1] if len(args) > 1 else 0
            comment = args[5] if len(args) > 5 else ''
            sl = args[3] if len(args) > 3 and not is_na(args[3]) and args[3] != 0 else None
            tp = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
            self.signals.append(Signal('entry', 'long', units, str(comment), order_type='limit', entry_price=price))
            return True

        if short_name == 'BuyStop':
            lots = args[0] if args else 1.0
            if is_na(lots) or lots <= 0: lots = 1.0
            units = float(lots) * self._contract_size
            price = args[1] if len(args) > 1 else 0
            comment = args[5] if len(args) > 5 else ''
            sl = args[3] if len(args) > 3 and not is_na(args[3]) and args[3] != 0 else None
            tp = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
            self.signals.append(Signal('entry', 'long', units, str(comment), order_type='stop', entry_price=price))
            return True

        if short_name == 'SellLimit':
            lots = args[0] if args else 1.0
            if is_na(lots) or lots <= 0: lots = 1.0
            units = float(lots) * self._contract_size
            price = args[1] if len(args) > 1 else 0
            comment = args[5] if len(args) > 5 else ''
            sl = args[3] if len(args) > 3 and not is_na(args[3]) and args[3] != 0 else None
            tp = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
            self.signals.append(Signal('entry', 'short', units, str(comment), order_type='limit', entry_price=price))
            return True

        if short_name == 'SellStop':
            lots = args[0] if args else 1.0
            if is_na(lots) or lots <= 0: lots = 1.0
            units = float(lots) * self._contract_size
            price = args[1] if len(args) > 1 else 0
            comment = args[5] if len(args) > 5 else ''
            sl = args[3] if len(args) > 3 and not is_na(args[3]) and args[3] != 0 else None
            tp = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
            self.signals.append(Signal('entry', 'short', units, str(comment), order_type='stop', entry_price=price))
            return True

        if short_name == 'PositionOpen':
            if len(args) >= 3:
                order_type = args[1]
                vol = args[2]
                price = args[3] if len(args) > 3 else 0
                sl = args[4] if len(args) > 4 and not is_na(args[4]) and args[4] != 0 else None
                tp = args[5] if len(args) > 5 and not is_na(args[5]) and args[5] != 0 else None
                comment = args[6] if len(args) > 6 else ''
                # ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, BUY_LIMIT=2, SELL_LIMIT=3, BUY_STOP=4, SELL_STOP=5
                if order_type in (0, 1):
                    direction = 'long' if order_type == 0 else 'short'
                    self.signals.append(Signal('entry', direction, vol, str(comment)))
                    if sl or tp:
                        self.signals.append(Signal('exit', direction, comment=str(comment), stop=sl, limit=tp, from_entry=str(comment)))
                elif order_type in (2, 3):
                    direction = 'long' if order_type == 2 else 'short'
                    self.signals.append(Signal('entry', direction, vol, str(comment), order_type='limit', entry_price=price))
                elif order_type in (4, 5):
                    direction = 'long' if order_type == 4 else 'short'
                    self.signals.append(Signal('entry', direction, vol, str(comment), order_type='stop', entry_price=price))
            return True

        if short_name in ('PositionClose', 'PositionCloseBy'):
            self.signals.append(Signal('close', 'all'))
            return True

        if short_name == 'PositionModify':
            sl = args[1] if len(args) > 1 and not is_na(args[1]) and args[1] != 0 else None
            tp = args[2] if len(args) > 2 and not is_na(args[2]) and args[2] != 0 else None
            direction = 'long' if self._position_type == 0 else 'short'
            self.signals.append(Signal('exit', direction, comment='modify', stop=sl, limit=tp))
            return True

        # CTrade no-ops
        if short_name.startswith('Set') and '.' in name or short_name.startswith('Request') or short_name.startswith('Check') or short_name.startswith('Result'):
            return True

        # ── Position queries ──
        if name == 'PositionSelect':
            return self._position_type >= 0

        if name == 'PositionGetDouble':
            prop = args[0] if args else 0
            if prop == 100: return self._position_price_open  # POSITION_PRICE_OPEN
            if prop == 101: return self._position_sl
            if prop == 102: return self._position_tp
            if prop == 103: return self._position_profit
            if prop == 104: return self._position_volume
            if prop == 105: return self.variables.get('close', 0)  # POSITION_PRICE_CURRENT
            return 0.0

        if name == 'PositionGetInteger':
            prop = args[0] if args else 0
            if prop == 106: return self._position_type  # POSITION_TYPE
            return 0

        if name == 'PositionGetString':
            prop = args[0] if args else 0
            if prop == 200: return 'SYMBOL'  # POSITION_SYMBOL — always matches
            return ''

        if name == 'PositionsTotal':
            return 1 if self._position_type >= 0 else 0

        if name == 'PositionGetTicket':
            return 1 if self._position_type >= 0 else 0

        # ── OrderSend ──
        if name == 'OrderSend':
            # Simplified: args[0] would be MqlTradeRequest struct — not practical to fully simulate
            # Just return true
            return True

        # ── Symbol info ──
        if name == 'SymbolInfoDouble':
            prop = args[1] if len(args) > 1 else 0
            if prop == 200: return self.variables.get('Bid', 0)  # SYMBOL_BID
            if prop == 201: return self.variables.get('Ask', 0)  # SYMBOL_ASK
            if prop == 202: return self.variables.get('_Point', 0.00001)  # SYMBOL_POINT
            if prop == 205:  # SYMBOL_TRADE_TICK_VALUE
                profit_ccy = self.variables.get('__profit_ccy__', 'USD')
                price = self.variables.get('close', 1.0)
                if profit_ccy == 'USD':
                    # USD-denominated: $10 per pip per standard lot
                    return self._contract_size * 0.0001  # 100000 * 0.0001 = $10
                elif profit_ccy == 'JPY':
                    # JPY pairs: convert via price
                    return self._contract_size * 0.01 / price if price > 0 else 1.0
                else:
                    # Cross pairs (CHF, CAD, GBP, etc.): tick value = contract * tick_size / price
                    # For NZDCHF at 0.46: 100000 * 0.00001 / 0.46 ~ $2.17 per pip
                    tick_size = 0.001 if self._is_jpy_pair else 0.00001
                    return self._contract_size * tick_size / price if price > 0 else 1.0
            if prop == 206: return self._contract_size  # SYMBOL_TRADE_CONTRACT_SIZE
            if prop == 207: return 0.01  # SYMBOL_VOLUME_MIN
            if prop == 208: return 100.0  # SYMBOL_VOLUME_MAX
            if prop == 209: return 0.01  # SYMBOL_VOLUME_STEP
            if prop == 210:  # SYMBOL_TRADE_TICK_SIZE
                return 0.001 if self._is_jpy_pair else 0.00001
            return 0.0

        if name == 'SymbolInfoInteger':
            prop = args[1] if len(args) > 1 else 0
            if prop == 203: return self.variables.get('_Digits', 5)  # SYMBOL_DIGITS
            return 0

        if name == 'SymbolInfoString':
            prop = args[1] if len(args) > 1 else 0
            if prop == 300:  # SYMBOL_CURRENCY_PROFIT — last 3 chars of symbol
                return self.variables.get('__profit_ccy__', 'USD')
            if prop == 301:  # SYMBOL_CURRENCY_BASE — first 3 chars of symbol
                return self.variables.get('__base_ccy__', 'NZD')
            return ''

        if name == 'SymbolInfoTick':
            return True

        if name == 'PeriodSeconds':
            tf = int(args[0]) if args else 0
            # Convert timeframe enum to seconds
            tf_map = {0: 300, 1: 60, 5: 300, 15: 900, 30: 1800, 60: 3600,
                      240: 14400, 1440: 86400, 10080: 604800, 43200: 2592000}
            return tf_map.get(tf, 300)

        if name == 'TimeToStruct':
            # Fill MqlDateTime struct fields from timestamp
            # args[0] = datetime (Unix timestamp), args[1] = struct variable (passed by ref)
            timestamp = int(args[0]) if args and not is_na(args[0]) else 0
            if timestamp > 0:
                import datetime as _dt
                try:
                    # Apply broker UTC offset
                    utc_off = int(self.variables.get('__utc_offset__', 0))
                    t = _dt.datetime.utcfromtimestamp(timestamp + utc_off * 3600)
                    # Find the struct variable name from the AST node
                    # Since MQL5 passes by reference, we need to set fields like "dt.hour", "dt.min"
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

        # ── Account info ──
        if name == 'AccountInfoDouble':
            prop = args[0] if args else 0
            if prop == 300: return self._account_balance
            if prop == 301: return self._account_equity
            if prop == 302: return self._position_profit
            return 0.0

        # ── Indicator handles ──
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

        # ── CopyBuffer ──
        if name == 'CopyBuffer':
            return self._copy_buffer(args)

        # ── Array functions ──
        if name == 'ArrayResize':
            arr = args[0] if args else []
            new_size = int(args[1]) if len(args) > 1 else 0
            if isinstance(arr, list):
                while len(arr) < new_size: arr.append(0.0)
                while len(arr) > new_size: arr.pop()
            return len(arr) if isinstance(arr, list) else 0

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
            dst = args[0] if args else []
            src = args[1] if len(args) > 1 else []
            dst_start = int(args[2]) if len(args) > 2 else 0
            src_start = int(args[3]) if len(args) > 3 else 0
            count = int(args[4]) if len(args) > 4 else len(src) - src_start
            if isinstance(dst, list) and isinstance(src, list):
                while len(dst) < dst_start + count:
                    dst.append(0.0)
                for i in range(count):
                    if src_start + i < len(src):
                        dst[dst_start + i] = src[src_start + i]
            return count

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

        # ── iClose, iOpen, iHigh, iLow, iVolume ──
        if name in ('iClose', 'iOpen', 'iHigh', 'iLow', 'iVolume'):
            symbol = str(args[0]) if args else 'SYMBOL'
            tf = int(args[1]) if len(args) > 1 else 0
            shift = int(args[2]) if len(args) > 2 else 0
            col_map = {'iClose': 'close', 'iOpen': 'open', 'iHigh': 'high', 'iLow': 'low', 'iVolume': 'volume'}
            col = col_map[name]
            # Route to SMT data if symbol matches SMT symbol
            if symbol != 'SYMBOL' and symbol != self.variables.get('_Symbol', 'SYMBOL') and self._smt_series:
                series = self._smt_series.get(col)
                if series: return series.get(shift)
                return NA
            # Use MTF engine for higher timeframes
            if tf > 0 and self.variables.get('__mtf__') is not None:
                tf_str = self.variables.get('__mtf__').resolve_tf(tf)
                if tf_str and self.variables.get('__mtf__').has_tf(tf_str):
                    val = self.variables.get('__mtf__').get(tf_str, col, self.bar_index, shift)
                    return val if val is not None else NA
            # Base timeframe: use series directly
            series = self.series_data.get(col)
            if series is None: return NA
            return series.get(shift)
        if name == 'iTime':
            tf = int(args[1]) if len(args) > 1 else 0
            shift = int(args[2]) if len(args) > 2 else 0
            effective_bar = max(0, self.bar_index - shift)
            # For HTF: use MTF engine for proper bar boundaries
            if tf > 0 and self.variables.get('__mtf__') is not None:
                tf_str = self.variables.get('__mtf__').resolve_tf(tf)
                if tf_str and self.variables.get('__mtf__').has_tf(tf_str):
                    return self.variables.get('__mtf__').get_htf_time(tf_str, effective_bar)
            # Base TF: use real timestamp from CSV data
            if effective_bar < len(self._timestamps):
                timestamp = self._timestamps[effective_bar]
            else:
                timestamp = self._timestamps[-1] if self._timestamps else 0
            if tf > 0:
                htf_seconds = tf * 60
                timestamp = (timestamp // htf_seconds) * htf_seconds
            return timestamp
        if name == 'iBarShift':
            return 0

        # ── Math functions ──
        if name in ('MathSqrt', 'sqrt'):
            return math.sqrt(args[0]) if args and not is_na(args[0]) and args[0] >= 0 else NA
        if name in ('MathAbs', 'fabs'):
            return abs(args[0]) if args and not is_na(args[0]) else NA
        if name in ('MathMax', 'fmax'):
            if len(args) >= 2 and not is_na(args[0]) and not is_na(args[1]):
                return max(float(args[0]), float(args[1]))
            return NA
        if name in ('MathMin', 'fmin'):
            if len(args) >= 2 and not is_na(args[0]) and not is_na(args[1]):
                return min(float(args[0]), float(args[1]))
            return NA
        if name in ('MathPow', 'pow'):
            return math.pow(args[0], args[1]) if len(args) >= 2 and not is_na(args[0]) and not is_na(args[1]) else NA
        if name in ('MathLog', 'log'):
            return math.log(args[0]) if args and not is_na(args[0]) and args[0] > 0 else NA
        if name in ('MathLog10',):
            return math.log10(args[0]) if args and not is_na(args[0]) and args[0] > 0 else NA
        if name in ('MathExp', 'exp'):
            return math.exp(args[0]) if args and not is_na(args[0]) else NA
        if name in ('MathFloor', 'floor'):
            return math.floor(args[0]) if args and not is_na(args[0]) else NA
        if name in ('MathCeil', 'ceil'):
            return math.ceil(args[0]) if args and not is_na(args[0]) else NA
        if name in ('MathRound', 'round'):
            return round(args[0]) if args and not is_na(args[0]) else NA
        if name == 'NormalizeDouble':
            if len(args) >= 2 and not is_na(args[0]):
                return round(float(args[0]), int(args[1]))
            return NA
        if name in ('MathSin', 'sin'):
            return math.sin(args[0]) if args and not is_na(args[0]) else NA
        if name in ('MathCos', 'cos'):
            return math.cos(args[0]) if args and not is_na(args[0]) else NA
        if name in ('MathTan', 'tan'):
            return math.tan(args[0]) if args and not is_na(args[0]) else NA

        # ── String functions ──
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
            return str(int(args[0])) if args and not is_na(args[0]) else ''
        if name == 'DoubleToString':
            if args and not is_na(args[0]):
                digits = int(args[1]) if len(args) > 1 else 8
                return f'{float(args[0]):.{digits}f}'
            return ''
        if name == 'StringConcatenate':
            return ''.join(str(a) for a in args)

        # ── String formatting ──
        if name == 'StringFormat':
            if args:
                fmt = str(args[0])
                # Simple printf-style formatting
                try:
                    return fmt % tuple(args[1:]) if len(args) > 1 else fmt
                except (TypeError, ValueError):
                    return fmt
            return ''

        # ── Output ──
        if name in ('Print', 'Comment', 'Alert', 'PrintFormat', 'printf'):
            return None  # no-op

        # ── VWAP ──
        if name in ('ta.vwap', 'vwap', 'VWAP'):
            source = args[0] if args else self.variables.get('close', 0)
            current_vol = self.variables.get('volume', 0)
            if is_na(source) or is_na(current_vol): return NA
            current_price = float(source)
            current_vol = float(current_vol)
            day_key = ''
            if self.bar_index < len(self._timestamps):
                ts = self._timestamps[self.bar_index]
                try:
                    import datetime as _dt
                    utc_off = int(self.variables.get('__utc_offset__', 0))
                    t = _dt.datetime.utcfromtimestamp(ts + utc_off * 3600)
                    day_key = t.strftime('%Y-%m-%d')
                except Exception:
                    day_key = str(self.bar_index // 288)
            else:
                day_key = str(self.bar_index // 288)
            prev_day = self.variables.get('_vwap_day', '')
            if prev_day != day_key:
                self.variables['_vwap_num'] = 0.0
                self.variables['_vwap_den'] = 0.0
                self.variables['_vwap_day'] = day_key
            self.variables['_vwap_num'] = self.variables.get('_vwap_num', 0.0) + current_price * current_vol
            self.variables['_vwap_den'] = self.variables.get('_vwap_den', 0.0) + current_vol
            den = self.variables['_vwap_den']
            return self.variables['_vwap_num'] / den if den > 0 else NA

        # ── Platform SMT Divergence (any script can use these) ──
        if name == 'SMTDivergence' or name == 'IsSMTDivergence':
            # SMTDivergence(1) = bullish, SMTDivergence(-1) = bearish, SMTDivergence(0) = any
            direction = int(args[0]) if args else 0
            if direction == 1:
                return self.variables.get('_smt_bull_active', False)
            elif direction == -1:
                return self.variables.get('_smt_bear_active', False)
            else:
                return self.variables.get('_smt_bull_active', False) or self.variables.get('_smt_bear_active', False)

        if name == 'IsSMTAvailable':
            return self.variables.get('_smt_available', False)

        # ── Bars/time ──
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


        # ── Indicator release ──
        if name == 'IndicatorRelease':
            handle = int(args[0]) if args else 0
            self._indicator_handles.pop(handle, None)
            return True

        # ── No-ops ──
        if name in ('ObjectCreate', 'ObjectDelete', 'ObjectSetInteger', 'ObjectSetDouble',
                     'ObjectSetString', 'ObjectFind', 'ChartRedraw', 'ChartSetInteger',
                     'ChartIndicatorAdd', 'EventSetTimer', 'EventKillTimer',
                     'GlobalVariableSet', 'GlobalVariableGet', 'GlobalVariableCheck',
                     'Sleep', 'GetLastError', 'ResetLastError',
                     'MarketInfo', 'RefreshRates',
                     'StringToTime', 'TimeToString', 'TimeToStruct',
                     'MathIsValidNumber', 'MathRand', 'MathSrand'):
            return 0

        if name == '__array_set__':
            # Internal: array element assignment
            arr_name = self._node_name(node.args[0]) if len(node.args) > 0 else None
            arr = args[0] if args else []
            idx = int(args[1]) if len(args) > 1 else 0
            val = args[2] if len(args) > 2 else NA
            if isinstance(arr, list):
                is_series = self._array_series_flags.get(id(arr), False)
                if is_series:
                    actual_idx = len(arr) - 1 - idx
                else:
                    actual_idx = idx
                if 0 <= actual_idx < len(arr):
                    arr[actual_idx] = val
            return None

        # ── User-defined functions ──
        if name in self.user_functions:
            fdef = self.user_functions[name]
            saved = {}
            for i, (pname, pdefault) in enumerate(fdef.params):
                saved[pname] = self.variables.get(pname)
                if i < len(args):
                    self.variables[pname] = args[i]
                elif pdefault is not None:
                    self.variables[pname] = self._eval(pdefault)
                else:
                    self.variables[pname] = NA
            result = NA
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

        return NA

    # ─── Indicator Support ────────────────────────────────────────────────

    def _create_indicator(self, ind_type, args):
        """Create an indicator handle. Returns an integer handle ID."""
        handle = self._next_handle
        self._next_handle += 1
        self._indicator_handles[handle] = {
            'type': ind_type,
            'args': args,
            'cache': {},  # bar_index -> computed values per buffer
        }
        return handle

    def _copy_buffer(self, args):
        """CopyBuffer(handle, buffer_index, start_pos, count, array)"""
        if len(args) < 5: return -1
        handle = int(args[0])
        buffer_idx = int(args[1])
        start_pos = int(args[2])
        count = int(args[3])
        arr = args[4]
        if not isinstance(arr, list): return -1

        ind = self._indicator_handles.get(handle)
        if ind is None: return -1

        # Compute indicator values for requested range (with per-bar cache)
        cb_cache_key = (handle, buffer_idx, start_pos, count)
        cached = self._bar_cache.get(cb_cache_key)
        if cached is not None:
            values = cached
        else:
            values = []
            for shift in range(start_pos, start_pos + count):
                val = self._compute_indicator(ind, buffer_idx, shift)
                values.append(val)
            self._bar_cache[cb_cache_key] = values

        # Copy to array (reversed if array is set as series)
        is_series = self._array_series_flags.get(id(arr), False)
        while len(arr) < count: arr.append(0.0)

        if is_series:
            for i, v in enumerate(values):
                arr[i] = v if not is_na(v) else 0.0
        else:
            for i, v in enumerate(values):
                arr[i] = v if not is_na(v) else 0.0

        return count

    def _compute_indicator(self, ind, buffer_idx, shift):
        """Compute a single indicator value for the given shift from current bar."""
        ind_type = ind['type']
        ind_args = ind['args']

        if ind_type == 'MA':
            # iMA(symbol, timeframe, period, ma_shift, method, price)
            period = int(ind_args[2]) if len(ind_args) > 2 else 14
            method = int(ind_args[4]) if len(ind_args) > 4 else 0  # MODE_SMA
            price_type = int(ind_args[5]) if len(ind_args) > 5 else 0  # PRICE_CLOSE
            source_name = self._price_type_to_series(price_type)

            if method == 0:  # SMA
                return self._calc_sma(source_name, period, shift)
            elif method == 1:  # EMA
                return self._calc_ema(source_name, period, shift)
            elif method == 2:  # SMMA
                return self._calc_sma(source_name, period, shift)  # approximate
            elif method == 3:  # LWMA
                return self._calc_wma(source_name, period, shift)
            return NA

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
            if buffer_idx == 0:  # MACD main
                fast_ema = self._calc_ema(source_name, fast, shift)
                slow_ema = self._calc_ema(source_name, slow, shift)
                if is_na(fast_ema) or is_na(slow_ema): return NA
                return fast_ema - slow_ema
            return NA  # signal/histogram not implemented

        if ind_type == 'ATR':
            period = int(ind_args[2]) if len(ind_args) > 2 else 14
            return self._calc_atr(period, shift)

        if ind_type == 'BB':
            period = int(ind_args[2]) if len(ind_args) > 2 else 20
            deviation = float(ind_args[4]) if len(ind_args) > 4 else 2.0
            price_type = int(ind_args[5]) if len(ind_args) > 5 else 0
            source_name = self._price_type_to_series(price_type)
            basis = self._calc_sma(source_name, period, shift)
            if is_na(basis): return NA
            values = self._get_history(source_name, period, shift)
            valid = [v for v in values if v is not None]
            if len(valid) < period: return NA
            std = math.sqrt(sum((v - basis) ** 2 for v in valid) / len(valid))
            if buffer_idx == 0: return basis  # middle
            if buffer_idx == 1: return basis + deviation * std  # upper
            if buffer_idx == 2: return basis - deviation * std  # lower
            return NA

        if ind_type == 'STOCH':
            k_period = int(ind_args[2]) if len(ind_args) > 2 else 14
            highs = self._get_history('high', k_period, shift)
            lows = self._get_history('low', k_period, shift)
            vh = [v for v in highs if v is not None]
            vl = [v for v in lows if v is not None]
            if len(vh) < k_period or len(vl) < k_period: return NA
            highest = max(vh); lowest = min(vl)
            close_val = self.series_data['close'].get(shift)
            if is_na(close_val): return NA
            if highest == lowest: return NA
            return 100.0 * (close_val - lowest) / (highest - lowest)

        return NA

    def _price_type_to_series(self, price_type):
        mapping = {0: 'close', 1: 'open', 2: 'high', 3: 'low',
                   4: 'close', 5: 'close', 6: 'close'}
        return mapping.get(price_type, 'close')

    def _get_history(self, source_name, length, extra_shift=0):
        """Get history values, optionally shifted further back."""
        series = self.series_data.get(source_name)
        if series is None or len(series) == 0: return []
        values = []
        for i in range(length - 1, -1, -1):  # oldest to newest
            v = series.get(i + extra_shift)
            values.append(None if is_na(v) else float(v))
        return values

    def _calc_sma(self, source_name, period, shift=0):
        cache_key = ('sma', source_name, period, shift)
        if cache_key in self._bar_cache: return self._bar_cache[cache_key]
        values = self._get_history(source_name, period, shift)
        valid = [v for v in values if v is not None]
        result = sum(valid) / len(valid) if len(valid) >= period else NA
        self._bar_cache[cache_key] = result
        return result

    def _calc_ema(self, source_name, period, shift=0):
        cache_key = f'_ema_{source_name}_{period}'
        series = self.series_data.get(source_name)
        if series is None: return NA
        current_val = series.get(shift)
        if is_na(current_val): return NA

        if shift > 0:
            # For shifted values, compute from scratch
            values = self._get_history(source_name, period + shift, 0)
            valid = [v for v in values if v is not None]
            if len(valid) == 0: return NA
            # Simple fallback for shifted EMA
            return sum(valid[:period]) / min(len(valid), period)

        prev_ema = self._ema_cache.get(cache_key, NA)
        if is_na(prev_ema):
            values = self._get_history(source_name, period, shift)
            valid = [v for v in values if v is not None]
            if len(valid) == 0: return NA
            ema = sum(valid) / len(valid)
        else:
            k = 2.0 / (period + 1)
            ema = float(current_val) * k + prev_ema * (1 - k)
        self._ema_cache[cache_key] = ema
        return ema

    def _calc_wma(self, source_name, period, shift=0):
        values = self._get_history(source_name, period, shift)
        valid = [v for v in values if v is not None]
        if len(valid) < period: return NA
        ws = sum(range(1, period + 1))
        return sum(v * (i + 1) for i, v in enumerate(valid)) / ws

    def _calc_rsi(self, source_name, period, shift=0):
        if shift > 0:
            # For shifted RSI, compute from scratch (no smoothing state)
            values = self._get_history(source_name, period + 1, shift)
            valid = [v for v in values if v is not None]
            if len(valid) < period + 1: return NA
            gains = [max(valid[i] - valid[i-1], 0) for i in range(1, len(valid))]
            losses = [max(valid[i-1] - valid[i], 0) for i in range(1, len(valid))]
            ag = sum(gains) / period; al = sum(losses) / period
            if al == 0: return 100.0
            return 100.0 - (100.0 / (1.0 + ag / al))

        series = self.series_data.get(source_name)
        if series is None or len(series) < 2: return NA
        current = series.get(0); prev = series.get(1)
        if is_na(current) or is_na(prev): return NA

        change = float(current) - float(prev)
        gain = max(change, 0); loss = max(-change, 0)

        gain_key = f'_rsi_gain_{source_name}_{period}'
        loss_key = f'_rsi_loss_{source_name}_{period}'
        prev_ag = self._ema_cache.get(gain_key, NA)
        prev_al = self._ema_cache.get(loss_key, NA)

        if is_na(prev_ag):
            values = self._get_history(source_name, period + 1, 0)
            valid = [v for v in values if v is not None]
            if len(valid) < period + 1: return NA
            gains = [max(valid[i] - valid[i-1], 0) for i in range(1, len(valid))]
            losses = [max(valid[i-1] - valid[i], 0) for i in range(1, len(valid))]
            ag = sum(gains) / period; al = sum(losses) / period
        else:
            ag = (prev_ag * (period - 1) + gain) / period
            al = (prev_al * (period - 1) + loss) / period

        self._ema_cache[gain_key] = ag
        self._ema_cache[loss_key] = al

        if al == 0: return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    def _calc_atr(self, period, shift=0):
        if shift > 0:
            # For shifted ATR, simple average (no state)
            tr_values = []
            for i in range(period):
                h = self.series_data['high'].get(i + shift)
                l = self.series_data['low'].get(i + shift)
                c_prev = self.series_data['close'].get(i + shift + 1)
                if is_na(h) or is_na(l): return NA
                if not is_na(c_prev):
                    tr_values.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
                else:
                    tr_values.append(h - l)
            return sum(tr_values) / len(tr_values) if tr_values else NA

        h = self.series_data['high'].get(0)
        l = self.series_data['low'].get(0)
        c_prev = self.series_data['close'].get(1)
        if is_na(h) or is_na(l): return NA
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev)) if not is_na(c_prev) else h - l

        atr_key = f'_atr_{period}'
        prev_atr = self._ema_cache.get(atr_key, NA)

        if is_na(prev_atr):
            tr_values = []
            for i in range(period):
                hi = self.series_data['high'].get(i)
                lo = self.series_data['low'].get(i)
                cp = self.series_data['close'].get(i + 1)
                if is_na(hi) or is_na(lo): return NA
                tr_values.append(max(hi - lo, abs(hi - cp), abs(lo - cp)) if not is_na(cp) else hi - lo)
            atr = sum(tr_values) / len(tr_values) if tr_values else NA
        else:
            atr = (prev_atr * (period - 1) + tr) / period

        if not is_na(atr):
            self._ema_cache[atr_key] = atr
        return atr


# ─── Public API ────────────────────────────────────────────────────────────────

def parse_mql5(source: str) -> Program:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()

def create_interpreter(source: str):
    ast = parse_mql5(source)
    try:
        from mql5_fast import FastMQL5Interpreter
        return FastMQL5Interpreter(ast)
    except ImportError:
        return MQL5Interpreter(ast)
