"""
A full Python 3.14 tokenizer built with the lexer, checked against Python's own.

It produces the same tokens as the standard ``tokenize`` module: names, numbers,
strings, operators, comments, NEWLINE and NL, INDENT and DEDENT, f-strings and
t-strings (PEP 701 and PEP 750), and ENDMARKER, with the same positions.

Everything is done with grammars and hooks:

* Each indentation level is an entered grammar. INDENT enters a deeper block and
  DEDENT leaves it, so the indentation stack *is* the tokenizer's grammar stack,
  and a checkpoint undoes it with everything else.
* Brackets, f-string bodies, replacement fields and format specs are modes too.
* Lookbehinds tell NEWLINE (end of a statement) from NL (blank or comment-only
  line, or inside brackets), and keep backslash-continued lines from indenting.

Write your own code at the bottom of the file, in YOUR_CODE, and run it. The
output uses rich: ``pip install "lexmod[rich]"``.

Known limits: a tab counts as one column of indentation, where Python rounds it
up to the next multiple of 8, so mixing tabs and spaces isn't modeled. Python
3.15's changes I know of, such as the ``lazy`` soft keyword, add no new tokens:
soft keywords are plain names to a tokenizer.
"""
import functools
import io
import textwrap
import tokenize
from enum import Enum, auto

import regex
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from lexmod import Lexer

console = Console(highlight=False)

# Log every mode switch (indent, dedent, strings, fields, brackets) while lexing.
TRACE = True


class Py(Enum):
    NAME = auto()
    NUMBER = auto()
    STRING = auto()
    OP = auto()
    COMMENT = auto()
    NEWLINE = auto()
    NL = auto()
    INDENT = auto()
    DEDENT = auto()
    FSTRING_START = auto()
    FSTRING_MIDDLE = auto()
    FSTRING_END = auto()
    TSTRING_START = auto()
    TSTRING_MIDDLE = auto()
    TSTRING_END = auto()
    ENDMARKER = auto()
    ERRORTOKEN = auto()


KINDS = {
    "f": (Py.FSTRING_START, Py.FSTRING_MIDDLE, Py.FSTRING_END),
    "t": (Py.TSTRING_START, Py.TSTRING_MIDDLE, Py.TSTRING_END),
}

# Each operator is its own rule at the same priority, so the longest wins: "**="
# beats "**", which beats "*". ":" and brackets are left out, since what they do
# depends on the mode.
OPERATORS = (
    "...", "**=", "//=", ">>=", "<<=", "->", ":=", "**", "//", "<<", ">>", "<=", ">=",
    "==", "!=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "@=", "+", "-", "*",
    "/", "%", "@", "&", "|", "^", "~", "<", ">", "=", ".", ",", ";", "!",
)

NUMBER = (
    r"0[xX](?:_?[0-9a-fA-F])+|0[oO](?:_?[0-7])+|0[bB](?:_?[01])+"
    r"|(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?[jJ]?"
)

# A complete plain string, with an optional r, b, u, rb or br prefix.
STRING = (
    r"""(?i:rb|br|r|b|u)?"""
    r"""(?:'''(?:[^'\\]|\\.|'(?!''))*'''|\"\"\"(?:[^"\\]|\\.|"(?!""))*\"\"\""""
    r"""|'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")"""
)

# Only the start of an f-string or t-string: its prefix and opening quote.
STRING_START = r"""(?i:r?[ft]|[ft]r)(?:'''|\"\"\"|'|")"""

# A newline that ends a blank or comment-only line: Python calls it NL, not NEWLINE.
BLANK_NEWLINE = r"(?m)(?<=^[ \t]*(?:#[^\n]*)?)\n"

# The first character of real code on a line, so blank and comment-only lines
# never indent or dedent.
CODE_AHEAD = r"(?=[^ \t#\n])"

# The start of a line that isn't the continuation of a backslash-ended one.
LINE_START = r"(?<!\\\n)^"

STYLES = {
    Py.NAME: "cyan", Py.NUMBER: "blue", Py.STRING: "green", Py.OP: "yellow",
    Py.COMMENT: "dim italic", Py.NEWLINE: "dim", Py.NL: "dim", Py.INDENT: "bold white",
    Py.DEDENT: "bold white", Py.ENDMARKER: "dim", Py.ERRORTOKEN: "bold red",
    Py.FSTRING_START: "magenta", Py.FSTRING_MIDDLE: "green", Py.FSTRING_END: "magenta",
    Py.TSTRING_START: "bright_magenta", Py.TSTRING_MIDDLE: "bright_green", Py.TSTRING_END: "bright_magenta",
}


def trace(tokenizer, message: str, style: str = "dim") -> None:
    """Log a mode switch, indented by how deeply modes are nested."""
    if not TRACE:
        return
    line, col = tokenizer.cursor
    console.print(f"  [dim]{line:>3}:{col:<3}[/] {'  ' * tokenizer.depth}[{style}]{message}[/]")


# -- hooks -----------------------------------------------------------------------

def opener(close: str):
    """An opening bracket: emit it, then lex with that bracket's mode."""
    def hook(tokenizer, match):
        token = tokenizer.mint(match, Py.OP)
        trace(tokenizer, f"enter {match.group(0)}{close}")
        tokenizer.enter(bracket(close))
        return token
    return hook


def closer(tokenizer, match):
    """A closing bracket: emit it, then go back to the outer mode."""
    token = tokenizer.mint(match, Py.OP)
    tokenizer.leave()
    trace(tokenizer, f"leave {match.group(0)}")
    return token


def error(message: str, *, leave: bool = False):
    """An illegal rule: emit an error token and keep lexing."""
    def hook(tokenizer, match):
        trace(tokenizer, f"✗ {message}", "bold red")
        token = tokenizer.mint(match, Py.ERRORTOKEN)
        if leave:
            tokenizer.leave()
        return token
    return hook


def indent(widths: tuple[int, ...]):
    """Deeper indentation: emit INDENT for it, then lex with the deeper block."""
    def hook(tokenizer, match):
        width = len(match.group(0))
        token = tokenizer.mint(match, Py.INDENT)
        trace(tokenizer, f"indent to {width}", "bold")
        tokenizer.enter(block((*widths, width)))
        return token
    return hook


def dedent(widths: tuple[int, ...]):
    """Shallower indentation: emit one DEDENT and leave one block.

    The rule matches again in the outer block if the line is shallower still, so
    dedenting several levels takes several steps, one DEDENT each. Leaving changes
    the grammar, so these zero-width steps don't count as making no progress.
    """
    def hook(tokenizer, match):
        width = tokenizer.cursor[1] - 1
        if width not in widths:
            trace(tokenizer, "✗ unindent does not match any outer indentation level", "bold red")
            token = tokenizer.mint(match, Py.ERRORTOKEN)
            # Leave every block deeper than this line in one go, so it errors once.
            for _ in range(sum(w > width for w in widths)):
                tokenizer.leave()
            return token
        token = tokenizer.mint(match, Py.DEDENT)
        tokenizer.leave()
        trace(tokenizer, f"dedent from {widths[-1]}", "bold")
        return token
    return hook


def start_string(tokenizer, match):
    """The start of an f-string or t-string: emit it, then lex its body."""
    text = match.group(0)
    quote = text.lstrip("rRfFtT")
    prefix = text[: len(text) - len(quote)].lower()
    kind, raw = ("t" if "t" in prefix else "f"), "r" in prefix
    token = tokenizer.mint(match, KINDS[kind][0])
    trace(tokenizer, f"enter {kind}-string body, quote {quote}" + (", raw" if raw else ""), "magenta")
    tokenizer.enter(body(kind, raw, quote))
    return token


def then(token_type, action, message: str, style: str = "dim"):
    """A hook that emits its match, then does something to the tokenizer."""
    def hook(tokenizer, match):
        token = tokenizer.mint(match, token_type)
        action(tokenizer)
        trace(tokenizer, message, style)
        return token
    return hook


# -- modes -----------------------------------------------------------------------

def add_code_rules(mode: Lexer, *, newline: str | None) -> None:
    """The rules every code mode shares, with newlines handled as ``newline`` says."""
    mode.skip(r"[ \t]+")
    mode.skip(r"\\\n")  # explicit line joining
    mode.rule(r"#[^\n]*", Py.COMMENT)
    if newline == "nl":
        mode.rule(r"\n", Py.NL)
    elif newline == "skip":
        mode.skip(r"\n")
    mode.rule(r"[^\W\d]\w*", Py.NAME)
    mode.rule(NUMBER, Py.NUMBER)
    # Same priority as NAME, so for f"..." the longer string start beats the name f.
    mode.rule(STRING, Py.STRING, flags=regex.DOTALL)
    mode.rule(STRING_START, on_match=start_string)
    for op in OPERATORS:
        mode.rule(regex.escape(op), Py.OP)
    for open_, close in (("(", ")"), ("[", "]"), ("{", "}")):
        mode.rule(regex.escape(open_), on_match=opener(close))


def add_illegal_rules(mode: Lexer) -> None:
    """Error recovery shared by every code mode."""
    mode.illegal.rule(r"""['"]""", on_match=error("unterminated string"))
    mode.illegal.rule(r"\\", on_match=error("a backslash must end the line to continue it"))
    mode.illegal.rule(r"[)\]}]", on_match=error("closing bracket doesn't match an opening one"))
    mode.illegal.rule(r"[^\s]", on_match=error("invalid character"))


@functools.cache
def block(widths: tuple[int, ...]) -> Lexer:
    """Statements at one indentation level. ``widths`` is the whole indentation stack."""
    width = widths[-1]
    mode = Lexer()
    add_code_rules(mode, newline=None)
    mode.rule(":", Py.OP)
    mode.rule(BLANK_NEWLINE, Py.NL, priority=1)
    mode.rule(r"\n", Py.NEWLINE)
    # Priority 1, so they win over skipping the same whitespace.
    mode.rule(rf"(?m){LINE_START}[ \t]{{{width + 1},}}{CODE_AHEAD}", priority=1, on_match=indent(widths))
    if width:
        # Zero-width, just after the line's indentation, only when it's shallower.
        mode.rule(rf"(?m)(?<={LINE_START}[ \t]{{0,{width - 1}}}){CODE_AHEAD}", priority=1, on_match=dedent(widths))
    add_illegal_rules(mode)
    return mode


@functools.cache
def bracket(close: str) -> Lexer:
    """Inside a bracket: newlines are NL, and ":" is a slice, a dict or a lambda."""
    mode = Lexer()
    add_code_rules(mode, newline="nl")
    mode.rule(":", Py.OP)
    mode.rule(regex.escape(close), on_match=closer)
    add_illegal_rules(mode)
    return mode


def _newline_in_fields() -> str:
    """
    How this Python's tokenizer treats a newline at the top of a replacement field.

    Newer versions emit NL for it, older ones emit nothing, so ask instead of
    assuming one.
    """
    probe = 'f"""{\nx}"""\n'
    emits = any(t.type == tokenize.NL for t in tokenize.generate_tokens(io.StringIO(probe).readline))
    return "nl" if emits else "skip"


@functools.cache
def field(kind: str) -> Lexer:
    """The expression inside a replacement field, {...}."""
    mode = Lexer()
    add_code_rules(mode, newline=_newline_in_fields())
    # Above ":=", so at the top of a field ":" always starts the format spec, as in
    # Python: f"{x:=10}" formats x with "=10".
    mode.rule(":", priority=1, on_match=then(Py.OP, lambda t: t.enter(spec(kind)), "enter format spec", "blue"))
    mode.rule(r"\}", on_match=then(Py.OP, lambda t: t.leave(), "leave replacement field", "yellow"))
    add_illegal_rules(mode)
    return mode


@functools.cache
def spec(kind: str) -> Lexer:
    """A format spec, after the ":" of a replacement field. It can nest fields."""
    mode = Lexer()
    mode.rule(r"[^{}]+", KINDS[kind][1])
    mode.rule(r"\{", on_match=then(Py.OP, lambda t: t.enter(field(kind)), "enter nested replacement field", "yellow"))
    # This "}" closes the whole field, so leave both the spec and the field.
    mode.rule(r"\}", on_match=then(Py.OP, lambda t: (t.leave(), t.leave()), "leave format spec and field", "yellow"))
    return mode


@functools.cache
def body(kind: str, raw: bool, quote: str) -> Lexer:
    """The text of an f-string or t-string, for one prefix and quote."""
    _, middle, end = KINDS[kind]
    q = regex.escape(quote[0])
    parts = [r"\{\{", r"\}\}"]  # doubled braces are literal text
    if not raw:
        parts.append(r"\\N\{[^}\n]*\}")  # \N{NAME} braces are not a field
    # An escape never swallows a brace: in f"\{x}" the brace still opens a field.
    parts += [r"\\[^{}]", r"\\(?=[{}])"]
    if len(quote) == 3:
        parts += [rf"[^{{}}\\{q}]", rf"{q}(?!{q}{q})"]  # newlines and lone quotes allowed
    else:
        parts.append(rf"[^{{}}\\{q}\n]")

    mode = Lexer()
    mode.rule("(?:" + "|".join(parts) + ")+", middle)
    mode.rule(r"\{", on_match=then(Py.OP, lambda t: t.enter(field(kind)), "enter replacement field", "yellow"))
    mode.rule(regex.escape(quote), on_match=then(end, lambda t: t.leave(), f"leave {kind}-string body", "magenta"))
    mode.illegal.rule(r"\}", on_match=error(f"single '}}' is not allowed in an {kind}-string, write '}}}}'"))
    if len(quote) == 1:
        mode.illegal.rule(r"\n", on_match=error(f"unterminated {kind}-string", leave=True))
    return mode


PYTHON = block((0,))


def python_tokens(src: str):
    """
    Tokenize Python source, like ``tokenize.generate_tokens``.

    A missing final newline is added first, as Python's tokenizer does.
    """
    if not src.endswith("\n"):
        src += "\n"
    tokenizer = PYTHON.tokenize(src)
    yield from tokenizer
    # The tokenizer stops at the end of the input, so what Python adds there is made
    # here: a DEDENT for each block still open, then the ENDMARKER.
    while tokenizer.depth:
        if BLANK_NEWLINE in tokenizer.grammar:  # only blocks have this rule
            yield tokenizer.mint(tokenizer.scan(r""), Py.DEDENT)
        else:
            trace(tokenizer, "✗ unexpected end of input inside a string or bracket", "bold red")
            yield tokenizer.mint(tokenizer.scan(r""), Py.ERRORTOKEN)
        tokenizer.leave()
    yield tokenizer.mint(tokenizer.scan(r""), Py.ENDMARKER)


# -- checking against Python's tokenizer -----------------------------------------

def comparable(rows, *, ours: bool):
    """
    Make our tokens and Python's comparable, as (type, text, start, end).

    Ours keep the source text exactly, so "{{" stays "{{", while Python's tokenizer
    collapses doubled braces, splits the text around them and adds empty middles.
    So middles are merged and compared by text only, empty ones are dropped, and
    the doubled braces in ours are collapsed.
    """
    out = []
    for name, text, start, end in rows:
        if name.endswith("_MIDDLE"):
            if out and out[-1][0] == name:
                out[-1] = (name, out[-1][1] + text, None, None)
            else:
                out.append((name, text, None, None))
        else:
            out.append((name, text, start, end))
    return [
        (n, t.replace("{{", "{").replace("}}", "}"), s, e) if ours and n.endswith("_MIDDLE") else (n, t, s, e)
        for n, t, s, e in out
        if t or not n.endswith("_MIDDLE")
    ]


def python_span(token) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    A token's start and end in the convention of Python's tokenizer.

    Columns are 0-based there. And Python ends a NEWLINE or NL on its own line,
    one column past the break, where our tail is the start of the next line, as
    the [head, tail) rule says. Only the convention differs.
    """
    (line, col), (end_line, end_col) = token.head, token.tail
    if token.type in (Py.NEWLINE, Py.NL):
        return (line, col - 1), (line, col)
    return (line, col - 1), (end_line, end_col - 1)


def check(src: str, tokens) -> None:
    if not src.endswith("\n"):
        src += "\n"
    needs = "TSTRING_START" if any(t.type is Py.TSTRING_START for t in tokens) else "FSTRING_START"
    if not hasattr(tokenize, needs):
        console.print(f"[dim]⋯ skipped the check, this Python's tokenizer has no {needs}[/]")
        return
    if any(t.type is Py.ERRORTOKEN for t in tokens):
        console.print("[dim]⋯ skipped the check, the code has errors[/]")
        return
    try:
        reference = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (SyntaxError, tokenize.TokenError) as rejected:
        console.print(f"[yellow]⋯ Python's tokenizer rejects this code: {rejected}[/]")
        return
    ours = comparable(((t.type.name, t.data, *python_span(t)) for t in tokens), ours=True)
    theirs = comparable(((tokenize.tok_name[t.type], t.string, t.start, t.end) for t in reference), ours=False)
    if ours == theirs:
        console.print(f"[bold green]✓ matches Python's tokenizer exactly, positions included ({len(ours)} tokens)[/]")
        return
    at = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
    console.print(f"[bold red]✗ differs from Python's tokenizer at token {at}[/]")
    console.print(f"  ours:   {ours[at:at + 3]}")
    console.print(f"  Python: {theirs[at:at + 3]}")


def show(title: str, src: str) -> None:
    console.rule(f"[bold]{title}")
    console.print(Panel(Syntax(src.rstrip("\n"), "python", theme="ansi_dark", line_numbers=True), expand=False))
    if TRACE:
        console.print("[bold]trace[/]")
    tokens = list(python_tokens(src))

    table = Table(show_edge=False, pad_edge=False, box=None, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("type")
    table.add_column("text", overflow="fold")
    table.add_column("from → to", style="dim")
    for i, token in enumerate(tokens):
        style = STYLES[token.type]
        table.add_row(
            str(i),
            f"[{style}]{token.type.name}[/]",
            f"[{style}]{token.data!r}[/]",
            f"{token.head[0]}:{token.head[1]} → {token.tail[0]}:{token.tail[1]}",
        )
    console.print("[bold]tokens[/]")
    console.print(table)
    check(src, tokens)


# -- your code -------------------------------------------------------------------
# Write any Python inside the dedent and run this file. It's a raw string, so
# backslashes stay as you type them. If your code needs ''' itself, switch the
# delimiters to """.

YOUR_CODE = textwrap.dedent(r'''
    """A module docstring."""
    from __future__ import annotations

    import asyncio
    from dataclasses import dataclass

    type Pair[T] = tuple[T, T]


    @dataclass(frozen=True)
    class Point[T: (int, float)]:
        x: T
        y: T = 0  # default

        def __add__(self, other: Point[T]) -> Point[T]:
            return Point(self.x + other.x,
                         self.y + other.y)


    async def fetch(*urls: str, retries: int = 3, **options) -> list[bytes]:
        results = [b"\x00", rb"\d+", 0xFF_FF, 0o17, 0b1010, 1_000.5e-3, 3j, .5]
        while (n := len(results)) > retries:
            results = results[1:n:2]
        match options:
            case {"mode": "fast", **rest} if rest:
                pass
            case [first, *_]:
                ...
        total = 1 + \
            2
        return [f"{url!r:>{width}} {total = }" for url in urls if url]


    template = t"Hello {name!s}, you owe {amount:.2f}"
    nested = f"{'quoted'} {f"{"deep"}"} {{literal}}"
''').lstrip("\n")

if __name__ == "__main__":
    show("Your code", YOUR_CODE)
