# **lexmod**

**Modal lexing for Python.** Build tokenizers from regular expressions, and switch
grammars as the input changes: inside strings, comments, template fields or
indented blocks.

Most regex lexers have one flat list of rules. Real languages don't: the inside of
an f-string, its format spec, a comment or an indented block each lex differently,
and they nest. In lexmod, a rule's hook can *enter* another grammar and *leave*
it again, so the tokenizer carries a stack of modes. Lookahead is safe too: a
checkpoint undoes everything, the mode stack included.

How far that goes: a complete **Python 3.14 tokenizer** built with lexmod, with
f-strings, t-strings, INDENT and DEDENT, produces the same tokens at the same
positions as CPython's own `tokenize` on 482 files of the standard library.

```
pip install lexmod
```

lexmod needs **Python 3.15+**, and installs
[`regex`](https://pypi.org/project/regex/). [`rich`](https://pypi.org/project/rich/)
is optional: `pip install "lexmod[rich]"` adds it, for the examples' output.

## Quickstart

```python
from enum import Enum, auto
from lexmod import Lexer

class Kind(Enum):
    NAME = auto()
    NUMBER = auto()
    OP = auto()

lexer = Lexer[Kind, str]()
lexer.skip(r"\s+")
lexer.rule(r"[A-Za-z_]\w*", Kind.NAME)
lexer.rule(r"\d+", Kind.NUMBER)
lexer.rule(r"[-+*/=()]", Kind.OP)

for token in lexer.tokenize("total = (a + 42)"):
    print(token.type.name, repr(token.data), token.head, token.tail)
```

```
NAME 'total' (1, 1) (1, 6)
OP '=' (1, 7) (1, 8)
OP '(' (1, 9) (1, 10)
NAME 'a' (1, 10) (1, 11)
OP '+' (1, 12) (1, 13)
NUMBER '42' (1, 14) (1, 16)
OP ')' (1, 16) (1, 17)
```

The examples below add the token types they use to `Kind`.

## Tokens

A `Token` is immutable and knows exactly where it came from:

| Field  | Meaning                                            |
|--------|----------------------------------------------------|
| `type` | A member of your token enum                        |
| `data` | The matched text, `str` or `bytes` like the source |
| `span` | 0-based `(pos, endpos)` offsets into the source    |
| `head` | 1-based `(line, col)` of the first character       |
| `tail` | 1-based `(line, col)` just past the last character |

Spans and positions are half-open, so each token's `tail` is exactly the next
token's `head`, and a zero-width token such as INDENT or EOF has `head == tail`.
Tokens check these invariants when they're made.

## Rules

```python
import regex

lexer.rule(pattern, Kind.X)                 # matches become X tokens
lexer.rule(pattern, Kind.X, 5)              # with a priority, 0 by default
lexer.rule(pattern, Kind.X, flags=regex.I)  # with regex flags
lexer.skip(pattern)                         # matches are consumed, no token
```

Patterns are `str` or `bytes` sources, or compiled patterns from the
[`regex`](https://pypi.org/project/regex/) module, which adds variable-length
lookbehind and more to Python's `re`. All rules of a lexer share one pattern type,
fixed by the first rule. Flags are normalized, so equivalent patterns are the same
rule, and defining a rule twice is an error.

**Choosing between rules**, at each position:

1. The highest priority with any match wins.
2. Within that priority, the longest match wins.
3. On equal length, the rule added first wins.

So operators can be plain rules at the same priority, and `**=` still beats `**`,
which beats `*`.

## Hooks

Hooks let a rule decide what its match becomes. They receive the tokenizer
first, so they can consume input, look ahead or switch modes.

**`on_match(tokenizer, match)`** decides how the token is made. It returns:

- a `Token`, made by the hook itself, normally with `tokenizer.mint(match, type)`,
- a token type, for the default minter to use,
- or `Grammar.Skip`, to drop the match.

**`on_token(tokenizer, token)`** runs once the token is made. It returns a
replacement `Token`, a token type to make it with instead, `Grammar.Skip` to drop
it, or `None` to keep it unchanged. It never runs for a match `on_match` skipped.

Both are also decorators, and the hook is returned unchanged, so one hook can be
reused for several patterns by stacking them. Keywords are the classic case:

```python
KEYWORDS = {"if": Kind.IF, "else": Kind.ELSE}

@lexer.on_token(r"[A-Za-z_]\w*", Kind.NAME)
def keyword(tokenizer, token):
    return KEYWORDS.get(token.data)  # a keyword's type, or None to keep NAME
```

This lexes `if ifx else` as `IF NAME ELSE`. Giving `if` its own rule with a higher
priority instead would lex `ifx` as `IF` followed by `x`, since priority wins
before length does.

**Keep hooks repeatable.** A checkpoint can run a hook for tokens that are then
thrown away. Anything a hook does outside the tokenizer, such as logging or
changing global state, may happen more than once.

## Illegal rules

A lexer's `illegal` rules are tried only where no legal rule matches. They
describe mistakes the user is likely to make, so those become tokens you can
report precisely, instead of stopping at the first one:

```python
lexer.illegal.rule(r"[^\s]", Kind.ERROR)
```

Since they only run after every legal rule failed, they never change how valid
input lexes. Where neither kind of rule matches, iterating raises a `SyntaxError`
carrying the line, column and source line, so a traceback points at the exact
character.

## Modes

A hook can switch to another grammar with `tokenizer.enter(grammar)`, until a
later hook calls `tokenizer.leave()`. Entered grammars stack, so they nest. Here
strings lex their escapes with rules of their own:

```python
string = Lexer[Kind, str]()
string.rule(r'[^"\\]+', Kind.TEXT)
string.rule(r"\\.", Kind.ESCAPE)

@string.on_match(r'"')
def close_string(tokenizer, match):
    token = tokenizer.mint(match, Kind.QUOTE)
    tokenizer.leave()
    return token

@lexer.on_match(r'"')
def open_string(tokenizer, match):
    token = tokenizer.mint(match, Kind.QUOTE)
    tokenizer.enter(string)
    return token
```

`if "say \"hi\"" else` then lexes as `IF QUOTE TEXT ESCAPE TEXT ESCAPE QUOTE ELSE`.
An entered lexer brings its own illegal rules along. Grammars are snapshotted on
`enter`, so changing them later never affects a running tokenizer.

## Lookahead

Inside a `checkpoint`, a hook can take tokens to decide what to do, and everything
is undone when the block ends: position, consumed input and entered grammars.

```python
import dataclasses

@lexer.on_match(r"[A-Za-z_]\w*")
def name(tokenizer, match):
    token = tokenizer.mint(match, Kind.NAME)
    with tokenizer.checkpoint():
        following = next(tokenizer, None)
    if following is not None and following.data == "(":
        return dataclasses.replace(token, type=Kind.CALL)
    return token
```

Use `next(tokenizer, None)` rather than `next(tokenizer)` when looking ahead,
since the end of the input may come first.

## The tokenizer, from a hook

| Member                      | Does                                                         |
|-----------------------------|--------------------------------------------------------------|
| `src`, `pos`, `endpos`      | The source, the current offset and where lexing stops        |
| `cursor`                    | The current 1-based `(line, col)`                            |
| `depth`                     | How many grammars are entered and not left yet               |
| `grammar`                   | A copy of the legal rules in use right now                   |
| `scan(pattern)`             | Matches a pattern at the current position, without consuming |
| `move(match)`               | Consumes a match without making a token                      |
| `mint(match, type)`         | Consumes a match and makes a token from it                   |
| `enter(grammar)`, `leave()` | Switch modes                                                 |
| `checkpoint()`              | Look ahead, then go back                                     |

`scan(r"")` gives a zero-width match, so `tokenizer.mint(tokenizer.scan(r""), Kind.DEDENT)`
makes a synthetic token at the current position.

The tokenizer stops at the end of the input, whatever mode it's in. A `depth`
above 0 after the last token means something was left open, such as a string or
a bracket.

## Composing grammars

A `Lexer` is a `Grammar` with illegal rules, and a `Grammar` is a read-only
mapping from patterns to rules, iterated in the order they're tried:

```python
r"\d+" in lexer      # True
lexer[r"\d+"]        # the rule
lexer.include(other) # adds every rule of another grammar or lexer
```

`include` refuses grammars that define the same pattern, and changes nothing
unless it succeeds completely.

## Errors

Mistakes are reported where they happen, with messages that name the pattern or
hook involved:

- **Defining rules:** `TypeError` for a pattern of the wrong type, `ValueError` for
  a rule that would be redefined.
- **Lexing:** `SyntaxError` where no rule matches, and `RuntimeError` when a rule
  keeps matching without consuming input, a common slip with patterns like `\s*`.
- **Hooks:** `TypeError` when a hook returns something it may not, and
  `RuntimeError` when one lets `StopIteration` escape.

## Notes

- The source is normalized first: `\a`, `\b`, `\v`, `\f` and `\r` are removed, so
  lines break at `\n` only, and offsets refer to the normalized text.
- Columns count code points, or bytes for `bytes` sources, so they can differ from
  an editor's for tabs and characters outside the Basic Multilingual Plane.

## Examples

[`examples/python_lexer.py`](examples/python_lexer.py) is a complete Python 3.14
tokenizer: names, numbers, all string forms, f-strings and t-strings, NEWLINE and
NL, INDENT and DEDENT. Each indentation level is an entered grammar, so the
indentation stack is the tokenizer's own mode stack. It checks itself against
CPython's `tokenize`, and you can write your own code at the bottom of the file:

```
pip install "lexmod[rich]"
python examples/python_lexer.py
```

## Development

```
pip install -e ".[dev]"
pytest
```

The tests include the Python tokenizer above, compared with CPython's on a set of
standard library modules.

## License

MIT, see [LICENSE](LICENSE).
