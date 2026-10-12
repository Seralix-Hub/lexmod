"""
The Python tokenizer in examples/ against CPython's own, token for token,
positions included. It exercises nearly everything lexmod does: modes, nested
modes, zero-width tokens, lookbehinds and illegal rules.
"""
import importlib.util
import io
import pathlib
import textwrap
import tokenize

import pytest

pytest.importorskip("rich")  # the example prints with rich

_path = pathlib.Path(__file__).parent.parent / "examples" / "python_lexer.py"
_spec = importlib.util.spec_from_file_location("python_lexer", _path)
python_lexer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(python_lexer)
python_lexer.TRACE = False


def ours(src: str):
    tokens = python_lexer.python_tokens(src)
    return python_lexer.comparable(((t.type.name, t.data, *python_lexer.python_span(t)) for t in tokens), ours=True)


def theirs(src: str):
    if not src.endswith("\n"):
        src += "\n"
    reference = tokenize.generate_tokens(io.StringIO(src).readline)
    return python_lexer.comparable(((tokenize.tok_name[t.type], t.string, t.start, t.end) for t in reference), ours=False)


SNIPPETS = {
    "indentation": """
        def f(x):
            if x:
                return [
                    1,
                    2,
                ]
            # a comment

            return None
        """,
    "continuation": """
        total = 1 + \\
            2
        """,
    "numbers": "x = [0xFF_FF, 0o17, 0b1010, 1_000.5e-3, 3j, .5, 1.]\n",
    "strings": """s = (b"\\x00", rb"\\d+", '''multi\nline''', u"x")\n""",
    "f-strings": """f"{user.name!r:>10} {x=} {value:{width}.{precision}f} {{literal}} {x:=10}"\n""",
    "nested f-strings": """f"{'a'} {f"{"deep"}"}"\n""",
    "brackets in fields": """f"{ {'a': 1}['a'] } {(lambda: 1)()} {items[1:2]}"\n""",
    "match and type": """
        type Pair[T] = tuple[T, T]
        match command:
            case {"mode": "fast", **rest} if rest:
                pass
            case [first, *_]:
                ...
        """,
}


@pytest.mark.parametrize("name", SNIPPETS)
def test_snippet(name):
    src = textwrap.dedent(SNIPPETS[name]).lstrip("\n")
    assert ours(src) == theirs(src)


@pytest.mark.skipif(not hasattr(tokenize, "TSTRING_START"), reason="t-strings need Python 3.14+")
def test_t_strings():
    src = 'template = t"Hello {name!s}, you owe {amount:.2f}"\n'
    assert ours(src) == theirs(src)


@pytest.mark.parametrize(
    "module",
    ["argparse", "ast", "dataclasses", "enum", "functools", "inspect", "json.decoder", "pathlib", "string", "typing"],
)
def test_standard_library_module(module):
    origin = importlib.util.find_spec(module).origin
    src = pathlib.Path(origin).read_text(encoding="utf-8")
    if "\t" in src:
        pytest.skip("tabs in indentation aren't modeled")
    assert ours(src) == theirs(src)


def test_unclosed_bracket_is_reported():
    kinds = [t.type.name for t in python_lexer.python_tokens("(1, 2\n")]
    assert kinds[-2:] == ["ERRORTOKEN", "ENDMARKER"]
