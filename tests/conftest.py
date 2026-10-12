from enum import Enum, auto

import pytest

from lexmod import Lexer


class Kind(Enum):
    NAME = auto()
    NUMBER = auto()
    OP = auto()
    KEYWORD = auto()
    QUOTE = auto()
    TEXT = auto()
    ESCAPE = auto()
    NEWLINE = auto()
    ERROR = auto()
    MARK = auto()


@pytest.fixture
def lexer() -> Lexer[Kind, str]:
    """Names, numbers and a few operators, with whitespace skipped."""
    lexer = Lexer[Kind, str]()
    lexer.skip(r"[ \t]+")
    lexer.rule(r"\n", Kind.NEWLINE)
    lexer.rule(r"[A-Za-z_]\w*", Kind.NAME)
    lexer.rule(r"\d+", Kind.NUMBER)
    lexer.rule(r"[-+*/=()]", Kind.OP)
    return lexer


def kinds(tokens) -> list[str]:
    return [token.type.name for token in tokens]


def texts(tokens) -> list:
    return [token.data for token in tokens]
