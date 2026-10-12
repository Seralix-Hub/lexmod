import pytest

from lexmod import Token

from conftest import Kind


def test_single_line_token():
    token = Token(Kind.NAME, "abc", (4, 7), (1, 5), (1, 8))
    assert token.tail == (1, 8)


def test_tail_moves_to_the_next_line_after_a_break():
    Token(Kind.NEWLINE, "\n", (5, 6), (1, 6), (2, 1))
    Token(Kind.TEXT, "a\nbc", (0, 4), (1, 5), (2, 3))


def test_zero_width_token_has_equal_head_and_tail():
    token = Token(Kind.MARK, "", (3, 3), (2, 4), (2, 4))
    assert token.head == token.tail


def test_bytes_token():
    Token(Kind.TEXT, b"ab\ncd", (0, 5), (1, 1), (2, 3))


@pytest.mark.parametrize(
    ("span", "data", "head", "tail", "message"),
    [
        ((3, 2), "", (1, 1), (1, 1), "invalid span"),
        ((-1, 0), "", (1, 1), (1, 1), "invalid span"),
        ((0, 3), "ab", (1, 1), (1, 3), "data length"),
        ((0, 2), "ab", (0, 1), (0, 3), "invalid head"),
        ((0, 2), "ab", (1, 0), (1, 2), "invalid head"),
        # Not starting at column 1, so the tail must keep the head's column.
        ((4, 6), "ab", (1, 5), (1, 3), "does not match"),
        # A break right at the start restarts the column at 1.
        ((4, 6), "\nx", (1, 5), (2, 6), "does not match"),
    ],
)
def test_invalid_tokens_are_rejected(span, data, head, tail, message):
    with pytest.raises(ValueError, match=message):
        Token(Kind.TEXT, data, span, head, tail)


def test_tokens_are_immutable():
    token = Token(Kind.NAME, "a", (0, 1), (1, 1), (1, 2))
    with pytest.raises(AttributeError):
        token.data = "b"  # type: ignore[misc]
