import pytest
import regex

from lexmod.regex import Regex, coerce


def test_default_flags_are_dropped():
    assert Regex("a") == Regex("a", regex.UNICODE) == Regex(regex.compile("a"))
    assert Regex(b"a") == Regex(b"a", regex.ASCII)
    assert Regex("a").flags == 0


def test_flags_that_change_behavior_are_kept():
    assert Regex("a", regex.I).flags == regex.I
    assert Regex("a", regex.I) != Regex("a")


def test_compiled_pattern_brings_its_flags():
    assert Regex(regex.compile("a", regex.I)) == Regex("a", regex.I)


def test_given_encoding_replaces_the_compiled_one():
    assert Regex(regex.compile("a", regex.A), regex.U).flags == 0


def test_inline_flags_stay_in_the_source():
    rx = Regex("(?a)\\w")
    assert rx.flags == 0
    assert rx.pattern.match("é") is None


def test_equal_patterns_share_one_compiled_pattern():
    assert Regex("x+", regex.I).pattern is Regex("x+", regex.I).pattern


def test_str_and_repr():
    assert str(Regex("a+")) == "'a+'"
    assert str(Regex("a+", regex.I | regex.M)) == "im'a+'"
    assert str(Regex("a+", regex.V1)) == "v1-'a+'"
    assert repr(Regex("a+", regex.I)) == "Regex('a+', flags=RegexFlag.I)"
    rebuilt = eval(repr(Regex("a+", regex.I | regex.M)), {"Regex": Regex, "RegexFlag": regex.RegexFlag})
    assert rebuilt == Regex("a+", regex.I | regex.M)


@pytest.mark.parametrize(
    "form",
    ["a+", regex.compile("a+"), Regex("a+"), ("a+", 0), (regex.compile("a+"), 0)],
)
def test_coerce_accepts_every_form(form):
    assert coerce(form) == Regex("a+")


def test_coerce_keeps_a_regex_as_is():
    rx = Regex("a")
    assert coerce(rx) is rx


def test_coerce_rejects_anything_else():
    with pytest.raises(TypeError, match=r"tuple\[str, str\]"):
        coerce(("a", "b"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        coerce(42)  # type: ignore[arg-type]
