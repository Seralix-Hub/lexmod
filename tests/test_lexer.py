import copy

import pytest

from lexmod import Grammar, Lexer

from conftest import Kind, kinds


def test_illegal_rules_need_a_rule_first():
    with pytest.raises(RuntimeError, match="define a rule before"):
        Lexer[Kind, str]().illegal


def test_illegal_rules_use_the_lexer_pattern_type(lexer):
    with pytest.raises(TypeError):
        lexer.illegal.rule(b"x", Kind.ERROR)


def test_illegal_rules_are_only_a_fallback(lexer):
    # As broad as possible, and still never chosen where a legal rule matches.
    lexer.illegal.rule(r"[^\s]+", Kind.ERROR, 99)
    assert kinds(lexer.tokenize("a $ 1")) == ["NAME", "ERROR", "NUMBER"]


def test_tokenize_checks_the_source(lexer):
    with pytest.raises(TypeError, match="expected str source"):
        lexer.tokenize(b"a")
    with pytest.raises(RuntimeError, match="no rules"):
        Lexer[Kind, str]().tokenize("a")


def test_include_brings_illegal_rules_along(lexer):
    other = Lexer[Kind, str]()
    other.rule("@", Kind.OP)
    other.illegal.rule(r"\$", Kind.ERROR)
    lexer.include(other)
    assert kinds(lexer.tokenize("a @ $")) == ["NAME", "OP", "ERROR"]


def test_include_of_a_plain_grammar(lexer):
    other = Grammar[Kind, str]()
    other.rule("@", Kind.OP)
    lexer.include(other)
    assert "@" in lexer


def test_include_with_clashing_illegal_rules_changes_nothing(lexer):
    lexer.illegal.rule(r"\$", Kind.ERROR)
    other = Lexer[Kind, str]()
    other.rule("@", Kind.OP)
    other.illegal.rule(r"\$", Kind.ERROR)
    with pytest.raises(ValueError, match="illegal rules defined in both"):
        lexer.include(other)
    assert "@" not in lexer


def test_copies_keep_illegal_rules_independent(lexer):
    lexer.illegal.rule(r"\$", Kind.ERROR)
    for duplicate in (copy.copy(lexer), copy.deepcopy(lexer)):
        duplicate.illegal.rule("#", Kind.ERROR)
        assert "#" not in lexer.illegal
        assert r"\$" in duplicate.illegal


def test_changing_the_lexer_after_tokenize_does_not_affect_it(lexer):
    tokenizer = lexer.tokenize("a @")
    lexer.rule("@", Kind.OP)
    assert next(tokenizer).data == "a"
    with pytest.raises(SyntaxError):
        next(tokenizer)
