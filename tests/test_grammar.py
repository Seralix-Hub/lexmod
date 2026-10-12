import copy

import pytest

from lexmod import Grammar

from conftest import Kind


def hook(tokenizer, value):
    return None


@pytest.fixture
def grammar() -> Grammar[Kind, str]:
    return Grammar[Kind, str]()


class TestRulePaths:
    def test_new_rule_with_a_type(self, grammar):
        grammar.rule("a", Kind.NAME)
        assert grammar["a"].type is Kind.NAME

    def test_type_and_on_match_together(self, grammar):
        with pytest.raises(ValueError, match="both a type and an on_match"):
            grammar.rule("a", Kind.NAME, on_match=hook)

    def test_skip(self, grammar):
        grammar.skip(r"\s+")
        assert grammar[r"\s+"] is Grammar.Skip

    def test_on_token_without_anything_to_make_a_token(self, grammar):
        with pytest.raises(ValueError, match="needs a type or an on_match"):
            grammar.rule("a", on_token=hook)

    def test_anything_on_a_skipped_pattern(self, grammar):
        grammar.skip("a")
        with pytest.raises(ValueError, match="already defined as skipped"):
            grammar.on_token("a")(hook)

    def test_adding_on_token_to_a_rule(self, grammar):
        grammar.rule("a", Kind.NAME)
        grammar.on_token("a")(hook)
        assert grammar["a"].on_token is hook
        assert grammar["a"].type is Kind.NAME

    def test_adding_a_second_on_token(self, grammar):
        grammar.rule("a", Kind.NAME, on_token=hook)
        with pytest.raises(ValueError, match="already has an on_token"):
            grammar.on_token("a")(hook)

    def test_adding_on_token_with_a_priority(self, grammar):
        grammar.rule("a", Kind.NAME)
        with pytest.raises(ValueError, match="cannot change it"):
            grammar.on_token("a", priority=3)(hook)

    @pytest.mark.parametrize(
        "redefine",
        [
            lambda g: g.rule("a", Kind.NUMBER),
            lambda g: g.skip("a"),
            lambda g: g.on_match("a")(hook),
            lambda g: g.rule("a", priority=3),
        ],
    )
    def test_redefining_a_rule(self, grammar, redefine):
        grammar.rule("a", Kind.NAME)
        with pytest.raises(ValueError, match="already defined"):
            redefine(grammar)

    def test_equivalent_patterns_are_the_same_rule(self, grammar):
        grammar.rule("a", Kind.NAME)
        with pytest.raises(ValueError, match="already defined"):
            grammar.rule("a", Kind.NUMBER, flags=0)

    def test_pattern_type_is_fixed_by_the_first_rule(self, grammar):
        grammar.rule("a", Kind.NAME)
        with pytest.raises(TypeError, match="expected str pattern"):
            grammar.rule(b"b", Kind.NAME)

    def test_non_callable_hook(self, grammar):
        with pytest.raises(TypeError, match="must be callable"):
            grammar.on_match("a")(42)


def test_decorators_stack_and_return_the_hook(grammar):
    @grammar.on_match("x")
    @grammar.on_match("y")
    def both(tokenizer, match):
        return Kind.NAME

    assert grammar["x"].on_match is both
    assert grammar["y"].on_match is both


def test_iterates_in_priority_order(grammar):
    grammar.rule("low", Kind.NAME, 0)
    grammar.rule("high", Kind.NAME, 5)
    grammar.rule("mid", Kind.NAME, 2)
    assert [rx.regex for rx in grammar] == ["high", "mid", "low"]
    assert len(grammar) == 3
    assert "mid" in grammar
    assert "absent" not in grammar
    assert 42 not in grammar


def test_match_follows_priority_then_length(grammar):
    grammar.rule("if", Kind.KEYWORD, 5)
    grammar.rule("i", Kind.NAME, 5)
    grammar.rule(r"[a-z]+", Kind.NAME)
    rule, match = grammar.match("ifx")
    assert (rule.type, match.group()) == (Kind.KEYWORD, "if")
    rule, match = grammar.match("abc")
    assert (rule.type, match.group()) == (Kind.NAME, "abc")
    assert grammar.match("123") is None


def test_include_adds_rules_with_priorities_and_flags(grammar):
    other = Grammar[Kind, str]()
    other.rule("if", Kind.KEYWORD, 5, flags=2)  # IGNORECASE
    other.skip(r"\s+")
    grammar.rule(r"[a-z]+", Kind.NAME)
    grammar.include(other)
    rule, match = grammar.match("IF")
    assert rule.type is Kind.KEYWORD


def test_include_conflict_changes_nothing(grammar):
    grammar.rule("a", Kind.NAME)
    other = Grammar[Kind, str]()
    other.rule("b", Kind.NAME)
    other.rule("a", Kind.NUMBER)
    with pytest.raises(ValueError, match="defined in both"):
        grammar.include(other)
    assert "b" not in grammar


def test_include_of_another_pattern_type(grammar):
    grammar.rule("a", Kind.NAME)
    other = Grammar[Kind, bytes]()
    other.rule(b"b", Kind.NAME)
    with pytest.raises(TypeError):
        grammar.include(other)


def test_copies_are_independent(grammar):
    grammar.rule("a", Kind.NAME)
    grammar.skip(r"\s+")
    for duplicate in (copy.copy(grammar), copy.deepcopy(grammar)):
        duplicate.rule("b", Kind.NAME)
        assert "b" not in grammar
        assert duplicate[r"\s+"] is Grammar.Skip
