import dataclasses

import pytest
import regex

from lexmod import Grammar, Lexer, Token

from conftest import Kind, kinds, texts


class TestPositions:
    def test_tokens_tile_the_source(self, lexer):
        tokens = list(lexer.tokenize("ab 12\n  cd\nx"))
        assert [(t.data, t.head, t.tail) for t in tokens] == [
            ("ab", (1, 1), (1, 3)),
            ("12", (1, 4), (1, 6)),
            ("\n", (1, 6), (2, 1)),
            ("cd", (2, 3), (2, 5)),
            ("\n", (2, 5), (3, 1)),
            ("x", (3, 1), (3, 2)),
        ]

    def test_spans_index_the_source(self, lexer):
        tokenizer = lexer.tokenize("total = (a + 42)")
        for token in tokenizer:
            assert tokenizer.src[slice(*token.span)] == token.data

    def test_control_characters_are_removed(self, lexer):
        assert texts(lexer.tokenize("a\r\nb\fc")) == ["a", "\n", "bc"]

    def test_bytes_source(self):
        lexer = Lexer[Kind, bytes]()
        lexer.skip(rb"\s+")
        lexer.rule(rb"\w+", Kind.NAME)
        tokens = list(lexer.tokenize(b"ab\ncd"))
        assert [(t.data, t.head) for t in tokens] == [(b"ab", (1, 1)), (b"cd", (2, 1))]

    def test_cursor_and_pos_follow_the_tokens(self, lexer):
        tokenizer = lexer.tokenize("ab\ncd")
        next(tokenizer)
        assert (tokenizer.pos, tokenizer.cursor) == (2, (1, 3))


class TestHooks:
    def test_on_match_returning_skip_consumes_the_match(self, lexer):
        seen = []
        lexer.rule(r"#[^\n]*", on_match=lambda t, m: Grammar.Skip, on_token=lambda t, tok: seen.append(tok))
        assert kinds(lexer.tokenize("a # comment")) == ["NAME"]
        assert seen == []  # on_token never runs for a skipped match

    def test_on_match_returning_a_type(self, lexer):
        lexer.rule("@", on_match=lambda t, m: Kind.OP)
        assert kinds(lexer.tokenize("@")) == ["OP"]

    def test_on_match_returning_its_own_token(self, lexer):
        lexer.rule("@", on_match=lambda t, m: t.mint(m, Kind.MARK))
        assert kinds(lexer.tokenize("@ a")) == ["MARK", "NAME"]

    def test_on_match_returning_a_type_after_consuming(self, lexer):
        def both(tokenizer, match):
            tokenizer.mint(match, Kind.MARK)
            return Kind.OP

        lexer.rule("@", on_match=both)
        with pytest.raises(ValueError, match="after consuming input"):
            list(lexer.tokenize("@"))

    def test_on_match_returning_something_else(self, lexer):
        lexer.rule("@", on_match=lambda t, m: "OP")
        with pytest.raises(TypeError, match="returned str"):
            list(lexer.tokenize("@"))

    def test_on_token_keeps_replaces_or_drops(self):
        lexer = Lexer[Kind, str]()
        lexer.skip(r"\s+")
        keywords = {"if": Kind.KEYWORD}

        @lexer.on_token(r"\w+", Kind.NAME)
        def adjust(tokenizer, token):
            if token.data == "drop":
                return Grammar.Skip
            if token.data == "big":
                return dataclasses.replace(token, type=Kind.MARK)
            return keywords.get(token.data)

        tokens = list(lexer.tokenize("if ifx drop big"))
        assert [(t.type.name, t.data) for t in tokens] == [("KEYWORD", "if"), ("NAME", "ifx"), ("MARK", "big")]

    def test_hook_token_with_the_wrong_data_type(self, lexer):
        lexer.rule("@", on_match=lambda t, m: Token(Kind.OP, b"@", (0, 1), (1, 1), (1, 2)))
        with pytest.raises(TypeError, match="bytes data"):
            list(lexer.tokenize("@"))


class TestConsuming:
    def test_scan_does_not_consume(self, lexer):
        tokenizer = lexer.tokenize("abc")
        assert tokenizer.scan(r"ab").group() == "ab"
        assert tokenizer.pos == 0
        assert tokenizer.scan(r"x") is None

    def test_zero_width_token(self, lexer):
        tokenizer = lexer.tokenize("a")
        token = tokenizer.mint(tokenizer.scan(r""), Kind.MARK)
        assert (token.data, token.head, token.tail) == ("", (1, 1), (1, 1))

    def test_match_on_other_text_is_rejected(self, lexer):
        tokenizer = lexer.tokenize("abc")
        with pytest.raises(ValueError, match="different text"):
            tokenizer.move(regex.match("a", "a"))

    def test_match_on_a_slice_is_rejected(self, lexer):
        tokenizer = lexer.tokenize("ab cd")
        next(tokenizer)
        with pytest.raises(ValueError):
            tokenizer.move(regex.match(r"\s*cd", tokenizer.src[2:]))

    def test_match_away_from_the_position_is_rejected(self, lexer):
        tokenizer = lexer.tokenize("ab cd")
        with pytest.raises(ValueError, match="must start at 0"):
            tokenizer.move(regex.compile("cd").search(tokenizer.src))

    def test_mint_needs_an_enum_type(self, lexer):
        tokenizer = lexer.tokenize("ab")
        with pytest.raises(TypeError, match="Enum member"):
            tokenizer.mint(tokenizer.scan("ab"), "NAME")
        assert tokenizer.pos == 0  # nothing consumed


class TestModes:
    @pytest.fixture
    def strings(self, lexer):
        string = Lexer[Kind, str]()
        string.rule(r'[^"\\]+', Kind.TEXT)
        string.rule(r"\\.", Kind.ESCAPE)

        @string.on_match('"')
        def close(tokenizer, match):
            token = tokenizer.mint(match, Kind.QUOTE)
            tokenizer.leave()
            return token

        @lexer.on_match('"')
        def open_(tokenizer, match):
            token = tokenizer.mint(match, Kind.QUOTE)
            tokenizer.enter(string)
            return token

        return lexer

    def test_enter_and_leave(self, strings):
        assert kinds(strings.tokenize(r'a "b \" c" d')) == ["NAME", "QUOTE", "TEXT", "ESCAPE", "TEXT", "QUOTE", "NAME"]

    def test_depth_and_grammar(self, strings):
        tokenizer = strings.tokenize('a "b" c')
        assert tokenizer.depth == 0
        next(tokenizer), next(tokenizer)
        assert tokenizer.depth == 1
        assert r'[^"\\]+' in tokenizer.grammar

    def test_unclosed_mode_is_visible_at_the_end(self, strings):
        tokenizer = strings.tokenize('a "open')
        list(tokenizer)
        assert tokenizer.depth == 1

    def test_leaving_the_lexer_rules(self, lexer):
        with pytest.raises(RuntimeError, match="cannot leave"):
            lexer.tokenize("a").leave()

    def test_entering_an_empty_grammar(self, lexer):
        with pytest.raises(ValueError, match="empty"):
            lexer.tokenize("a").enter(Grammar[Kind, str]())

    def test_entering_another_pattern_type(self, lexer):
        other = Grammar[Kind, bytes]()
        other.rule(b"a", Kind.NAME)
        with pytest.raises(TypeError, match="bytes patterns"):
            lexer.tokenize("a").enter(other)


class TestCheckpoint:
    def test_everything_is_undone(self, lexer):
        tokenizer = lexer.tokenize("a b\nc")
        next(tokenizer)
        before = (tokenizer.pos, tokenizer.cursor, tokenizer.depth)
        with tokenizer.checkpoint():
            assert texts(tokenizer) == ["b", "\n", "c"]
        assert (tokenizer.pos, tokenizer.cursor, tokenizer.depth) == before
        assert texts(tokenizer) == ["b", "\n", "c"]

    def test_undone_after_an_exception_and_when_nested(self, lexer):
        tokenizer = lexer.tokenize("a b c")
        with pytest.raises(KeyError):
            with tokenizer.checkpoint():
                next(tokenizer)
                with tokenizer.checkpoint():
                    list(tokenizer)
                raise KeyError("boom")
        assert texts(tokenizer) == ["a", "b", "c"]

    def test_entered_grammars_are_undone(self, lexer):
        tokenizer = lexer.tokenize("a")
        with tokenizer.checkpoint():
            tokenizer.enter(lexer)
            assert tokenizer.depth == 1
        assert tokenizer.depth == 0

    def test_lookahead_from_a_hook(self):
        lexer = Lexer[Kind, str]()
        lexer.skip(r"\s+")
        lexer.rule(r"[()]", Kind.OP)

        @lexer.on_match(r"\w+")
        def name(tokenizer, match):
            token = tokenizer.mint(match, Kind.NAME)
            with tokenizer.checkpoint():
                following = next(tokenizer, None)
            if following is not None and following.data == "(":
                return dataclasses.replace(token, type=Kind.KEYWORD)
            return token

        assert kinds(lexer.tokenize("f(x) y")) == ["KEYWORD", "OP", "NAME", "OP", "NAME"]


class TestErrors:
    def test_no_rule_matches(self, lexer):
        with pytest.raises(SyntaxError) as raised:
            list(lexer.tokenize("ab\ncd ?"))
        assert (raised.value.lineno, raised.value.offset, raised.value.text) == (2, 4, "cd ?")

    def test_rule_that_never_consumes(self):
        lexer = Lexer[Kind, str]()
        lexer.skip(r"\s*", 5)
        lexer.rule(r"\w", Kind.NAME)
        with pytest.raises(RuntimeError, match="no progress"):
            list(lexer.tokenize("a b"))

    def test_stop_iteration_from_a_hook(self, lexer):
        def greedy(tokenizer, match):
            token = tokenizer.mint(match, Kind.MARK)
            next(tokenizer)
            return token

        lexer.rule("@", on_match=greedy)
        with pytest.raises(RuntimeError, match="StopIteration"):
            list(lexer.tokenize("@"))

    def test_long_runs_of_skipped_input(self):
        lexer = Lexer[Kind, str]()
        lexer.skip(r"#[^\n]*\n")
        lexer.rule("x", Kind.NAME)
        assert texts(lexer.tokenize("# comment\n" * 20_000 + "x")) == ["x"]
