"""
The tokenizer: iterates over the tokens of a source.

It is also the object every hook receives, with what a hook needs to consume
input, make tokens, switch grammars and look ahead.
"""
import builtins
import contextlib
import dataclasses
import typing
from collections.abc import Generator, Iterator
from copy import copy
from enum import Enum
from typing import NamedTuple, Literal, Self, Any

import regex
from regex import Match

from .grammar import Grammar
from .regex import AnyPattern, Regex
from .token import Token


def _hook_name(hook: Any, /) -> str:
    """
    Name a hook for error messages.

    :param hook: Any callable
    :return: Its qualified name, or its repr when it has none
    """
    return getattr(hook, "__qualname__", repr(hook))


class _TokenizerFrame[Type: Enum, AnyStr: (str, bytes)](NamedTuple):
    """
    One entry of the tokenizer's grammar stack: the rules in use at one level.

    The lexer's own rules are the bottom frame, and :meth:`Tokenizer.enter` pushes
    one per entered grammar. Both grammars are snapshots made by
    :meth:`Tokenizer._frame`, so changing the originals never affects a running
    tokenizer. Frames are immutable, so checkpoints can save the stack with a
    shallow copy, and the no-progress guard tells frames apart by identity.

    :ivar legal: The rules tried first at each position
    :ivar illegal: The rules tried only where no legal rule matches
    """
    legal: Grammar[Type, AnyStr]
    illegal: Grammar[Type, AnyStr]


# noinspection protected-member
class Tokenizer[Type: Enum, AnyStr: (str, bytes)](Iterator[Token[Type, AnyStr]]):
    """
    Iterates over the tokens of a source, by matching it against a grammar.

    At each position the best legal rule wins, and the illegal rules are only
    tried where no legal rule matches. A rule's type makes the token through the
    default minter, :meth:`mint`, while its hooks can make, change or drop tokens
    themselves. Hooks receive the tokenizer, so they can use :meth:`scan`,
    :meth:`move` and :meth:`mint`, switch grammars with :meth:`enter` and
    :meth:`leave`, and look ahead inside a :meth:`checkpoint`.

    The source is normalized first: ``\\a``, ``\\b``, ``\\v``, ``\\f`` and ``\\r``
    are removed, so lines break at ``\\n`` only.
    """

    @property
    def src(self) -> AnyStr:
        """
        The normalized source being tokenized.

        :return: The source, without the control characters removed on creation
        """
        return self._src  # type: ignore[return-value]

    @property
    def pos(self) -> int:
        """
        The offset of the next character to consume.

        :return: A 0-based offset into :attr:`src`
        """
        return self._pos

    @property
    def endpos(self) -> int:
        """
        The offset where tokenizing stops.

        :return: The length of :attr:`src`
        """
        return self._endpos

    @property
    def cursor(self) -> tuple[int, int]:
        """
        The line and column of the next character to consume.

        :return: A 1-based ``(line, col)`` position
        """
        return self._line, self._col

    @property
    def depth(self) -> int:
        """
        How many grammars are entered and not left yet.

        It is 0 while lexing with the lexer's own rules. At the end of the input, a
        depth above 0 means something was left open, such as a string or a bracket.

        :return: The number of :meth:`enter` calls without a matching :meth:`leave`
        """
        return len(self._frames) - 1

    @property
    def grammar(self) -> Grammar[Type, AnyStr]:
        """
        A copy of the legal rules in use right now.

        Lets a hook or the caller tell which mode the tokenizer is in, for example
        with ``pattern in tokenizer.grammar``. It is a copy, so changing it never
        affects the tokenizer.

        :return: The legal rules of the grammar entered last, or the lexer's own
        """
        return copy(self._frames[-1].legal)

    def __init__(
            self,
            src: AnyStr,
            /,
            legal: Grammar[Type, AnyStr],
            illegal: Grammar[Type, AnyStr] | None = None,
    ) -> None:
        """
        Prepare to tokenize a source.

        :param src: The text to tokenize
        :param legal: The rules tried first at each position
        :param illegal: The rules tried where no legal rule matches. When omitted, a
            lexer's own illegal rules are used, and none for a plain grammar
        :raises TypeError: If ``src`` is not ``str`` or ``bytes``, or a grammar is
            invalid or uses another pattern type
        :raises ValueError: If ``legal`` has no rules
        """
        match src:
            case str():
                self._src: str = regex.sub(r"[\a\b\v\f\r]", "", src)
            case bytes():
                self._src: bytes = regex.sub(rb"[\a\b\v\f\r]", b"", src)
            case _:
                raise TypeError(f"expected a str or bytes source, got {type(src).__qualname__}")
        self._pos, self._endpos = 0, len(self._src)
        self._line, self._col = 1, 1
        # Position, frame and pattern of the last step that consumed nothing, to stop a
        # rule from matching nothing forever.
        self._stalled: tuple[int, _TokenizerFrame[Type, AnyStr], Any] | None = None

        # The lexer's own rules are just the first frame, so enter() does all the
        # checking and the defaulting of the illegal rules.
        self._frames: list[_TokenizerFrame[Type, AnyStr]] = []
        self.enter(legal, illegal)

    def _frame(
            self,
            legal: Grammar[Type, AnyStr],
            illegal: Grammar[Type, AnyStr],
            /,
    ) -> _TokenizerFrame[Type, AnyStr]:
        """
        Check that both grammars fit the source, and snapshot them into a stack frame.

        :param legal: The rules tried first at each position
        :param illegal: The rules tried where no legal rule matches
        :return: A frame holding copies of both grammars
        :raises TypeError: If either is not a grammar, or uses a pattern type other
            than the source's
        """
        for name, grammar in (("legal", legal), ("illegal", illegal)):
            if not isinstance(grammar, Grammar):
                raise TypeError(f"{name} rules must be a Grammar, got {type(grammar).__qualname__}")
            # An empty grammar has no pattern type yet, so it fits any source.
            if grammar._type is not None and not issubclass(grammar._type, type(self._src)):  # type: ignore[misc]
                raise TypeError(f"{name} rules use {grammar._type.__qualname__} patterns, but the source is {type(self._src).__qualname__}")
        # Snapshot, so changing a grammar while tokenizing can't change how this tokenizer
        # lexes. Grammars only hold immutable patterns and rules, so a shallow copy is
        # already a full snapshot, and much cheaper than a deep one.
        return _TokenizerFrame(copy(legal), copy(illegal))

    def enter(
            self,
            legal: Grammar[Type, AnyStr],
            illegal: Grammar[Type, AnyStr] | None = None,
    ) -> None:
        """
        Lex with other rules from now on, until the matching :meth:`leave`.

        Meant for parts of the source with their own syntax, such as the inside of a
        string or a comment: a hook enters the grammar where that part starts, and
        another leaves it where it ends. Entered grammars stack, so they can nest.

        The grammars are snapshotted, so changing them afterward does not affect this
        tokenizer. Inside a checkpoint, entering is undone with everything else.

        :param legal: The rules tried first at each position
        :param illegal: The rules tried where no legal rule matches. When omitted, a
            lexer's own illegal rules are used, and none for a plain grammar
        :raises TypeError: If either is not a grammar, or uses a pattern type other
            than the source's
        :raises ValueError: If ``legal`` has no rules, since nothing could ever match
        """
        if not isinstance(legal, Grammar):
            raise TypeError(f"legal rules must be a Grammar, got {type(legal).__qualname__}")
        # Checked before reading illegal: on an empty Lexer that property raises
        # RuntimeError, which getattr does not catch.
        if legal._type is None:
            raise ValueError("legal rules are empty")
        # A Lexer brings its own illegal rules, a plain Grammar gets an empty set. getattr
        # instead of isinstance, since importing Lexer here would be circular.
        if illegal is None and (illegal := getattr(legal, "illegal", None)) is None:
            # Fix its pattern type to the legal rules', as Lexer.illegal does.
            (illegal := Grammar[Type, AnyStr]())._type = legal._type
        self._frames.append(self._frame(legal, illegal))

    def leave(self) -> None:
        """
        Go back to the rules that were in use before the matching :meth:`enter`.

        Inside a checkpoint, leaving is undone with everything else.

        :raises RuntimeError: If nothing was entered, since the lexer's own rules
            can never be left
        """
        # The base frame holds the lexer's own rules, so there is always a grammar to lex with.
        if len(self._frames) == 1:
            raise RuntimeError("cannot leave the lexer's own rules, every leave() needs a matching enter()")
        self._frames.pop()

    @contextlib.contextmanager
    def checkpoint(self) -> Generator[Self]:
        """
        Look ahead, then go back to where the tokenizer was.

        Everything done inside the block is undone when it ends, however it ends:
        tokens taken with ``next()``, input consumed with :meth:`move` or
        :meth:`mint`, and grammars entered or left. Checkpoints can nest. This is
        what lets a hook peek at the following tokens before deciding::

            with tokenizer.checkpoint():
                following = next(tokenizer, None)

        Use ``next(tokenizer, None)`` rather than ``next(tokenizer)``, since the end
        of the input may be reached while looking ahead.

        :return: A context manager yielding this tokenizer
        """
        state = (self._pos, self._line, self._col, self._frames, self._stalled)
        # Work on a copy of the stack, so enter() and leave() inside the block can't
        # touch the one being restored. Frames are immutable, so a shallow copy will do.
        self._frames = self._frames.copy()
        try:
            yield self
        finally:
            self._pos, self._line, self._col, self._frames, self._stalled = state

    def scan(self, pattern: AnyPattern[AnyStr], /, flags: int = 0) -> Match[AnyStr] | None:
        """
        Match a pattern at the current position, without consuming anything.

        Lets a hook look at what comes next with a pattern of its own, outside the
        grammar. The result can be passed to :meth:`move` or :meth:`mint` to consume
        it, since it is made on this source at the current position.

        :param pattern: The pattern source, or a compiled pattern
        :param flags: Regex flags, combined with those of a compiled pattern
        :return: The match, or ``None`` if the pattern does not match here
        :raises TypeError: If the pattern type differs from the source's
        :raises regex.error: If the pattern does not compile
        """
        # Regex normalizes the flags and caches the compiled pattern, so scanning with
        # the same pattern again doesn't compile it again.
        if not isinstance((rx := Regex(pattern, flags=flags)).regex, type(self._src)):
            raise TypeError(f"expected a {type(self._src).__qualname__} pattern, got {type(rx.regex).__qualname__}")
        # Bounded by endpos, so a match can never reach past what the tokenizer may consume.
        return rx.pattern.match(self.src, self.pos, self.endpos)

    def _check_match(self, match: Match[AnyStr], /) -> None:
        """
        Check that a match can be consumed: made on this source, at the current position.

        A match made outside the tokenizer is accepted too, as long as it was made on
        the same source and starts where the tokenizer is.

        :param match: The match to check
        :raises TypeError: If ``match`` is not a match
        :raises ValueError: If it was made on other text, does not start at the
            current position, or ends past the end position
        """
        if not isinstance(match, Match):
            raise TypeError(f"expected a Match, got {type(match).__qualname__}")
        # Identity is the normal case and costs nothing. Equality accepts an equal copy,
        # whose offsets are the same, but rejects a slice, whose offsets would be wrong.
        if match.string is not self._src and match.string != self._src:
            raise ValueError("match was made against a different text than the tokenizer's source")
        # span(), not pos and endpos: those are where the search was allowed to look,
        # not where the match is.
        pos, endpos = match.span(0)
        # Starting exactly at the current position keeps tokens contiguous: a later
        # start would silently drop the text in between.
        if pos != self._pos or endpos > self._endpos:
            raise ValueError(f"match spans {pos}..{endpos}, but it must start at {self._pos} and end by {self._endpos}")

    def _check_token(self, token: Token[Type, AnyStr], /) -> Token[Type, AnyStr]:
        """

        :param token:
        :return:
        """
        if not isinstance(token, Token):
            raise TypeError(f"expected a Token, got {type(token).__qualname__}")
        if not isinstance(token.data, type(self._src)):
            raise TypeError(f"token holds {type(token.data).__qualname__} data, but the source is {type(self._src).__qualname__}")
        return token

    def move(self, match: Match[AnyStr], /) -> None:
        """
        Consume a match, moving the position and the line and column past it.

        Use it for input that should produce no token, such as text a hook skips.
        To make a token from the match as well, use :meth:`mint`, which moves too.

        :param match: A match on this source, starting at the current position
        :raises TypeError: If ``match`` is not a match
        :raises ValueError: If it was made on other text, does not start at the
            current position, or ends past the end position
        """
        self._check_match(match)
        pos, endpos = match.span(0)
        match self._src:
            case str():
                nl: Literal['\n'] = "\n"
            case bytes():
                nl: Literal[b'\n'] = b"\n"
            case src:
                typing.assert_never(src)
        # Every line break in the match starts a new line, and columns restart at 1.
        if nls := self._src.count(nl, pos, endpos):  # type: ignore[overloaded-call]
            self._col = 1
        self._line += nls
        # Count the columns of the match's last line only: the text after its last line
        # break, or all of it when there is none (rfind gives -1, so max picks pos).
        self._col += endpos - max(self._src.rfind(nl, pos, endpos) + 1, pos)  # type: ignore[overloaded-call]
        self._pos = endpos

    def mint(self, match: Match[AnyStr], type: Type, /) -> Token[Type, AnyStr]:
        """
        Consume a match and make a token from it.

        This is the default minter: it is what the tokenizer uses for a rule's type,
        and what a hook uses to make its own token. The token covers exactly the
        match, with its positions taken from where the tokenizer was before and after.

        :param match: A match on this source, starting at the current position
        :param type: The token type
        :return: The token for the match
        :raises TypeError: If ``match`` is not a match, or ``type`` is not an enum member
        :raises ValueError: If the match was made on other text, does not start at
            the current position, or ends past the end position
        """
        # Checked before moving, so a bad call consumes nothing.
        if not isinstance(type, Enum):
            raise TypeError(f"token type must be an Enum member, got {builtins.type(type).__qualname__}")
        head: tuple[int, int] = self.cursor
        self.move(match)
        tail: tuple[int, int] = self.cursor
        return Token(
            type=type,
            data=match.group(0),
            span=match.span(0),
            head=head,
            tail=tail,
        )

    def _next_token(self) -> Token[Type, AnyStr] | None:
        """
        Match the best rule at the current position and apply it.

        :return: The rule's token, or None if it produced none
        :raises SyntaxError: If neither the legal nor the illegal rules match
        :raises RuntimeError: If a rule keeps matching without consuming any input
        """
        frame, pos = self._frames[-1], self._pos
        # Two steps in a row consuming nothing at the same place in the same grammar
        # would repeat forever, typically a skipped pattern that can match empty, like \s*.
        if self._stalled is not None and self._stalled[0] == pos and self._stalled[1] is frame:
            raise RuntimeError(f"no progress at line {self._line}, column {self._col}: the rule for {self._stalled[2]!r} matched without consuming any input")

        result: tuple[Grammar.Rule[Type, AnyStr] | Grammar.Skip, Match[AnyStr]] | None = frame.legal.match(self.src, pos, self._endpos)
        if result is None:
            result = frame.illegal.match(self.src, pos, self._endpos)
        if result is None:
            raise SyntaxError(
                f"no rule matches at line {self._line}, column {self._col}",
                (None, self._line, self._col, self._current_line()),
            )

        rule, match = result
        token: Token[Type, AnyStr] | None = self._apply(rule, match)  # type: ignore[assignment]
        # Remember a step that consumed nothing, so the next one can tell if it repeats.
        # A hook that entered or left a grammar changes the frame, so that isn't a stall.
        self._stalled = (pos, frame, match.re.pattern) if self._pos == pos else None
        return token

    def _apply(self, rule: Grammar.Rule[Type, AnyStr] | Grammar.Skip, match: Match[AnyStr], /) -> Token[Type, AnyStr] | None:
        """
        Turn a rule's match into a token, running its hooks.

        :param rule: The matched rule, or Skip
        :param match: Its match at the current position
        :return: The token, or None if the match or the token is dropped
        :raises TypeError: If a hook returns something it may not
        :raises ValueError: If on_match returns a type after consuming input
        """
        if rule is Grammar.Skip:
            self.move(match)
            return None

        type: Type | None = rule.type
        token: Token[Type, AnyStr] | None = None

        if rule.on_match is not None:
            result: Grammar.Rule[Type, AnyStr] | Grammar.Skip | Type = rule.on_match(self, match)
            if result is Grammar.Skip:
                # The match is dropped: no token is made, so on_token is never called.
                # Consume the match unless the hook already moved on by itself.
                if self._pos == match.start():
                    self.move(match)
                return None
            if isinstance(result, Token):
                # The hook made the token itself, normally with mint(), which also
                # consumed the match.
                token: Token[Type, AnyStr] = self._check_token(result)  # type: ignore[assignment]
            elif isinstance(result, Enum):
                # The default minter consumes the match, which only works if the hook
                # hasn't consumed anything yet.
                if self._pos != match.start():
                    raise ValueError(f"on_match hook {_hook_name(rule.on_match)} returned a token type after consuming input, it should return the token it made instead")
                type: Type = result
            else:
                raise TypeError(f"on_match hook {_hook_name(rule.on_match)} returned {builtins.type(result).__qualname__}, expected a Token, a token type or Grammar.Skip")

        if token is None:
            # Grammar guarantees a type when there is no on_match hook, and a hook that
            # returned a type set it above.
            token = self.mint(match, type)  # type: ignore[arg-type]

        if rule.on_token is not None:
            result: Grammar.Rule[Type, AnyStr] | Grammar.Skip | Type | None = rule.on_token(self, token)  # type: ignore[arg-type]
            if result is Grammar.Skip:
                return None
            if isinstance(result, Token):
                self._check_token(result)
                token = result
            elif isinstance(result, Enum):
                token = dataclasses.replace(token, type=result)  # type: ignore[arg-type]
            elif result is not None:
                raise TypeError(f"on_token hook {_hook_name(rule.on_token)} returned {builtins.type(result).__qualname__}, expected a Token, a token type, Grammar.Skip or None")
        return token

    def _current_line(self) -> str:
        """
        Get the source line at the current position, for error messages.

        :return: The line without its break, decoded for bytes sources
        """
        nl = "\n" if isinstance(self._src, str) else b"\n"
        start = self._src.rfind(nl, 0, self._pos) + 1  # type: ignore[arg-type]
        if (end := self._src.find(nl, self._pos)) < 0:  # type: ignore[arg-type]
            end = self._endpos
        line: str | bytes = self._src[start:end]
        # SyntaxError.text must be a str for tracebacks to display it.
        return line.decode("utf-8", "replace") if isinstance(line, bytes) else line

    def __next__(self) -> Token[Type, AnyStr]:
        """
        Make the next token.

        :return: The next token of the source
        :raises StopIteration: At the end of the source
        :raises SyntaxError: If no rule matches at the current position
        :raises RuntimeError: If a rule keeps matching without consuming input, or a
            hook lets StopIteration escape
        """
        # A loop rather than recursion, so long runs of skipped input can't overflow
        # the stack.
        while self._pos < self._endpos:
            try:
                token: Token[Type, AnyStr] | None = self._next_token()  # type: ignore[assignment]
            except StopIteration as error:
                # Escaping __next__, it would silently end the iteration and lose tokens.
                # It typically comes from a hook looking ahead with next(tokenizer).
                raise RuntimeError("a hook raised StopIteration, use next(tokenizer, None) to look ahead") from error
            if token is None:
                continue
            return token
        raise StopIteration
