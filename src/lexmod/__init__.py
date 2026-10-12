"""
lexmod: modal lexing for Python.

Build tokenizers from regular expressions, and switch grammars as the input
changes: inside strings, comments, template fields or indented blocks. A rule's
hook can enter another grammar and leave it again, so the tokenizer carries a
stack of modes, and a checkpoint undoes everything for safe lookahead.

:class:`Lexer` is the entry point: define its rules, then call
:meth:`Lexer.tokenize`. It returns a :class:`Tokenizer`, which iterates over
:class:`Token` objects and is what hooks use to consume input, switch modes and
look ahead. :class:`Grammar` is a set of rules without illegal ones, for modes
and for composing lexers.
"""

from enum import Enum
from typing import Self

from .grammar import Grammar
from .token import Token
from .tokenizer import Tokenizer


class Lexer[Type: Enum, AnyStr: (str, bytes)](Grammar[Type, AnyStr]):
    """
    A grammar with fallback rules for mistakes, ready to tokenize a source.

    The lexer's own rules are its legal rules: they're added with the methods it
    inherits from :class:`Grammar`, and tried in priority order at each position.
    Its :attr:`illegal` rules are only tried where no legal rule matches, to
    recognize likely mistakes instead of stopping at the first one.

    ::

        lexer = Lexer[Kind, str]()
        lexer.skip(r"\\s+")
        lexer.rule(r"\\d+", Kind.NUMBER)
        lexer.rule(r"[a-z_]\\w*", Kind.NAME)
        lexer.illegal.rule(r'"[^"\\n]*', Kind.UNTERMINATED_STRING)

        for token in lexer.tokenize('total 42 "oops'):
            ...

    Since a lexer is also a grammar, it can be included into another lexer, or
    entered by a hook to lex part of a source differently, and it brings its
    illegal rules along in both cases.
    """

    @property
    def illegal(self) -> Grammar[Type, AnyStr]:
        """
        The fallback rules, tried only where no legal rule matches.

        They describe mistakes the user is likely to make, such as an unterminated
        string, so that input the legal rules reject can still be recognized
        and reported precisely instead of stopping the lexer. Since they are only
        tried after every legal rule failed, they never change how legal input
        lexes, whatever their priorities.

        Rules are added to the returned grammar directly, with the same methods as
        the lexer itself, and must use the lexer's pattern type.

        :return: The live grammar of illegal rules, so changes to it apply to
            this lexer
        :raises RuntimeError: If the lexer has no rules yet, since its pattern type
            is still unknown
        """
        if self._type is None:
            raise RuntimeError("define a rule before any illegal rule, so the pattern type is known")
        # Fix the illegal grammar's pattern type to the lexer's, so a rule of the
        # other type is rejected there too.
        if self._illegal._type is None:
            self._illegal._type = self._type
        return self._illegal

    def __init__(self) -> None:
        """
        Create an empty lexer, whose pattern type is fixed by the first rule.
        """
        super().__init__()
        self._illegal: Grammar[Type, AnyStr] = Grammar[Type, AnyStr]()

    def __copy__(self) -> Self:
        """
        Make an independent copy of the lexer, illegal rules included.

        Patterns, rules and hooks are shared, which is safe since patterns and
        rules are immutable. Adding rules to either lexer, or to either one's
        illegal rules, leaves the other unchanged.

        :return: A new lexer with the same rules, illegal rules and priorities
        """
        cpy: Self = super().__copy__()  # type: ignore[assignment]
        cpy._illegal = self._illegal.__copy__()
        return cpy

    def __deepcopy__(self, memo: dict[int, object], /) -> Self:
        """
        Make an independent copy of the lexer, copying patterns and rules too.

        Hooks are still shared, since functions are never copied.

        :param memo: The :func:`copy.deepcopy` memo
        :return: A new lexer with the same rules, illegal rules and priorities
        """
        cpy: Self = super().__deepcopy__(memo)  # type: ignore[assignment]
        cpy._illegal = self._illegal.__deepcopy__(memo)
        return cpy

    def include(self, grammar: Grammar[Type, AnyStr], /) -> None:
        """
        Add every rule of ``grammar`` to this lexer, with its priority.

        When ``grammar`` is a lexer, its illegal rules are added to this lexer's
        illegal rules as well. The grammars must not share a pattern, in their
        legal or in their illegal rules. Since equal priorities keep insertion
        order, rules already here win ties against the included ones.

        Nothing changes unless the whole include succeeds.

        :param grammar: The grammar or lexer whose rules to add, left unchanged
        :raises TypeError: If the grammars use different pattern types
        :raises ValueError: If both define the same pattern, among either their
            legal or their illegal rules
        """
        # A plain Grammar has no illegal rules to bring along.
        if not isinstance(grammar, Lexer):
            return super().include(grammar)
        # Check the illegal rules before changing anything, so a failed include
        # leaves both grammars of this lexer unchanged.
        if conflicts := self._illegal.keys() & grammar._illegal.keys():
            raise ValueError(f"illegal rules defined in both lexers: {', '.join(sorted(map(str, conflicts)))}")
        super().include(grammar)
        return self._illegal.include(grammar._illegal)

    def tokenize(self, src: AnyStr, /) -> Tokenizer[Type, AnyStr]:
        """
        Start tokenizing a source with this lexer's rules.

        Tokens are made lazily, one per ``next()``, so a large source is never
        tokenized all at once, and iteration can stop at any point. The rules are
        snapshotted here, so changing the lexer afterward doesn't affect the
        returned tokenizer.

        The source is normalized first: ``\\a``, ``\\b``, ``\\v``, ``\\f`` and ``\\r``
        are removed, so lines break at ``\\n`` only, and token offsets refer to the
        normalized text.

        :param src: The text to tokenize, of the lexer's pattern type
        :return: An iterator over the source's tokens
        :raises RuntimeError: If the lexer has no rules
        :raises TypeError: If ``src`` is not of the lexer's pattern type
        """
        if self._type is None:
            raise RuntimeError("cannot tokenize with a lexer that has no rules")
        if not isinstance(src, self._type):
            raise TypeError(f"expected {self._type.__qualname__} source, got {type(src).__qualname__}")
        return Tokenizer(src, self, self.illegal)


__all__ = ("Lexer", "Grammar", "Token", "Tokenizer")
