"""
Grammars: which patterns to match, and what their matches become.

A :class:`Grammar` maps each pattern to a rule: a token type for the default
minter, hooks that decide or change the token, or :attr:`Grammar.Skip` for text
that is consumed without making one.
"""

import builtins
import dataclasses
from collections.abc import Callable, Iterator, Mapping
from copy import copy, deepcopy
from enum import Enum
from typing import ClassVar, Self, TYPE_CHECKING

from regex import Match

from .alternations import Alternations
from .. import regex
from ..regex import AnyPattern, Regex, AnyRegex

if TYPE_CHECKING:
    from ..token import Token
    from ..tokenizer import Tokenizer


type OnMatchHook[Type: Enum, AnyStr: (str, bytes)] = Callable[
    [Tokenizer[Type, AnyStr], Match[AnyStr]],
    Token[Type, AnyStr] | Type | Grammar.Skip,
]
type OnTokenHook[Type: Enum, AnyStr: (str, bytes)] = Callable[
    [Tokenizer[Type, AnyStr], Token[Type, AnyStr]],
    Token[Type, AnyStr] | Type | Grammar.Skip | None,
]


class Grammar[Type: Enum, AnyStr: (str, bytes)](Mapping[Regex[AnyStr], 'Grammar.Rule[Type, AnyStr] | Skip']):
    """
    The lexing rules of a language: which patterns to match, and what each match becomes.

    Maps each pattern to its :class:`Rule`, or to :attr:`Skip` for text that is
    matched and dropped, such as whitespace. Iteration follows priority order,
    which is the order the patterns are tried in. All patterns share one type,
    ``str`` or ``bytes``: the first rule fixes it.

    The mapping is read-only. Rules are added with :meth:`rule`, or with the
    :meth:`skip`, :meth:`on_match` and :meth:`on_token` shortcuts, and whole
    grammars with :meth:`include`. A defined rule can never be changed, except
    for adding an ``on_token`` hook it does not have yet.

    Hooks receive the tokenizer as their first argument, because lexing must be
    repeatable: the tokenizer can checkpoint, so a hook may run for tokens that
    are generated and then thrown away, inside :meth:`Tokenizer.checkpoint`.
    A hook should therefore keep all of its state in the tokenizer, where it is
    rolled back with everything else. Anything it does outside, such as logging
    or changing global state, may happen more than once or for tokens that are
    never kept.

    :cvar Skip: Marks a rule whose matches are dropped, and is returned by a hook
        to drop the match or token it was given
    """
    Skip: ClassVar[sentinel] = sentinel('Skip')

    @dataclasses.dataclass(slots=True, frozen=True)
    class Rule[RuleType: Enum, DataType: (str, bytes)]:
        """
        What a pattern's match becomes.

        A rule has either a ``type`` or an ``on_match`` hook, never both, since
        both decide how the token is made. An ``on_token`` hook can be added to
        either.

        :ivar type: The token type, used by the default minter to make the token
            from the match
        :ivar on_match: Decides how to make the token from the match: it returns
            the token itself, a type for the default minter to use, or
            :attr:`Grammar.Skip` to drop the match
        :ivar on_token: Runs once the token is made, for example to log it,
            replace it or drop it: it returns a replacement token, a type to make
            it with instead, :attr:`Grammar.Skip` to drop it, or ``None`` to keep
            it unchanged. It is never called for a match ``on_match`` dropped
            with :attr:`Grammar.Skip`, since no token is made then
        """
        type: RuleType | None = None
        on_match: OnMatchHook[RuleType, DataType] | None = None
        on_token: OnTokenHook[RuleType, DataType] | None = None

    @property
    def rules(self) -> frozendict[Regex[AnyStr], Rule[Type, AnyStr] | Skip]:
        """
        A snapshot of every rule by pattern, in priority order.

        :return: The rules, which later changes to the grammar do not affect
        """
        return frozendict({rx: self._rules[rx] for rx in self._alternations})

    @property
    def alternations(self) -> frozendict[Regex[AnyStr], int]:
        """
        A snapshot of every pattern's priority, highest first.

        :return: The priorities, which later changes to the grammar do not affect
        """
        return frozendict(self._alternations)

    def __init__(self) -> None:
        """
        Create an empty grammar, whose pattern type is fixed by the first rule.
        """
        self._type: type[AnyStr] | None = None
        self._rules: dict[Regex[AnyStr], Grammar.Rule[Type, AnyStr] | Grammar.Skip] = {}
        self._alternations: Alternations[AnyStr, int] = Alternations[AnyStr, int]()

    def __getitem__(self, key: AnyRegex[AnyStr], /) -> Rule[Type, AnyStr] | Skip:
        """
        Look up the rule of a pattern.

        :param key: The pattern, in any form :func:`regex.coerce` accepts
        :return: Its rule, or :attr:`Skip` if its matches are dropped
        :raises KeyError: If the pattern has no rule, including a key that could
            never be a pattern because of its form
        """
        try:
            rx: Regex[AnyStr] = regex.coerce(key)  # type: ignore[union-attr]
        except TypeError:
            raise KeyError(key) from None
        return self._rules[rx]

    def __iter__(self) -> Iterator[Regex[AnyStr]]:
        """
        Iterate over the patterns in the order they are tried, highest priority first.

        :return: An iterator over the patterns
        :raises RuntimeError: If the grammar changes during iteration
        """
        return iter(self._alternations)

    def __len__(self) -> int:
        """
        Count the rules.

        :return: The number of patterns with a rule, skipped ones included
        """
        return len(self._rules)

    def rule(
            self,
            rx: AnyPattern[AnyStr],
            type: Type | None = None,
            /,
            priority: int | None = None,
            *,
            flags: int = 0,
            on_match: OnMatchHook[Type, AnyStr] | None = None,
            on_token: OnTokenHook[Type, AnyStr] | None = None,
    ) -> None:
        """
        Define what a pattern's matches become, or add an ``on_token`` hook to an existing rule.

        With a ``type``, the default minter makes the token. With ``on_match``, the
        hook decides instead. With neither, matches are skipped. An ``on_token``
        hook can come with a new rule, or be added later to a rule that has none.

        :param rx: The pattern source, or a compiled pattern
        :param type: The token type for the default minter
        :param priority: Patterns with higher priority are tried first, ``0`` when
            omitted for a new rule. It cannot be given when adding an
            ``on_token`` hook to an existing rule
        :param flags: Regex flags, combined with those of a compiled pattern
        :param on_match: Decides how to make the token, instead of ``type``
        :param on_token: Runs once the token is made
        :raises TypeError: If the pattern type differs from the grammar's
        :raises ValueError: If both ``type`` and ``on_match`` are given, if the
            pattern already has a rule this call would change, or if
            ``on_token`` is given for a new rule that never makes a token
        :raises regex.error: If the pattern does not compile
        """
        if not isinstance((rx :=  Regex(rx, flags)).regex, self._type or (str, bytes)):
            raise TypeError(f"expected {getattr(self._type, "__qualname__", "str or bytes")} pattern, got {builtins.type(rx.regex).__qualname__}")

        if type is not None and on_match is not None:
            raise ValueError(f"rule {rx} cannot have both a type and an on_match hook, since the hook decides the type")

        existing: Grammar.Rule[Type, AnyStr] | Grammar.Skip = self._rules.get(rx)
        if existing is Grammar.Skip:
            raise ValueError(f"rule {rx} is already defined as skipped")
        if existing is not None:
            # A defined rule only accepts an added on_token hook, everything else would
            # redefine it.
            if type is not None or on_match is not None or on_token is None:
                raise ValueError(f"rule {rx} is already defined, only an on_token hook can be added to it")
            if existing.on_token is not None:
                raise ValueError(f"rule {rx} already has an on_token hook")
            if priority is not None:
                raise ValueError(f"rule {rx} already has a priority, adding an on_token hook cannot change it")
            # Rule is frozen and hashable, so replace it instead of mutating it.
            self._rules[rx] = dataclasses.replace(existing, on_token=on_token)
            return

        if type is None and on_match is None:
            # Without a type or an on_match hook no token is ever produced, so an
            # on_token hook would never run.
            if on_token is not None:
                raise ValueError(f"rule {rx} needs a type or an on_match hook to produce the tokens its on_token hook receives")
            self._rules[rx] = Grammar.Skip
        else:
            self._rules[rx] = Grammar.Rule(type, on_match, on_token)
        # The public setter, so Alternations drops its cached order.
        self._alternations[rx] = 0 if priority is None else priority
        if self._type is None:
            self._type = builtins.type(rx.regex)  # type: ignore[assignment]

    def skip(
            self,
            rx: AnyPattern[AnyStr],
            /,
            priority: int = 0,
            *,
            flags: int = 0,
    ) -> None:
        """
        Drop the matches of a pattern, such as whitespace or comments.

        The text is still consumed, it just produces no token.

        :param rx: The pattern source, or a compiled pattern
        :param priority: Patterns with higher priority are tried first
        :param flags: Regex flags, combined with those of a compiled pattern
        :raises TypeError: If the pattern type differs from the grammar's
        :raises ValueError: If the pattern already has a rule
        :raises regex.error: If the pattern does not compile
        """
        self.rule(
            rx,
            priority=priority,
            flags=flags,
        )

    def on_match(
            self,
            rx: AnyPattern[AnyStr],
            /,
            priority: int = 0,
            *,
            flags: int = 0,
    ) -> Callable[[OnMatchHook[Type, AnyStr]], OnMatchHook[Type, AnyStr]]:
        """
        Decorate a hook that decides how a pattern's matches become tokens.

        The hook receives the tokenizer and the match, and returns the token, a
        type for the default minter to use, or :attr:`Skip` to drop the match.
        A hook returning its own token consumes the input itself, normally with
        the tokenizer's ``mint``, while one returning a type or :attr:`Skip`
        leaves the match for the tokenizer to consume.

        When the hook returns :attr:`Skip`, no token is made, so an ``on_token``
        hook on the same pattern is never called for that match.

        The hook is returned unchanged, so decorators stack to reuse one hook for
        several patterns::

            @grammar.on_match(r"\\d+")
            @grammar.on_match(r"0x[0-9a-f]+", flags=regex.IGNORECASE)
            def number(tokenizer, match):
                ...

        :param rx: The pattern source, or a compiled pattern
        :param priority: Patterns with higher priority are tried first
        :param flags: Regex flags, combined with those of a compiled pattern
        :return: A decorator that registers the hook and returns it unchanged
        :raises TypeError: When decorating, if the hook is not callable or the
            pattern type differs from the grammar's
        :raises ValueError: When decorating, if the pattern already has a rule
        :raises regex.error: When decorating, if the pattern does not compile
        """

        def decorator(hook: OnMatchHook[Type, AnyStr]) -> OnMatchHook[Type, AnyStr]:
            """
            Register ``hook`` as the pattern's on_match hook.

            :param hook: The hook to register
            :return: The same hook, so further decorators can register it too
            """
            if not callable(hook):
                raise TypeError(f"on_match hook must be callable, got {builtins.type(hook).__qualname__}")
            self.rule(
                rx,
                priority=priority,
                flags=flags,
                on_match=hook,
            )
            return hook

        return decorator

    def on_token(
            self,
            rx: AnyPattern[AnyStr],
            type: Type | None = None,
            /,
            priority: int | None = None,
            *,
            flags: int = 0,
    ) -> Callable[[OnTokenHook[Type, AnyStr]], OnTokenHook[Type, AnyStr]]:
        """
        Decorate a hook that runs once a pattern's token is made.

        Use it for anything that needs the finished token, such as logging,
        rewriting or dropping it. The hook receives the tokenizer and the token,
        and returns a replacement token, a type to make it with instead,
        :attr:`Skip` to drop it, or ``None`` to keep it unchanged.

        It only runs when a token is actually made: if the rule's ``on_match``
        hook returns :attr:`Skip`, it is never called for that match.

        With a ``type``, this defines a new rule with the hook. Without one, it
        adds the hook to the pattern's existing rule. The hook is returned
        unchanged, so decorators stack to reuse one hook for several patterns::

            @grammar.on_token(r"[a-z_]\\w*", Kind.NAME)
            @grammar.on_token(r"\\d+", Kind.NUMBER)
            def trace(tokenizer, token):
                ...

        :param rx: The pattern source, or a compiled pattern
        :param type: The token type for a new rule, omitted to add the hook to an
            existing one
        :param priority: Patterns with higher priority are tried first, ``0`` when
            omitted for a new rule. It cannot be given for an existing rule
        :param flags: Regex flags, combined with those of a compiled pattern
        :return: A decorator that registers the hook and returns it unchanged
        :raises TypeError: When decorating, if the hook is not callable or the
            pattern type differs from the grammar's
        :raises ValueError: When decorating, if a new rule has no ``type``, or the
            existing rule already has an ``on_token`` hook or is skipped
        :raises regex.error: When decorating, if the pattern does not compile
        """

        def decorator(hook: OnTokenHook[Type, AnyStr]) -> OnTokenHook[Type, AnyStr]:
            """
            Register ``hook`` as the pattern's on_token hook.

            :param hook: The hook to register
            :return: The same hook, so further decorators can register it too
            """
            if not callable(hook):
                raise TypeError(f"on_token hook must be callable, got {builtins.type(hook).__qualname__}")
            self.rule(
                rx,
                type,
                priority=priority,
                flags=flags,
                on_token=hook,
            )
            return hook

        return decorator

    def __copy__(self) -> Self:
        """
        Make an independent copy of the grammar.

        Patterns, rules and hooks are shared, which is safe since patterns and
        rules are immutable. Adding rules to either grammar leaves the other
        unchanged.

        :return: A new grammar with the same rules and priorities
        """
        cpy: Self = (cls := type(self)).__new__(cls)  # type: ignore[assignment]
        cpy._type = self._type
        cpy._rules = copy(self._rules)
        cpy._alternations = copy(self._alternations)
        return cpy

    def __deepcopy__(self, memo: dict[int, object], /) -> Self:
        """
        Make an independent copy of the grammar, copying patterns and rules too.

        Hooks are still shared, since functions are never copied, and
        :attr:`Skip` keeps its identity.

        :param memo: The :func:`copy.deepcopy` memo
        :return: A new grammar with the same rules and priorities
        """
        memo[id(self)] = cpy = (cls := type(self)).__new__(cls)
        cpy._type = self._type
        cpy._rules = deepcopy(self._rules, memo)
        cpy._alternations = deepcopy(self._alternations, memo)
        return cpy  # type: ignore[return-value]

    def include(self, grammar: "Grammar[Type, AnyStr]", /) -> None:
        """
        Add every rule of ``grammar`` to this grammar, with its priority.

        The grammars must not share a pattern. Since equal priorities keep insertion
        order, rules already here win ties against the included ones.

        :param grammar: The grammar whose rules to add, left unchanged
        :raises TypeError: If the grammars use different pattern types
        :raises ValueError: If both grammars define the same pattern
        """
        if grammar._type is None:
            return
        if not issubclass(grammar._type, self._type or (str, bytes)):
            raise TypeError( f"cannot include a {grammar._type.__qualname__} grammar into a {self._type.__qualname__} one")  # type: ignore[union-attr]

        # Check every conflict first, so a failed include leaves this grammar unchanged.
        if conflicts := self.keys() & grammar.keys():
            raise ValueError(f"rules defined in both grammars: {', '.join(sorted(map(str, conflicts)))}")
        # noinspection protected-member
        for rx, priority in grammar._alternations.items():
            # Regex() doesn't accept a Regex, so rebuild each key from its source and flags.
            if (rule := grammar._rules[rx]) is Grammar.Skip:
                self.rule(
                    rx.regex,
                    priority=priority,
                    flags=rx.flags,
                )
            else:
                self.rule(
                    rx.regex,
                    rule.type,
                    priority=priority,
                    flags=rx.flags,
                    on_match=rule.on_match,
                    on_token=rule.on_token,
                )

    def match(
            self,
            input: AnyStr,
            /,
            pos: int | None = None,
            endpos: int | None = None,
            timeout: float | int | None = None,
            *,
            partial: bool | None = None,
            concurrent: bool | None = None,
    ) -> tuple[Rule[Type, AnyStr] | Skip, Match[AnyStr]] | None:
        """
        Find the rule whose pattern matches best at ``pos``.

        The highest priority with any match wins, then the longest match within
        it, then the rule added first. Empty matches are allowed, so the caller is
        responsible for making progress. Only the match is made here, the rule's
        type and hooks are left for the tokenizer to apply.

        :param input: The source to match against, of the grammar's pattern type
        :param pos: Where matching starts, the beginning of ``input`` when omitted
        :param endpos: Where matching must stop, the end of ``input`` when omitted
        :param timeout: Seconds each pattern may run, so one call can take up to
            this long per pattern tried
        :param partial: Whether a match may stop incomplete at the end of
            ``input``, as in :meth:`regex.Pattern.match`
        :param concurrent: Whether to release the GIL while matching, as in
            :meth:`regex.Pattern.match`
        :return: The winning rule, or :attr:`Skip`, with its match, or ``None``
            if nothing matches
        :raises TypeError: If ``input`` is not of the grammar's pattern type
        :raises TimeoutError: If a pattern runs longer than ``timeout``
        """
        result: tuple[Regex[AnyStr], Match[AnyStr]] | None = self._alternations.match(
            input,
            pos,
            endpos,
            timeout=timeout,
            partial=partial,
            concurrent=concurrent,
        )
        if result is None:
            return None
        rx, match = result
        return self._rules[rx], match


__all__ = ("Grammar",)
