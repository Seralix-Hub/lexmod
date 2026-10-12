"""
The priority-ordered set of patterns a grammar tries at each position, and how
the best match among them is chosen.
"""

import typing
from collections.abc import Iterator, MutableMapping, KeysView, ValuesView, ItemsView
from typing import Self, Any

from regex import Match

from .. import regex
from ..regex import AnyRegex, Regex


@typing.final
class _AlternationsKeys[AnyStr: (str, bytes), Priority: int](KeysView[Regex[AnyStr]]):  # NOQA: F841
    """
    A live view of the alternatives in an :class:`Alternations`.

    Iterates in the same order as the mapping, highest priority first, and
    reflects later changes to it.
    """

    def __iter__(self) -> Iterator[Regex[AnyStr]]:
        """
        Iterate over the alternatives from highest to lowest priority.

        :return: An iterator over the alternatives
        :raises RuntimeError: If the order changes during iteration
        """
        yield from self._mapping


@typing.final
class _AlternationsValues[AnyStr: (str, bytes), Priority: int](ValuesView[Priority]):  # NOQA: F841
    """
    A live view of the priorities in an :class:`Alternations`.

    Iterates in the same order as the mapping, highest priority first, and
    reflects later changes to it.
    """

    def __iter__(self) -> Iterator[Priority]:
        """
        Iterate over the priorities, in the order of their alternatives.

        :return: An iterator over the priorities
        :raises RuntimeError: If the order changes during iteration
        """
        # Read the inner dict, since looking each key up through the mapping would
        # coerce and type-check it again.
        # noinspection protected-member
        yield from map(self._mapping._mapping.__getitem__, self._mapping)


@typing.final
class _AlternationsItems[AnyStr: (str, bytes), Priority: int](ItemsView[Regex[AnyStr], Priority]):
    """
    A live view of the ``(alternative, priority)`` pairs in an :class:`Alternations`.

    Iterates in the same order as the mapping, highest priority first, and
    reflects later changes to it.
    """

    def __iter__(self) -> Iterator[tuple[Regex[AnyStr], Priority]]:
        """
        Iterate over the pairs from highest to lowest priority.

        :return: An iterator over ``(alternative, priority)`` pairs
        :raises RuntimeError: If the order changes during iteration
        """
        # One pass over the ordered keys, reading each priority from the inner dict,
        # since looking it up through the mapping would coerce the key again.
        # noinspection protected-member
        yield from map(lambda rx: (rx, self._mapping._mapping[rx]), self._mapping)


class Alternations[AnyStr: (str, bytes), Priority: int](MutableMapping[Regex[AnyStr], Priority]):
    """
    A priority-ordered set of alternative patterns, the candidates tried at one lexing position.

    Maps each :class:`Regex` to its priority. Keys may be given in any form
    :func:`regex.coerce` accepts, and all of them must share one pattern type,
    ``str`` or ``bytes``: the first key fixes it, and it is free again once the
    mapping is empty.

    Iteration yields the alternatives from highest to lowest priority, equal
    priorities in the order they were first added, and the :meth:`keys`,
    :meth:`values` and :meth:`items` views follow the same order. That order is
    cached and only rebuilt after a change that can affect it.
    """

    def __init__(self) -> None:
        """
        Create an empty set of alternatives.
        """
        # Bumped on every change that affects the order, so running iterations can detect it.
        self._version: int = 0
        self._type: type[AnyStr] | None = None
        self._ordered: tuple[Regex[AnyStr], ...] | None = None
        self._mapping: dict[Regex[AnyStr], int] = {}

    def _invalidate(self) -> None:
        """
        Drop the cached order and signal running iterations that it changed.
        """
        self._version += 1
        self._ordered = None

    def _coerce(self, key: AnyRegex[AnyStr], /) -> Regex[AnyStr]:
        """
        Convert a key to a :class:`Regex` and check it has the alternatives' pattern type.

        :param key: The pattern, in any form :func:`regex.coerce` accepts
        :return: The key as a :class:`Regex`
        :raises TypeError: If ``key`` is not an accepted form, or its pattern type
            differs from the alternatives already stored
        """
        if not isinstance((key := regex.coerce(key)).regex, self._type or (str, bytes)):
            raise TypeError(f"expected {getattr(self._type, "__qualname__", "str or bytes")} pattern, got {type(key.regex).__qualname__}")
        return key

    def __setitem__(self, key: AnyRegex[AnyStr], value: Priority, /) -> None:
        """
        Add an alternative, or change the priority of an existing one.

        :param key: The pattern, in any form :func:`regex.coerce` accepts
        :param value: Its priority, higher values are tried first
        :raises TypeError: If ``key`` is not an accepted form, or its pattern type
            differs from the alternatives already stored
        """
        # Re-adding a pattern with its current priority leaves the order intact, so
        # only a new pattern or a changed priority drops the cached order.
        if self._mapping.get(rx := self._coerce(key)) != value:  # type: ignore[arg-type]
            self._invalidate()
        self._mapping[rx] = value  # type: ignore[arg-type]
        if self._type is None:
            self._type = type(rx.regex)

    def __delitem__(self, key: Regex[AnyStr], /) -> None:
        """
        Remove an alternative.

        Removing the last one frees the pattern type, so the next key may be
        ``str`` or ``bytes`` again.

        :param key: The pattern, in any form :func:`regex.coerce` accepts
        :raises KeyError: If the pattern is not stored, including a key that could
            never be stored because of its form or pattern type
        """
        try:
            rx: Regex[AnyStr] = self._coerce(key)  # type: ignore[assignment]
        except TypeError:
            raise KeyError(key) from None
        del self._mapping[rx]
        self._invalidate()
        if not self._mapping:
            self._type = None

    def __getitem__(self, key: Regex[AnyStr], /) -> Priority:
        """
        Look up the priority of an alternative.

        :param key: The pattern, in any form :func:`regex.coerce` accepts
        :return: Its priority
        :raises KeyError: If the pattern is not stored, including a key that could
            never be stored because of its form or pattern type
        """
        try:
            rx: Regex[AnyStr] = self._coerce(key)  # type: ignore[assignment]
        except TypeError:
            raise KeyError(key) from None
        return self._mapping[rx]

    def __len__(self) -> int:
        """
        Count the alternatives.

        :return: The number of stored alternatives
        """
        return len(self._mapping)

    def __iter__(self) -> Iterator[Regex[AnyStr]]:
        """
        Iterate over the alternatives from highest to lowest priority.

        Equal priorities come in the order they were first added, since sorting is
        stable over the mapping's insertion order.

        :return: An iterator over the alternatives
        :raises RuntimeError: If the order changes during iteration
        """
        if self._ordered is None:
            self._ordered = tuple(sorted(
                self._mapping, key=self._mapping.__getitem__, reverse=True
            ))
        version: int = self._version
        for rx in self._ordered:  # type: ignore[misc]
            # Same guard as dict iteration: the snapshot would silently go stale otherwise.
            if self._version != version:
                raise RuntimeError("alternations changed during iteration")
            yield rx

    def __copy__(self) -> Self:
        """
        Make an independent copy of the alternatives.

        The patterns themselves are shared, which is safe since :class:`Regex` is
        immutable. The cached order is shared too, since it is a tuple that only
        ever gets replaced, never changed.

        :return: A new instance with the same alternatives and priorities
        """
        cpy: Self = (cls := type(self)).__new__(cls)  # type: ignore[assignment]
        cpy._version = self._version
        cpy._type = self._type
        cpy._ordered = self._ordered
        cpy._mapping = self._mapping.copy()
        return cpy

    def __deepcopy__(self, memo: dict[int, Any], /) -> Self:
        """
        Same as a shallow copy, since everything a deep copy would duplicate is immutable.

        :param memo: The :func:`copy.deepcopy` memo, unused
        :return: A new instance with the same alternatives and priorities
        """
        return self.__copy__()

    def keys(self) -> _AlternationsKeys[AnyStr, Priority]:
        """
        Get a live view of the alternatives, highest priority first.

        :return: A view that reflects later changes to the alternatives
        """
        return _AlternationsKeys(self)

    def values(self) -> _AlternationsValues[AnyStr, Priority]:
        """
        Get a live view of the priorities, in the order of their alternatives.

        Unlike the default view, it reads the priorities without coercing each key
        again.

        :return: A view that reflects later changes to the alternatives
        """
        return _AlternationsValues(self)

    def items(self) -> _AlternationsItems[AnyStr, Priority]:
        """
        Get a live view of the ``(alternative, priority)`` pairs, highest priority first.

        Unlike the default view, it reads the priorities without coercing each key
        again, so it is cheap enough for the matching loop.

        :return: A view that reflects later changes to the alternatives
        """
        return _AlternationsItems(self)

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
    ) -> tuple[Regex[AnyStr], Match[AnyStr]] | None:
        """
        Find the alternative that matches best at ``pos``.

        The highest priority with any match wins. Within that priority the longest
        match wins, and on equal length the alternative added first. Once a
        priority has matched, lower priorities are never tried.

        Empty matches are allowed and win when nothing longer matches at the
        winning priority, so the caller is responsible for making progress.

        :param input: The source to match against, of the same type as the patterns
        :param pos: Where matching starts, the beginning of ``input`` when omitted
        :param endpos: Where matching must stop, the end of ``input`` when omitted
        :param timeout: Seconds each pattern may run, so one call can take up to
            this long per alternative tried
        :param partial: Whether a match may stop incomplete at the end of
            ``input``, as in :meth:`regex.Pattern.match`
        :param concurrent: Whether to release the GIL while matching, as in
            :meth:`regex.Pattern.match`
        :return: The winning alternative and its match, or ``None`` if nothing matches
        :raises TypeError: If ``input`` is not of the alternatives' pattern type
        :raises TimeoutError: If a pattern runs longer than ``timeout``
        :raises RuntimeError: If the alternations change during the search
        """
        if not isinstance(input, self._type or (str, bytes)):
            raise TypeError(f"expected {getattr(self._type, "__qualname__", "str or bytes")} input, got {type(input).__qualname__}")

        # Only pass what the caller set, so regex's own defaults apply otherwise.
        options: dict[str, Any] = {}
        if timeout is not None:
            options["timeout"] = timeout
        if partial is not None:
            options["partial"] = partial
        if concurrent is not None:
            options["concurrent"] = concurrent

        (best_rx, best_match), best_prior = (None, None), None
        for rx, prior in self.items():
            # Alternatives come in descending priority, so once something matched,
            # a lower priority can no longer win.
            if best_match is not None and best_prior > prior:  # type: ignore[misc]
                break
            if (match := rx.pattern.match(  # type: ignore[overloaded-call]
                    input,
                    pos,
                    endpos,
                    **options,
            )) is None:
                continue
            # Every match starts at pos, so the furthest end is the longest. The strict
            # comparison keeps the earlier alternative on equal length.
            if best_match is None or match.end(0) > best_match.end(0):  # type: ignore[union-attr]
                (best_rx, best_match), best_prior = (rx, match), prior

        if best_match is None:
            return None
        return best_rx, best_match  # type: ignore[return-value]


__all__ = ("Alternations",)
