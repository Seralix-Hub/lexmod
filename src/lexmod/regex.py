"""
Regular expressions with normalized flags.

A :class:`Regex` keeps only the flags that change behavior, so equivalent
patterns compare and hash equal, and share one compiled pattern. Patterns use
the third-party :mod:`regex` module, which extends :mod:`re` with variable-length
lookbehind and more. :func:`coerce` turns any accepted pattern form into a
:class:`Regex`.
"""
import dataclasses
import functools
from typing import Final, TYPE_CHECKING

import regex
from regex import RegexFlag, Pattern

if TYPE_CHECKING:
    # Only for the __rich_repr__ annotation, so rich stays optional: it calls that
    # method when it's installed, and nothing else needs it.
    from rich.repr import RichReprResult

type AnyPattern[AnyStr: (str, bytes)] = AnyStr | Pattern[AnyStr]
type AnyRegex[AnyStr: (str, bytes)] = Regex[AnyStr] | AnyPattern[AnyStr] | tuple[AnyPattern[AnyStr], int]

_REGEX_ENCODINGS: Final[RegexFlag] = RegexFlag.UNICODE | RegexFlag.ASCII | RegexFlag.LOCALE
_REGEX_VERSIONS: Final[RegexFlag] = RegexFlag.V0 | RegexFlag.V1


@functools.cache
def _compile_pattern[AnyStr: (str, bytes)](pattern: AnyStr, /, flags: int) -> Pattern[AnyStr]:
    """
    Compile a pattern once per distinct source and flags.

    Cached, so equal :class:`Regex` instances share one compiled pattern instead of
    each compiling their own.

    :param pattern: The pattern source
    :param flags: The flags to compile with
    :return: The compiled pattern
    :raises regex.error: If the pattern does not compile
    :raises ValueError: If the flags are incompatible with each other or the pattern
    """
    return regex.compile(pattern, flags)


def _implicit_flags[AnyStr: (str, bytes)](pattern: AnyStr, /, flags: int) -> RegexFlag:
    """
    Remove the flags the engine applies on its own, keeping only those that change behavior.

    :param pattern: The pattern source, which decides the encoding default
    :param flags: The flags to reduce
    :return: ``flags`` without the default encoding, the default version, or
        ``FULLCASE`` when ``VERSION1`` already implies it
    """
    # Default encoding: UNICODE for str sources, ASCII for bytes sources.
    flags &= ~(RegexFlag.UNICODE if isinstance(pattern, str) else RegexFlag.ASCII)
    # VERSION1 turns FULLCASE on by itself.
    if flags & RegexFlag.VERSION1:
        flags &= ~RegexFlag.FULLCASE
    if (flags & _REGEX_VERSIONS) == regex.DEFAULT_VERSION:
        flags &= ~_REGEX_VERSIONS
    return RegexFlag(flags)


def _explicit_flags[AnyStr: (str, bytes)](pattern: AnyStr, /, flags: int) -> RegexFlag:
    """
    Add back the flags the engine applies on its own, giving the full set it reports.

    Inline flags in the pattern are invisible here, so the result is only exact
    for patterns without an inline encoding or version, such as ``(?a)`` or
    ``(?V1)``. Never compile with it, since those would then conflict.

    :param pattern: The pattern source, which decides the encoding default
    :param flags: The flags to complete
    :return: ``flags`` with the default encoding and version added when none is
        set, and ``FULLCASE`` added when ``VERSION1`` is set
    """
    # ASCII, LOCALE and UNICODE exclude each other, so only add the default when none is set.
    if not flags & _REGEX_ENCODINGS:
        flags |= RegexFlag.UNICODE if isinstance(pattern, str) else RegexFlag.ASCII
    if not flags & _REGEX_VERSIONS:
        flags |= regex.DEFAULT_VERSION
    if flags & RegexFlag.VERSION1:
        flags |= RegexFlag.FULLCASE
    return RegexFlag(flags)


@dataclasses.dataclass(init=False, slots=True, frozen=True)
class Regex[AnyStr: (str, bytes)]:
    """
    An immutable regular expression whose flags are normalized, so it compares by meaning.

    Only the flags that change behavior are stored: engine defaults such as the
    encoding and version are left out. Building from a source or from a compiled
    pattern, with or without redundant flags, therefore gives equal, equally
    hashing instances. Inline flags such as ``(?i)`` stay part of the source.

    :ivar regex: The pattern source, ``str`` or ``bytes``
    :ivar flags: The flags that change behavior beyond the engine's defaults
    :ivar pattern: The compiled pattern, shared between equal instances and left
        out of comparison, hashing and the repr
    """
    regex: AnyStr
    flags: RegexFlag
    pattern: Pattern[AnyStr] = dataclasses.field(repr=False, hash=False, compare=False)

    def __init__(self, regex: AnyPattern[AnyStr], /, flags: int = 0) -> None:
        """
        Build from a pattern source or from a compiled pattern.

        A compiled pattern contributes its source and its flags. The given flags
        are added to those, and where they choose an encoding (``ASCII``,
        ``LOCALE``, ``UNICODE``) or a version (``VERSION0``, ``VERSION1``), they
        replace the compiled pattern's choice.

        :param regex: The pattern source, or a compiled pattern to take the source
            and flags from
        :param flags: Extra flags, combined with a compiled pattern's own
        :raises regex.error: If the pattern does not compile
        :raises ValueError: If the flags are incompatible, such as two encodings
        """
        if isinstance(regex, Pattern):
            inherited: RegexFlag = _implicit_flags(regex.pattern, regex.flags)
            # ASCII/LOCALE/UNICODE and VERSION0/VERSION1 exclude each other, so a group
            # set in the given flags replaces the compiled pattern's choice.
            if flags & _REGEX_ENCODINGS:
                inherited &= ~_REGEX_ENCODINGS
            if flags & _REGEX_VERSIONS:
                inherited &= ~_REGEX_VERSIONS
            flags |= inherited
            regex: AnyStr = regex.pattern
        flags: RegexFlag = _implicit_flags(regex, flags)
        object.__setattr__(self, "regex", regex)
        object.__setattr__(self, "flags", flags)
        # Compile with the reduced flags: the engine re-adds its defaults itself, and
        # adding them here would conflict with inline flags such as (?a) or (?V1).
        object.__setattr__(self, "pattern", _compile_pattern(regex, flags))

    def __str__(self) -> str:
        """
        Render the pattern as a literal prefixed with its flags, like ``im'a+b'``.

        A non-default version comes first, followed by a dash, as in ``v1-i'a+b'``.

        :return: The lowercase flag letters followed by the pattern's ``repr``
        """
        version: RegexFlag = self.flags & _REGEX_VERSIONS
        # A zero flag has no name, so the prefix only exists when a version is set.
        txt = f"{(version.name or "").lower()}-" if version else ""
        for flag in self.flags & ~_REGEX_VERSIONS:
            txt += (flag.name or "").lower()
        return txt + repr(self.regex)

    def __repr__(self) -> str:
        """
        Render the constructor call that rebuilds an equal instance.

        :return: For example ``Regex('a+b')`` or
            ``Regex('a+b', flags=RegexFlag.I|RegexFlag.M)``
        """
        if self.flags:
            # flag.name, since formatting the member itself would give "regex.I".
            return f"{type(self).__qualname__}({self.regex!r}, flags={
                "|".join(f"RegexFlag.{flag.name}" for flag in self.flags)
            })"
        return f"{type(self).__qualname__}({self.regex!r})"

    def __rich_repr__(self) -> "RichReprResult":
        """
        Describe the same constructor call as ``__repr__``, so rich can colorize it.

        :return: The source as a positional argument, then ``flags``, which rich
            leaves out when it equals the default of ``0``
        """
        yield self.regex
        yield "flags", self.flags, 0


def coerce[AnyStr: (str, bytes)](regex: AnyRegex[AnyStr], /) -> Regex[AnyStr]:
    """
    Convert any accepted pattern form into a :class:`Regex`.

    Lets an API take whichever form is most convenient for the caller: a
    :class:`Regex`, returned unchanged; a ``str`` or ``bytes`` pattern source; a
    compiled :class:`Pattern`; or a ``(pattern, flags)`` pair whose pattern is a
    source or a compiled :class:`Pattern`.

    :param regex: The pattern, in any of the accepted forms
    :return: The equivalent :class:`Regex`, the same object when one was given
    :raises TypeError: If ``regex`` is none of the accepted forms
    :raises regex.error: If the pattern does not compile
    :raises ValueError: If the flags are incompatible, such as two encodings
    """
    match regex:
        case Regex():
            return regex
        case str() | bytes() | Pattern() as regex:
            return Regex(regex)  # type: ignore[arg-type]
        case str() | bytes() | Pattern() as regex, int(flags):
            return Regex(regex, flags)  # type: ignore[arg-type]
        case _:
            given: str = type(regex).__qualname__
            if isinstance(regex, (list, tuple)):
                given += f"[{', '.join(type(item).__qualname__ for item in regex)}]"
            raise TypeError(
                f"expected a Regex, a str or bytes pattern, a Pattern, or a (pattern, flags) pair, got {given}"
            )


__all__ = ("AnyPattern", "AnyRegex", "Regex")
