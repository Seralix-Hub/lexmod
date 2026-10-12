"""
Tokens: the text a rule matched, its type, and exactly where it sits in the source.
"""
import dataclasses
from enum import Enum


@dataclasses.dataclass(slots=True, frozen=True)
class Token[Type: Enum, AnyStr: (str, bytes)]:
    """
    An immutable lexeme: the text one grammar rule matched, its kind, and where it
    sits in the source.

    ``Type`` is the language's token enum and ``AnyStr`` is the source type, so
    ``data`` always has the same type as the source it came from.

    The location is stored twice: as offsets into the source, for slicing, and as
    line and column positions, for diagnostics. Both are half-open, so a
    zero-width token such as EOF, INDENT or DEDENT has equal start and end, and
    each token's ``tail`` is exactly the next token's ``head``.

    The source is normalized before lexing: ``\\a``, ``\\b``, ``\\v``, ``\\f`` and
    ``\\r`` are removed, so lines break at ``\\n`` only, and ``span`` gives offsets
    into the normalized source rather than the original file. Columns count code
    points for ``str`` sources and bytes for ``bytes`` sources. They can therefore
    differ from what an editor displays for tabs or for characters outside the
    Basic Multilingual Plane.

    :ivar type: the kind of token, a member of the language's token enum
    :ivar data: the matched text, as returned by ``match.group(0)``, always equal
        to ``source[span[0]:span[1]]``
    :ivar span: 0-based ``(pos, endpos)`` offsets into the source, half-open, as
        returned by ``match.span(0)``
    :ivar head: 1-based ``(line, col)`` of the first character
    :ivar tail: 1-based ``(line, col)`` of the position just past the last
        character, exclusive, equal to ``head`` for an empty token, and at column 1
        of the next line when ``data`` ends with ``\\n``
    """
    type: Type
    data: AnyStr
    span: tuple[int, int]
    head: tuple[int, int]
    tail: tuple[int, int]

    def __post_init__(self) -> None:
        """
        Validate the invariants that the field types cannot express.

        Checks that ``span`` is non-negative and ordered, that ``data`` is as long
        as ``span`` covers, that ``head`` is a valid 1-based position, and that
        ``tail`` is exactly the position reached by advancing ``head`` over
        ``data``. Whether ``head`` agrees with ``span`` is not checked, since that
        requires the source.

        :raises ValueError: if any invariant does not hold
        """
        pos, endpos = self.span
        if not 0 <= pos <= endpos:
            raise ValueError(f"invalid span {self.span}: expected 0 <= pos <= endpos")
        # The source isn't available here, so matching lengths is the strongest
        # check that data really is source[pos:endpos].
        if len(self.data) != endpos - pos:
            raise ValueError(f"data length {len(self.data)} does not match span {self.span}")
        if self.head[0] < 1 or self.head[1] < 1:
            raise ValueError(f"invalid head {self.head}: lines and cols start at 1")

        # Separate branches so the type checker narrows AnyStr in each one.
        if isinstance(self.data, str):
            nls, idx = self.data.count("\n"), self.data.rfind("\n")
        else:
            nls, idx = self.data.count(b"\n"), self.data.rfind(b"\n")
        # The tail column is where the last line starts (head's column if there is no
        # break, 1 after one; rfind gives -1 for none) plus the characters on that line.
        if self.tail != (tail := (self.head[0] + nls, ((self.head[1] if idx < 0 else 1) + len(self.data) - (idx + 1)))):
            raise ValueError(f"tail {self.tail} does not match head {self.head} and data, expected {tail}")

__all__ = ("Token",)
