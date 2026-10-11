"""
Timestamp parsing shared by storage and the hash chain.

Both must read a stored timestamp identically on every supported Python,
because the parsed value feeds the content hash.
"""

import re
import sys
from datetime import datetime
from typing import Optional

# Extended-format ISO 8601 date-time: date, any separator, time with an
# optional fraction ('.' or ','), then an optional zone ('Z' or an offset,
# which may include seconds).
# The fields are ASCII digits, as Python 3.11+ requires. A fraction is 1-5
# ASCII digits, or 6 ASCII digits followed by further digits, which 3.11+
# truncates away (it accepts non-ASCII ones there only before a zone).
_EXTENDED = re.compile(
    r"(?P<base>[0-9]{4}-[0-9]{2}-[0-9]{2}.[0-9]{2}"
    r"(?::[0-9]{2}(?::[0-9]{2}(?:[.,](?:[0-9]{1,5}|[0-9]{6}(?P<extra>\d*)))?)?)?)"
    r"(?P<zone>Z|[+-][0-9]{2}:[0-9]{2}(?::[0-9]{2})?)?"
)
# A fraction only ever follows the seconds field
_FRACTION = re.compile(r"(?<=:[0-9]{2})[.,](\d+)$")

# Forms that supported Python versions disagree on, refused everywhere: an
# empty fraction ('10:00:45.+00:00', rejected by 3.13 only) and anything
# after a 'Z' (accepted before 3.11)
_INCONSISTENT = re.compile(r"[.,](?![0-9])|Z.", re.DOTALL)

# What may follow the sign of a UTC offset: hours, then optional minutes and
# seconds, each optionally after a ':'
_OFFSET_FIELDS = re.compile(r"[0-9]{2}(?::?[0-9]{2}){0,2}")
_SIGN = re.compile(r"[+-]")


def _offset_has_extra_part(value: str) -> bool:
    """
    Whether a UTC offset in the value is followed by anything, such as a
    fraction ('+00:00.5', '+00:00:00:5', '+000000123456').

    Versions read such a value differently: 3.11 and 3.12 drop the fraction
    of a zero offset, 3.8-3.12 drop one written after ':', and 3.13 keeps
    both.

    The offset starts at the first sign in the time, as fromisoformat()
    looks for it. Signs before index 11 (extended dates) or 9 (basic
    dates) are the date's own separators or the date-time separator; a
    time with an offset sign there would have fewer than two hour digits.
    """
    start = 11 if value[4:5] == "-" else 9
    sign = _SIGN.search(value, start)
    return sign is not None and not _OFFSET_FIELDS.fullmatch(value, sign.end())


def _as_python38_isoformat(value: str) -> Optional[str]:
    """
    Rewrite an extended-format value into a form fromisoformat() reads on
    Python 3.8-3.10, with the meaning Python 3.11+ gives the original: a
    ',' decimal becomes '.', the fraction is padded or truncated to six
    digits (3.11+ truncates extra digits), and 'Z' becomes '+00:00'.

    Returns None if the value is not extended-format ISO 8601.
    """
    match = _EXTENDED.fullmatch(value)
    if not match:
        return None
    extra = match.group("extra") or ""
    if not match.group("zone") and not extra.isascii():
        return None
    base = match.group("base")
    fraction = _FRACTION.search(base)
    if fraction:
        base = base[:fraction.start()] + "." + (fraction.group(1) + "000000")[:6]
    zone = match.group("zone") or ""
    if zone == "Z":
        zone = "+00:00"
    return base + zone


def parse_iso_datetime(value: str) -> datetime:
    """
    Parse an ISO 8601 timestamp the same way on every supported Python.

    This is datetime.fromisoformat(), with two adjustments so that a stored
    timestamp (which feeds the content hash) never reads differently on
    different interpreters:

    - Forms the supported versions disagree on are refused everywhere: an
      empty fraction, anything after a 'Z', and anything after a UTC
      offset's hours, minutes and seconds, such as a fraction.
    - Before Python 3.11, an extended-format value fromisoformat() rejects
      there (a 'Z' suffix, a ',' decimal, or a fraction that is not 3 or 6
      digits long) is rewritten to the equivalent form it accepts.

    Some forms only Python 3.11+ accepts are still refused on older
    versions: basic format (no '-' or ':' separators, including offsets like
    +0000) and a fraction on the hour or minute field.

    Raises:
        ValueError: If the value is not a valid timestamp
        TypeError: If the value is not a string
    """
    if isinstance(value, str) and (_INCONSISTENT.search(value) or _offset_has_extra_part(value)):
        raise ValueError(f"Invalid isoformat string: {value!r}")
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        rewritten = None
        if sys.version_info < (3, 11) and isinstance(value, str):
            rewritten = _as_python38_isoformat(value)
        if rewritten is None or rewritten == value:
            raise
        result = datetime.fromisoformat(rewritten)
    # Backstop: an offset with a sub-second part is never read the same way
    # everywhere, whatever form produced it
    offset = result.utcoffset()
    if offset is not None and offset.microseconds:
        raise ValueError(f"Invalid isoformat string: {value!r}")
    return result
