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
# optional fraction ('.' or ','), then an optional zone ('Z' or an offset).
# The fields are ASCII digits, as Python 3.11+ requires. A fraction is 1-5
# ASCII digits, or 6 ASCII digits followed by further digits, which 3.11+
# truncates away (it accepts non-ASCII ones there only before a zone).
_EXTENDED = re.compile(
    r"(?P<base>[0-9]{4}-[0-9]{2}-[0-9]{2}.[0-9]{2}"
    r"(?::[0-9]{2}(?::[0-9]{2}(?:[.,](?:[0-9]{1,5}|[0-9]{6}(?P<extra>\d*)))?)?)?)"
    r"(?P<zone>Z|[+-][0-9]{2}:[0-9]{2})?"
)
# A fraction only ever follows the seconds field
_FRACTION = re.compile(r"(?<=:[0-9]{2})[.,](\d+)$")

# Forms fromisoformat() accepts before Python 3.11 but 3.11+ rejects or reads
# differently: an empty fraction ('10:00:45.+00:00'), anything after a 'Z',
# and an offset with fractional seconds (whose fraction older versions drop)
_NOT_PYTHON_311 = re.compile(r"[.,](?![0-9])|Z.|[+-][0-9]{2}:[0-9]{2}:[0-9]{2}[.,]")


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
    Parse an ISO 8601 timestamp as datetime.fromisoformat() does on 3.11+.

    On Python 3.11 and later this is fromisoformat() itself. On older
    versions, an extended-format value that fromisoformat() rejects there
    (a 'Z' suffix, a ',' decimal, or a fraction that is not 3 or 6 digits
    long) is rewritten to the equivalent form those versions accept, so it
    is read the same way everywhere. Forms that older versions accept but
    3.11+ rejects or reads differently (an empty fraction, anything after a
    'Z', an offset with fractional seconds) are refused there. So on every
    supported Python a value is either read exactly as 3.11+ reads it or
    refused. Some forms 3.11+ also tolerates still need Python 3.11+:
    basic format (no '-' or ':' separators, including offsets like +0000),
    a fraction on the hour or minute field, and offsets with seconds.

    Raises:
        ValueError: If the value is not a valid timestamp
        TypeError: If the value is not a string
    """
    old_python = sys.version_info < (3, 11)
    if old_python and isinstance(value, str) and _NOT_PYTHON_311.search(value):
        raise ValueError(f"Invalid isoformat string: {value!r}")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if old_python and isinstance(value, str):
            rewritten = _as_python38_isoformat(value)
            if rewritten is not None and rewritten != value:
                return datetime.fromisoformat(rewritten)
        raise
