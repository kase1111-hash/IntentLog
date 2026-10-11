"""
Timestamp parsing shared by storage and the hash chain.

Both must read a stored timestamp identically on every supported Python,
because the parsed value feeds the content hash.
"""

import re
from datetime import datetime

# A date-and-time value whose only zone designator is a trailing 'Z'
_Z_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}.\d{2}(?::\d{2}(?::\d{2}(?:\.\d+)?)?)?Z")


def parse_iso_datetime(value: str) -> datetime:
    """
    Parse an ISO 8601 timestamp as datetime.fromisoformat() does on 3.11+.

    Before Python 3.11, fromisoformat() rejects a 'Z' UTC suffix. A
    date-and-time value ending in 'Z' is then retried with '+00:00', so it
    is read the same way on every supported Python. Nothing that 3.11+
    rejects is accepted.

    Raises:
        ValueError: If the value is not a valid timestamp
        TypeError: If the value is not a string
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if isinstance(value, str) and _Z_DATETIME.fullmatch(value):
            return datetime.fromisoformat(value[:-1] + "+00:00")
        raise
