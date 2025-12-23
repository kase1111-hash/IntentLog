"""
Signal Types for MP-02 Protocol

Signals are raw observable traces of effort, including:
- Voice transcripts
- Text edits
- Command history
- Structured tool interaction

Per MP-02 Section 5, signals MUST be time-stamped and preserve ordering.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import hashlib
import json
import uuid


class SignalType(Enum):
    """Types of signals that can be captured"""
    TEXT_EDIT = "text_edit"           # File edits, document changes
    COMMAND = "command"               # Terminal/CLI commands
    VOICE = "voice"                   # Voice transcripts
    TOOL_INTERACTION = "tool"         # Structured tool use
    ANNOTATION = "annotation"         # Human markers/notes
    KEYSTROKE = "keystroke"           # Individual keystrokes
    FILE_CHANGE = "file_change"       # File system changes
    BROWSER = "browser"               # Web browsing activity
    IDE = "ide"                       # IDE interactions
    CUSTOM = "custom"                 # User-defined signal types


@dataclass
class SignalSource:
    """
    Source metadata for a signal.

    Observers MUST disclose capture modality (Section 5).
    """
    observer_id: str                  # Unique observer identifier
    observer_type: str                # Type of observer (e.g., "text", "command")
    capture_modality: str             # How signal was captured
    location: str = ""                # Where captured (file path, terminal, etc.)
    application: str = ""             # Application context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observer_id": self.observer_id,
            "observer_type": self.observer_type,
            "capture_modality": self.capture_modality,
            "location": self.location,
            "application": self.application,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalSource":
        return cls(
            observer_id=data.get("observer_id", ""),
            observer_type=data.get("observer_type", ""),
            capture_modality=data.get("capture_modality", ""),
            location=data.get("location", ""),
            application=data.get("application", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Signal:
    """
    A raw observable trace of effort.

    Per MP-02 Section 5:
    - Observers MUST time-stamp all signals
    - Observers MUST preserve ordering
    - Observers MUST NOT alter raw signals
    - Observers MUST NOT infer intent beyond observed data
    """
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType = SignalType.CUSTOM
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""                 # Raw signal content
    source: Optional[SignalSource] = None
    sequence_number: int = 0          # Ordering within session
    duration_ms: Optional[int] = None # Duration if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hash of raw content for integrity verification
    _content_hash: str = field(default="", repr=False)

    def __post_init__(self):
        """Compute content hash after initialization"""
        if not self._content_hash and self.content:
            self._content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of raw content"""
        content_bytes = self.content.encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()[:16]

    @property
    def content_hash(self) -> str:
        """Get the content hash, computing if needed"""
        if not self._content_hash and self.content:
            self._content_hash = self._compute_hash()
        return self._content_hash

    def verify_integrity(self) -> bool:
        """Verify that content hasn't been altered"""
        if not self._content_hash:
            return True  # No hash stored
        return self._compute_hash() == self._content_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "source": self.source.to_dict() if self.source else None,
            "sequence_number": self.sequence_number,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Deserialize from dictionary"""
        signal = cls(
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            signal_type=SignalType(data.get("signal_type", "custom")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            content=data.get("content", ""),
            source=SignalSource.from_dict(data["source"]) if data.get("source") else None,
            sequence_number=data.get("sequence_number", 0),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata", {}),
        )
        signal._content_hash = data.get("content_hash", "")
        return signal

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Signal":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))
