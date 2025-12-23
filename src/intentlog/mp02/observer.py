"""
Observer System for MP-02 Protocol

Per MP-02 Section 5, Observers:
- MAY record continuous or intermittent signals
- MAY record multi-modal inputs
- MUST time-stamp all signals
- MUST preserve ordering
- MUST disclose capture modality
- MUST NOT alter raw signals
- MUST NOT infer intent beyond observed data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import hashlib
import os
import threading
import time
import uuid

from .signal import Signal, SignalType, SignalSource


@dataclass
class ObserverConfig:
    """Configuration for observers"""
    observer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    buffer_size: int = 1000           # Max signals to buffer
    auto_flush_interval: float = 30.0 # Seconds between auto-flushes
    capture_modality: str = ""        # How signals are captured
    metadata: Dict[str, Any] = field(default_factory=dict)


class Observer(ABC):
    """
    Abstract base class for effort observers.

    Observers capture raw signals from human activity without
    altering or interpreting them beyond basic categorization.
    """

    def __init__(self, config: Optional[ObserverConfig] = None):
        self.config = config or ObserverConfig()
        self._signals: List[Signal] = []
        self._sequence_counter: int = 0
        self._is_running: bool = False
        self._lock = threading.Lock()
        self._flush_callbacks: List[Callable[[List[Signal]], None]] = []
        self._start_time: Optional[datetime] = None

    @property
    @abstractmethod
    def observer_type(self) -> str:
        """Return the type of this observer"""
        pass

    @abstractmethod
    def _capture(self) -> Optional[Signal]:
        """
        Capture a single signal.

        Returns None if no signal is available.
        Implementations MUST NOT alter raw content.
        """
        pass

    def start(self) -> None:
        """Start observing"""
        if self._is_running:
            return
        self._is_running = True
        self._start_time = datetime.now()
        self._on_start()

    def stop(self) -> None:
        """Stop observing"""
        if not self._is_running:
            return
        self._is_running = False
        self._on_stop()
        self.flush()

    def _on_start(self) -> None:
        """Hook called when observer starts"""
        pass

    def _on_stop(self) -> None:
        """Hook called when observer stops"""
        pass

    @property
    def is_running(self) -> bool:
        """Check if observer is running"""
        return self._is_running

    def _create_source(self, location: str = "", application: str = "") -> SignalSource:
        """Create a SignalSource for captured signals"""
        return SignalSource(
            observer_id=self.config.observer_id,
            observer_type=self.observer_type,
            capture_modality=self.config.capture_modality or self.observer_type,
            location=location,
            application=application,
            metadata=self.config.metadata.copy(),
        )

    def capture(self) -> Optional[Signal]:
        """
        Capture a signal and add it to the buffer.

        Returns the captured signal or None if no signal available.
        """
        if not self._is_running:
            return None

        signal = self._capture()
        if signal is None:
            return None

        with self._lock:
            self._sequence_counter += 1
            signal.sequence_number = self._sequence_counter
            self._signals.append(signal)

            # Auto-flush if buffer is full
            if len(self._signals) >= self.config.buffer_size:
                self._do_flush()

        return signal

    def get_signals(self) -> List[Signal]:
        """Get all buffered signals without clearing"""
        with self._lock:
            return self._signals.copy()

    def flush(self) -> List[Signal]:
        """Flush and return all buffered signals"""
        with self._lock:
            return self._do_flush()

    def _do_flush(self) -> List[Signal]:
        """Internal flush (caller must hold lock)"""
        signals = self._signals.copy()
        self._signals.clear()

        # Notify callbacks
        for callback in self._flush_callbacks:
            try:
                callback(signals)
            except Exception:
                pass  # Don't let callback errors affect observer

        return signals

    def on_flush(self, callback: Callable[[List[Signal]], None]) -> None:
        """Register a callback to be called on flush"""
        self._flush_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get observer statistics"""
        with self._lock:
            return {
                "observer_id": self.config.observer_id,
                "observer_type": self.observer_type,
                "is_running": self._is_running,
                "signals_buffered": len(self._signals),
                "signals_captured": self._sequence_counter,
                "start_time": self._start_time.isoformat() if self._start_time else None,
            }


class TextObserver(Observer):
    """
    Observer for text file changes.

    Watches specified files or directories for text changes
    and captures them as signals.
    """

    def __init__(
        self,
        paths: Optional[List[str]] = None,
        config: Optional[ObserverConfig] = None,
    ):
        super().__init__(config)
        self.paths = paths or []
        self._file_states: Dict[str, str] = {}  # path -> content hash
        self._poll_interval = 1.0  # seconds
        self._poll_thread: Optional[threading.Thread] = None

    @property
    def observer_type(self) -> str:
        return "text"

    def add_path(self, path: str) -> None:
        """Add a path to watch"""
        if path not in self.paths:
            self.paths.append(path)
            # Initialize state for existing files
            p = Path(path)
            if p.exists() and p.is_file():
                self._file_states[path] = self._hash_file(path)

    def _hash_file(self, path: str) -> str:
        """Compute hash of file contents"""
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""

    def _read_file(self, path: str) -> str:
        """Read file contents safely"""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def _capture(self) -> Optional[Signal]:
        """Check for file changes and capture as signal"""
        for path in self.paths:
            p = Path(path)
            if not p.exists():
                continue

            if p.is_dir():
                # Watch all files in directory
                for file_path in p.rglob("*"):
                    if file_path.is_file():
                        signal = self._check_file(str(file_path))
                        if signal:
                            return signal
            else:
                signal = self._check_file(path)
                if signal:
                    return signal

        return None

    def _check_file(self, path: str) -> Optional[Signal]:
        """Check a single file for changes"""
        current_hash = self._hash_file(path)
        previous_hash = self._file_states.get(path, "")

        if current_hash and current_hash != previous_hash:
            self._file_states[path] = current_hash
            content = self._read_file(path)

            return Signal(
                signal_type=SignalType.TEXT_EDIT,
                timestamp=datetime.now(),
                content=content,
                source=self._create_source(
                    location=path,
                    application="file_watcher",
                ),
                metadata={
                    "file_path": path,
                    "content_hash": current_hash,
                    "previous_hash": previous_hash,
                    "file_size": len(content),
                },
            )

        return None

    def _on_start(self) -> None:
        """Start polling thread"""
        # Initialize file states
        for path in self.paths:
            p = Path(path)
            if p.exists() and p.is_file():
                self._file_states[path] = self._hash_file(path)

        # Start background polling
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _on_stop(self) -> None:
        """Stop polling thread"""
        # Thread will exit when is_running becomes False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        """Background polling loop"""
        while self._is_running:
            self.capture()
            time.sleep(self._poll_interval)


class CommandObserver(Observer):
    """
    Observer for command-line activity.

    Captures terminal commands from shell history or explicit logging.
    """

    def __init__(
        self,
        history_file: Optional[str] = None,
        config: Optional[ObserverConfig] = None,
    ):
        super().__init__(config)
        self.history_file = history_file or os.path.expanduser("~/.bash_history")
        self._last_history_line: int = 0
        self._pending_commands: List[str] = []

    @property
    def observer_type(self) -> str:
        return "command"

    def log_command(self, command: str, working_dir: str = "", exit_code: int = 0) -> Signal:
        """
        Explicitly log a command execution.

        This allows programmatic command logging without relying on history files.
        """
        signal = Signal(
            signal_type=SignalType.COMMAND,
            timestamp=datetime.now(),
            content=command,
            source=self._create_source(
                location=working_dir or os.getcwd(),
                application="shell",
            ),
            metadata={
                "exit_code": exit_code,
                "working_dir": working_dir or os.getcwd(),
            },
        )

        with self._lock:
            self._sequence_counter += 1
            signal.sequence_number = self._sequence_counter
            self._signals.append(signal)

        return signal

    def _read_history(self) -> List[str]:
        """Read new lines from shell history"""
        try:
            with open(self.history_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            new_lines = lines[self._last_history_line:]
            self._last_history_line = len(lines)

            return [line.strip() for line in new_lines if line.strip()]
        except Exception:
            return []

    def _capture(self) -> Optional[Signal]:
        """Check for new commands in history"""
        if self._pending_commands:
            command = self._pending_commands.pop(0)
            return Signal(
                signal_type=SignalType.COMMAND,
                timestamp=datetime.now(),
                content=command,
                source=self._create_source(
                    location=self.history_file,
                    application="shell_history",
                ),
                metadata={
                    "source": "history_file",
                    "history_file": self.history_file,
                },
            )

        # Check for new history entries
        new_commands = self._read_history()
        if new_commands:
            self._pending_commands.extend(new_commands[1:])  # Queue remaining
            command = new_commands[0]
            return Signal(
                signal_type=SignalType.COMMAND,
                timestamp=datetime.now(),
                content=command,
                source=self._create_source(
                    location=self.history_file,
                    application="shell_history",
                ),
                metadata={
                    "source": "history_file",
                    "history_file": self.history_file,
                },
            )

        return None


class AnnotationObserver(Observer):
    """
    Observer for explicit human annotations.

    Allows users to add markers and notes during work sessions.
    This implements "explicit human markers" per Section 6.
    """

    def __init__(self, config: Optional[ObserverConfig] = None):
        super().__init__(config)
        self._pending_annotations: List[Dict[str, Any]] = []

    @property
    def observer_type(self) -> str:
        return "annotation"

    def annotate(
        self,
        text: str,
        category: str = "note",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Add a human annotation/marker.

        Categories:
        - "note": General note
        - "milestone": Significant progress point
        - "question": Open question or uncertainty
        - "decision": Decision made
        - "todo": Task to do later
        - "blocker": Something blocking progress
        """
        signal = Signal(
            signal_type=SignalType.ANNOTATION,
            timestamp=datetime.now(),
            content=text,
            source=self._create_source(
                application="annotation",
            ),
            metadata={
                "category": category,
                **(metadata or {}),
            },
        )

        with self._lock:
            self._sequence_counter += 1
            signal.sequence_number = self._sequence_counter
            self._signals.append(signal)

        return signal

    def _capture(self) -> Optional[Signal]:
        """Check for pending annotations"""
        if self._pending_annotations:
            annotation = self._pending_annotations.pop(0)
            return Signal(
                signal_type=SignalType.ANNOTATION,
                timestamp=datetime.now(),
                content=annotation.get("text", ""),
                source=self._create_source(application="annotation"),
                metadata=annotation.get("metadata", {}),
            )
        return None
