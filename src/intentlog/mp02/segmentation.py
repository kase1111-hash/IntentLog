"""
Segmentation Engine for MP-02 Protocol

Per MP-02 Section 6, signals are grouped into Effort Segments based on:
- Time windows
- Activity boundaries
- Explicit human markers

Segmentation rules MUST be deterministic and disclosed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
import hashlib
import json
import uuid

from .signal import Signal, SignalType


class SegmentationMethod(Enum):
    """Methods for segmenting signals into effort segments"""
    TIME_WINDOW = "time_window"       # Fixed time intervals
    ACTIVITY_BOUNDARY = "activity"    # Gaps in activity
    EXPLICIT_MARKER = "marker"        # Human-defined boundaries
    HYBRID = "hybrid"                 # Combination of methods


@dataclass
class SegmentationRule:
    """
    A rule for how to segment signals.

    Rules MUST be deterministic and disclosed per Section 6.
    """
    method: SegmentationMethod = SegmentationMethod.TIME_WINDOW
    time_window_minutes: int = 30     # For TIME_WINDOW method
    gap_threshold_minutes: int = 5    # For ACTIVITY_BOUNDARY method
    marker_categories: List[str] = field(default_factory=lambda: ["milestone", "segment"])
    min_signals: int = 1              # Minimum signals for valid segment
    max_signals: int = 10000          # Maximum signals per segment
    description: str = ""             # Human-readable rule description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "time_window_minutes": self.time_window_minutes,
            "gap_threshold_minutes": self.gap_threshold_minutes,
            "marker_categories": self.marker_categories,
            "min_signals": self.min_signals,
            "max_signals": self.max_signals,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationRule":
        return cls(
            method=SegmentationMethod(data.get("method", "time_window")),
            time_window_minutes=data.get("time_window_minutes", 30),
            gap_threshold_minutes=data.get("gap_threshold_minutes", 5),
            marker_categories=data.get("marker_categories", ["milestone", "segment"]),
            min_signals=data.get("min_signals", 1),
            max_signals=data.get("max_signals", 10000),
            description=data.get("description", ""),
        )

    def get_disclosure(self) -> str:
        """Generate human-readable disclosure of segmentation method"""
        if self.method == SegmentationMethod.TIME_WINDOW:
            return f"Signals grouped into {self.time_window_minutes}-minute time windows"
        elif self.method == SegmentationMethod.ACTIVITY_BOUNDARY:
            return f"Signals grouped by activity, with {self.gap_threshold_minutes}-minute gaps as boundaries"
        elif self.method == SegmentationMethod.EXPLICIT_MARKER:
            return f"Signals grouped by explicit markers: {', '.join(self.marker_categories)}"
        else:
            return "Hybrid segmentation combining time windows and activity boundaries"


@dataclass
class EffortSegment:
    """
    A bounded time slice of signals treated as a unit of analysis.

    Per MP-02 Section 6, this is the fundamental unit for validation and receipts.
    """
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signals: List[Signal] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    rule: Optional[SegmentationRule] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed fields
    _signal_hashes: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Compute time bounds from signals if not set"""
        if self.signals:
            if not self.start_time:
                self.start_time = min(s.timestamp for s in self.signals)
            if not self.end_time:
                self.end_time = max(s.timestamp for s in self.signals)
            self._signal_hashes = [s.content_hash for s in self.signals]

    @property
    def duration(self) -> Optional[timedelta]:
        """Get segment duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def signal_count(self) -> int:
        """Get number of signals in segment"""
        return len(self.signals)

    @property
    def signal_hashes(self) -> List[str]:
        """Get hashes of all signals"""
        if not self._signal_hashes and self.signals:
            self._signal_hashes = [s.content_hash for s in self.signals]
        return self._signal_hashes

    def compute_hash(self) -> str:
        """Compute deterministic hash of segment"""
        data = {
            "segment_id": self.segment_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "signal_hashes": sorted(self.signal_hashes),
            "rule": self.rule.to_dict() if self.rule else None,
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get_summary(self) -> str:
        """Generate a summary of the segment"""
        duration_str = ""
        if self.duration:
            minutes = int(self.duration.total_seconds() / 60)
            duration_str = f"{minutes} minutes"

        signal_types = {}
        for s in self.signals:
            t = s.signal_type.value
            signal_types[t] = signal_types.get(t, 0) + 1

        types_str = ", ".join(f"{v} {k}" for k, v in signal_types.items())

        return f"Segment with {self.signal_count} signals ({types_str}) over {duration_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "signal_count": self.signal_count,
            "signal_hashes": self.signal_hashes,
            "rule": self.rule.to_dict() if self.rule else None,
            "metadata": self.metadata,
            "hash": self.compute_hash(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], signals: Optional[List[Signal]] = None) -> "EffortSegment":
        return cls(
            segment_id=data.get("segment_id", str(uuid.uuid4())),
            signals=signals or [],
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            rule=SegmentationRule.from_dict(data["rule"]) if data.get("rule") else None,
            metadata=data.get("metadata", {}),
        )


class SegmentationEngine:
    """
    Engine for grouping signals into effort segments.

    Supports multiple segmentation strategies as defined in Section 6.
    """

    def __init__(self, rule: Optional[SegmentationRule] = None):
        self.rule = rule or SegmentationRule()
        self._current_segment_signals: List[Signal] = []
        self._segments: List[EffortSegment] = []

    def segment(self, signals: List[Signal]) -> List[EffortSegment]:
        """
        Segment a list of signals according to the configured rule.

        This is the main entry point for batch segmentation.
        """
        if not signals:
            return []

        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)

        if self.rule.method == SegmentationMethod.TIME_WINDOW:
            return self._segment_by_time_window(sorted_signals)
        elif self.rule.method == SegmentationMethod.ACTIVITY_BOUNDARY:
            return self._segment_by_activity(sorted_signals)
        elif self.rule.method == SegmentationMethod.EXPLICIT_MARKER:
            return self._segment_by_markers(sorted_signals)
        else:
            return self._segment_hybrid(sorted_signals)

    def _segment_by_time_window(self, signals: List[Signal]) -> List[EffortSegment]:
        """Segment by fixed time windows"""
        if not signals:
            return []

        window = timedelta(minutes=self.rule.time_window_minutes)
        segments: List[EffortSegment] = []
        current_signals: List[Signal] = []
        window_start = signals[0].timestamp

        for signal in signals:
            if signal.timestamp >= window_start + window:
                # Close current window
                if current_signals and len(current_signals) >= self.rule.min_signals:
                    segments.append(EffortSegment(
                        signals=current_signals.copy(),
                        rule=self.rule,
                    ))
                current_signals = [signal]
                window_start = signal.timestamp
            else:
                current_signals.append(signal)

                # Check max signals
                if len(current_signals) >= self.rule.max_signals:
                    segments.append(EffortSegment(
                        signals=current_signals.copy(),
                        rule=self.rule,
                    ))
                    current_signals = []
                    window_start = signal.timestamp + timedelta(milliseconds=1)

        # Close final segment
        if current_signals and len(current_signals) >= self.rule.min_signals:
            segments.append(EffortSegment(
                signals=current_signals,
                rule=self.rule,
            ))

        return segments

    def _segment_by_activity(self, signals: List[Signal]) -> List[EffortSegment]:
        """Segment by activity boundaries (gaps in activity)"""
        if not signals:
            return []

        gap_threshold = timedelta(minutes=self.rule.gap_threshold_minutes)
        segments: List[EffortSegment] = []
        current_signals: List[Signal] = [signals[0]]

        for i in range(1, len(signals)):
            gap = signals[i].timestamp - signals[i-1].timestamp

            if gap > gap_threshold:
                # Gap detected, close current segment
                if len(current_signals) >= self.rule.min_signals:
                    segments.append(EffortSegment(
                        signals=current_signals.copy(),
                        rule=self.rule,
                    ))
                current_signals = [signals[i]]
            else:
                current_signals.append(signals[i])

                # Check max signals
                if len(current_signals) >= self.rule.max_signals:
                    segments.append(EffortSegment(
                        signals=current_signals.copy(),
                        rule=self.rule,
                    ))
                    current_signals = []

        # Close final segment
        if current_signals and len(current_signals) >= self.rule.min_signals:
            segments.append(EffortSegment(
                signals=current_signals,
                rule=self.rule,
            ))

        return segments

    def _segment_by_markers(self, signals: List[Signal]) -> List[EffortSegment]:
        """Segment by explicit human markers"""
        if not signals:
            return []

        segments: List[EffortSegment] = []
        current_signals: List[Signal] = []

        for signal in signals:
            current_signals.append(signal)

            # Check if this is a segment marker
            if signal.signal_type == SignalType.ANNOTATION:
                category = signal.metadata.get("category", "")
                if category in self.rule.marker_categories:
                    # Close segment at marker
                    if len(current_signals) >= self.rule.min_signals:
                        segments.append(EffortSegment(
                            signals=current_signals.copy(),
                            rule=self.rule,
                            metadata={"marker": category},
                        ))
                    current_signals = []

        # Close final segment
        if current_signals and len(current_signals) >= self.rule.min_signals:
            segments.append(EffortSegment(
                signals=current_signals,
                rule=self.rule,
            ))

        return segments

    def _segment_hybrid(self, signals: List[Signal]) -> List[EffortSegment]:
        """Hybrid segmentation: markers take priority, then activity boundaries"""
        if not signals:
            return []

        # First, split by explicit markers
        marker_groups: List[List[Signal]] = []
        current_group: List[Signal] = []

        for signal in signals:
            current_group.append(signal)

            if signal.signal_type == SignalType.ANNOTATION:
                category = signal.metadata.get("category", "")
                if category in self.rule.marker_categories:
                    marker_groups.append(current_group.copy())
                    current_group = []

        if current_group:
            marker_groups.append(current_group)

        # Then apply activity boundary segmentation to each group
        segments: List[EffortSegment] = []
        for group in marker_groups:
            if not group:
                continue
            # Apply activity-based segmentation within the group
            gap_threshold = timedelta(minutes=self.rule.gap_threshold_minutes)
            current_signals: List[Signal] = [group[0]]

            for i in range(1, len(group)):
                gap = group[i].timestamp - group[i-1].timestamp
                if gap > gap_threshold:
                    if len(current_signals) >= self.rule.min_signals:
                        segments.append(EffortSegment(
                            signals=current_signals.copy(),
                            rule=self.rule,
                        ))
                    current_signals = [group[i]]
                else:
                    current_signals.append(group[i])

            if current_signals and len(current_signals) >= self.rule.min_signals:
                segments.append(EffortSegment(
                    signals=current_signals,
                    rule=self.rule,
                ))

        return segments

    def add_signal(self, signal: Signal) -> Optional[EffortSegment]:
        """
        Add a signal to the current segment (streaming mode).

        Returns an EffortSegment if a segment boundary is reached.
        """
        self._current_segment_signals.append(signal)

        # Check for segment boundary
        should_close = False

        if self.rule.method == SegmentationMethod.EXPLICIT_MARKER:
            if signal.signal_type == SignalType.ANNOTATION:
                category = signal.metadata.get("category", "")
                if category in self.rule.marker_categories:
                    should_close = True

        elif self.rule.method == SegmentationMethod.TIME_WINDOW:
            if len(self._current_segment_signals) > 1:
                first_time = self._current_segment_signals[0].timestamp
                window = timedelta(minutes=self.rule.time_window_minutes)
                if signal.timestamp >= first_time + window:
                    should_close = True

        elif self.rule.method == SegmentationMethod.ACTIVITY_BOUNDARY:
            if len(self._current_segment_signals) > 1:
                prev_time = self._current_segment_signals[-2].timestamp
                gap_threshold = timedelta(minutes=self.rule.gap_threshold_minutes)
                if signal.timestamp - prev_time > gap_threshold:
                    # Remove last signal (it starts new segment)
                    self._current_segment_signals.pop()
                    should_close = True
                    # Save the signal for next segment
                    next_signal = signal

        # Check max signals
        if len(self._current_segment_signals) >= self.rule.max_signals:
            should_close = True

        if should_close and len(self._current_segment_signals) >= self.rule.min_signals:
            segment = EffortSegment(
                signals=self._current_segment_signals.copy(),
                rule=self.rule,
            )
            self._segments.append(segment)
            self._current_segment_signals = []

            # For activity boundary, start new segment with the triggering signal
            if self.rule.method == SegmentationMethod.ACTIVITY_BOUNDARY and 'next_signal' in locals():
                self._current_segment_signals = [next_signal]

            return segment

        return None

    def flush(self) -> Optional[EffortSegment]:
        """Flush any remaining signals as a final segment"""
        if self._current_segment_signals and len(self._current_segment_signals) >= self.rule.min_signals:
            segment = EffortSegment(
                signals=self._current_segment_signals.copy(),
                rule=self.rule,
            )
            self._segments.append(segment)
            self._current_segment_signals = []
            return segment
        return None

    def get_segments(self) -> List[EffortSegment]:
        """Get all completed segments"""
        return self._segments.copy()

    def get_rule_disclosure(self) -> str:
        """Get human-readable disclosure of segmentation method"""
        return self.rule.get_disclosure()
