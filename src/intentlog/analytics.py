"""
Analytics Module for IntentLog

Provides statistical analysis and insights for intent logs:
- Latency distribution statistics
- Intent frequency by category
- Error and correction tracking
- Activity patterns and trends
- Report generation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import statistics
import json

from .core import Intent


@dataclass
class LatencyStats:
    """Statistics for latency measurements"""
    count: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_dev_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    # Aliases for CLI compatibility
    @property
    def mean(self) -> float:
        return self.mean_ms

    @property
    def median(self) -> float:
        return self.median_ms

    @property
    def std_dev(self) -> float:
        return self.std_dev_ms

    @property
    def min(self) -> float:
        return self.min_ms

    @property
    def max(self) -> float:
        return self.max_ms

    @property
    def p95(self) -> float:
        return self.p95_ms

    @property
    def p99(self) -> float:
        return self.p99_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


@dataclass
class FrequencyStats:
    """Statistics for intent frequency"""
    total_count: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_hour: Dict[int, int] = field(default_factory=dict)
    by_day: Dict[str, int] = field(default_factory=dict)
    by_week: Dict[str, int] = field(default_factory=dict)
    top_names: List[Tuple[str, int]] = field(default_factory=list)
    date_range_days: int = 0
    intents_per_day: float = 0.0
    intents_per_hour: float = 0.0
    peak_hour: int = 0
    peak_day: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "by_category": self.by_category,
            "by_hour": self.by_hour,
            "by_day": self.by_day,
            "by_week": self.by_week,
            "top_names": self.top_names,
        }


@dataclass
class ErrorStats:
    """Statistics for errors and corrections"""
    total_errors: int = 0
    error_rate: float = 0.0
    corrections: int = 0
    correction_rate: float = 0.0
    error_categories: Dict[str, int] = field(default_factory=dict)
    high_latency_count: int = 0
    high_latency_threshold_ms: float = 5000.0

    @property
    def errors_by_type(self) -> Dict[str, int]:
        """Alias for error_categories"""
        return self.error_categories

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "corrections": self.corrections,
            "correction_rate": round(self.correction_rate, 4),
            "error_categories": self.error_categories,
            "high_latency_count": self.high_latency_count,
            "high_latency_threshold_ms": self.high_latency_threshold_ms,
        }


@dataclass
class ActivityPattern:
    """Activity pattern analysis"""
    active_hours: List[int] = field(default_factory=list)  # Most active hours
    active_days: List[str] = field(default_factory=list)   # Most active days
    avg_intents_per_day: float = 0.0
    avg_intents_per_session: float = 0.0
    session_count: int = 0
    avg_session_duration_minutes: float = 0.0
    longest_gap_minutes: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_hours": self.active_hours,
            "active_days": self.active_days,
            "avg_intents_per_day": round(self.avg_intents_per_day, 2),
            "avg_intents_per_session": round(self.avg_intents_per_session, 2),
            "session_count": self.session_count,
            "avg_session_duration_minutes": round(self.avg_session_duration_minutes, 2),
            "longest_gap_minutes": round(self.longest_gap_minutes, 2),
        }


@dataclass
class AnalyticsReport:
    """Complete analytics report"""
    generated_at: datetime = field(default_factory=datetime.now)
    intent_count: int = 0
    _date_range: Tuple[Optional[datetime], Optional[datetime]] = field(default=(None, None), repr=False)
    latency: Optional[LatencyStats] = None
    frequency: Optional[FrequencyStats] = None
    errors: Optional[ErrorStats] = None
    activity: Optional[ActivityPattern] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_intents(self) -> int:
        """Alias for intent_count"""
        return self.intent_count

    @property
    def date_range(self) -> Dict[str, Optional[str]]:
        """Return date range as dictionary for CLI compatibility"""
        return {
            "start": self._date_range[0].isoformat() if self._date_range[0] else None,
            "end": self._date_range[1].isoformat() if self._date_range[1] else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "intent_count": self.intent_count,
            "date_range": {
                "start": self._date_range[0].isoformat() if self._date_range[0] else None,
                "end": self._date_range[1].isoformat() if self._date_range[1] else None,
            },
            "latency": self.latency.to_dict() if self.latency else None,
            "frequency": self.frequency.to_dict() if self.frequency else None,
            "errors": self.errors.to_dict() if self.errors else None,
            "activity": self.activity.to_dict() if self.activity else None,
            "custom_metrics": self.custom_metrics,
        }

    def to_json(self, pretty: bool = True) -> str:
        """Convert to JSON string"""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, default=str)


class IntentAnalytics:
    """
    Analytics engine for intent logs.

    Provides comprehensive analysis of intent patterns,
    performance metrics, and activity insights.
    """

    def __init__(
        self,
        intents: List[Intent],
        session_gap_minutes: float = 30.0,
        high_latency_threshold_ms: float = 5000.0,
    ):
        """
        Initialize analytics engine.

        Args:
            intents: List of intents to analyze
            session_gap_minutes: Gap threshold for session detection
            high_latency_threshold_ms: Threshold for high latency
        """
        self.intents = sorted(intents, key=lambda i: i.timestamp)
        self.session_gap_minutes = session_gap_minutes
        self.high_latency_threshold = high_latency_threshold_ms

    def compute_latency_stats(self) -> LatencyStats:
        """Compute latency statistics from intent metadata"""
        latencies = []
        for intent in self.intents:
            latency = intent.metadata.get("latency_ms")
            if latency is not None:
                latencies.append(float(latency))

        if not latencies:
            return LatencyStats()

        latencies.sort()
        n = len(latencies)

        return LatencyStats(
            count=n,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_dev_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p90_ms=latencies[int(n * 0.9)] if n > 0 else 0.0,
            p95_ms=latencies[int(n * 0.95)] if n > 0 else 0.0,
            p99_ms=latencies[int(n * 0.99)] if n > 0 else 0.0,
        )

    def compute_frequency_stats(self) -> FrequencyStats:
        """Compute frequency statistics"""
        if not self.intents:
            return FrequencyStats()

        # By category
        categories = Counter()
        for intent in self.intents:
            cat = intent.metadata.get("category", "uncategorized")
            categories[cat] += 1

        # By hour
        hours = Counter()
        for intent in self.intents:
            hours[intent.timestamp.hour] += 1

        # By day of week
        days = Counter()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for intent in self.intents:
            days[day_names[intent.timestamp.weekday()]] += 1

        # By week (ISO week number)
        weeks = Counter()
        for intent in self.intents:
            week_key = intent.timestamp.strftime("%Y-W%W")
            weeks[week_key] += 1

        # Top intent names
        names = Counter(i.intent_name for i in self.intents)
        top_names = names.most_common(10)

        # Calculate date range and rates
        first_ts = self.intents[0].timestamp
        last_ts = self.intents[-1].timestamp
        date_range_days = max(1, (last_ts - first_ts).days + 1)
        total_hours = max(1, (last_ts - first_ts).total_seconds() / 3600)

        # Find peaks
        peak_hour = max(hours.keys(), key=lambda h: hours[h]) if hours else 0
        peak_day = max(days.keys(), key=lambda d: days[d]) if days else ""

        return FrequencyStats(
            total_count=len(self.intents),
            by_category=dict(categories),
            by_hour=dict(hours),
            by_day=dict(days),
            by_week=dict(weeks),
            top_names=top_names,
            date_range_days=date_range_days,
            intents_per_day=len(self.intents) / date_range_days,
            intents_per_hour=len(self.intents) / total_hours,
            peak_hour=peak_hour,
            peak_day=peak_day,
        )

    def compute_error_stats(self) -> ErrorStats:
        """Compute error and correction statistics"""
        total = len(self.intents)
        if total == 0:
            return ErrorStats()

        errors = 0
        corrections = 0
        error_categories = Counter()
        high_latency = 0

        for intent in self.intents:
            # Check for error indicators
            is_error = intent.metadata.get("is_error", False)
            error_type = intent.metadata.get("error_type", "")

            if is_error or error_type:
                errors += 1
                if error_type:
                    error_categories[error_type] += 1

            # Check for corrections
            if intent.metadata.get("is_correction", False):
                corrections += 1

            # Check for high latency
            latency = intent.metadata.get("latency_ms")
            if latency and float(latency) > self.high_latency_threshold:
                high_latency += 1

        return ErrorStats(
            total_errors=errors,
            error_rate=errors / total if total > 0 else 0.0,
            corrections=corrections,
            correction_rate=corrections / total if total > 0 else 0.0,
            error_categories=dict(error_categories),
            high_latency_count=high_latency,
            high_latency_threshold_ms=self.high_latency_threshold,
        )

    def compute_activity_pattern(self) -> ActivityPattern:
        """Analyze activity patterns"""
        if not self.intents:
            return ActivityPattern()

        # Detect sessions based on gaps
        sessions: List[List[Intent]] = []
        current_session: List[Intent] = []

        for i, intent in enumerate(self.intents):
            if i == 0:
                current_session.append(intent)
                continue

            gap = (intent.timestamp - self.intents[i-1].timestamp).total_seconds() / 60
            if gap > self.session_gap_minutes:
                if current_session:
                    sessions.append(current_session)
                current_session = [intent]
            else:
                current_session.append(intent)

        if current_session:
            sessions.append(current_session)

        # Calculate session metrics
        session_durations = []
        for session in sessions:
            if len(session) > 1:
                duration = (session[-1].timestamp - session[0].timestamp).total_seconds() / 60
                session_durations.append(duration)

        # Find most active hours
        hour_counts = Counter(i.timestamp.hour for i in self.intents)
        active_hours = [h for h, _ in hour_counts.most_common(5)]

        # Find most active days
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_counts = Counter(day_names[i.timestamp.weekday()] for i in self.intents)
        active_days = [d for d, _ in day_counts.most_common(3)]

        # Calculate daily average
        if self.intents:
            first_day = self.intents[0].timestamp.date()
            last_day = self.intents[-1].timestamp.date()
            days_span = (last_day - first_day).days + 1
            avg_per_day = len(self.intents) / days_span if days_span > 0 else 0
        else:
            avg_per_day = 0

        # Find longest gap
        longest_gap = 0
        for i in range(1, len(self.intents)):
            gap = (self.intents[i].timestamp - self.intents[i-1].timestamp).total_seconds() / 60
            longest_gap = max(longest_gap, gap)

        return ActivityPattern(
            active_hours=active_hours,
            active_days=active_days,
            avg_intents_per_day=avg_per_day,
            avg_intents_per_session=len(self.intents) / len(sessions) if sessions else 0,
            session_count=len(sessions),
            avg_session_duration_minutes=statistics.mean(session_durations) if session_durations else 0,
            longest_gap_minutes=longest_gap,
        )

    def generate_report(self) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        date_range = (None, None)
        if self.intents:
            date_range = (
                min(i.timestamp for i in self.intents),
                max(i.timestamp for i in self.intents),
            )

        return AnalyticsReport(
            intent_count=len(self.intents),
            _date_range=date_range,
            latency=self.compute_latency_stats(),
            frequency=self.compute_frequency_stats(),
            errors=self.compute_error_stats(),
            activity=self.compute_activity_pattern(),
        )

    def get_trending_intents(
        self,
        window_days: int = 7,
        top_n: int = 10,
    ) -> List[Tuple[str, int, float]]:
        """
        Get trending intents based on recent activity.

        Returns list of (intent_name, count, trend_score)
        """
        now = datetime.now()
        cutoff = now - timedelta(days=window_days)

        recent = [i for i in self.intents if i.timestamp >= cutoff]
        older = [i for i in self.intents if i.timestamp < cutoff]

        recent_counts = Counter(i.intent_name for i in recent)
        older_counts = Counter(i.intent_name for i in older)

        trends = []
        for name, count in recent_counts.most_common(top_n * 2):
            old_count = older_counts.get(name, 0)
            if old_count > 0:
                trend = (count - old_count) / old_count
            else:
                trend = float(count)  # New intent
            trends.append((name, count, trend))

        # Sort by trend score
        trends.sort(key=lambda x: x[2], reverse=True)
        return trends[:top_n]

    def get_bottlenecks(
        self,
        latency_threshold_ms: Optional[float] = None,
        top_n: int = 10,
    ) -> List[Tuple[str, float, int]]:
        """
        Identify bottleneck intents (high latency).

        Returns list of (intent_name, avg_latency_ms, count)
        """
        threshold = latency_threshold_ms or self.high_latency_threshold

        # Group by name and calculate average latency
        latency_by_name: Dict[str, List[float]] = {}
        for intent in self.intents:
            latency = intent.metadata.get("latency_ms")
            if latency is not None:
                name = intent.intent_name
                if name not in latency_by_name:
                    latency_by_name[name] = []
                latency_by_name[name].append(float(latency))

        bottlenecks = []
        for name, latencies in latency_by_name.items():
            avg_latency = statistics.mean(latencies)
            if avg_latency >= threshold:
                bottlenecks.append((name, avg_latency, len(latencies)))

        # Sort by average latency
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks[:top_n]

    def compare_periods(
        self,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime,
    ) -> Dict[str, Any]:
        """
        Compare analytics between two time periods.

        Returns comparison metrics.
        """
        p1_intents = [i for i in self.intents
                      if period1_start <= i.timestamp <= period1_end]
        p2_intents = [i for i in self.intents
                      if period2_start <= i.timestamp <= period2_end]

        p1_analytics = IntentAnalytics(p1_intents, self.session_gap_minutes, self.high_latency_threshold)
        p2_analytics = IntentAnalytics(p2_intents, self.session_gap_minutes, self.high_latency_threshold)

        p1_report = p1_analytics.generate_report()
        p2_report = p2_analytics.generate_report()

        # Calculate deltas
        def safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            return b - a

        return {
            "period1": {
                "start": period1_start.isoformat(),
                "end": period1_end.isoformat(),
                "report": p1_report.to_dict(),
            },
            "period2": {
                "start": period2_start.isoformat(),
                "end": period2_end.isoformat(),
                "report": p2_report.to_dict(),
            },
            "comparison": {
                "intent_count_delta": p2_report.intent_count - p1_report.intent_count,
                "latency_mean_delta": safe_delta(
                    p1_report.latency.mean_ms if p1_report.latency else None,
                    p2_report.latency.mean_ms if p2_report.latency else None,
                ),
                "error_rate_delta": safe_delta(
                    p1_report.errors.error_rate if p1_report.errors else None,
                    p2_report.errors.error_rate if p2_report.errors else None,
                ),
            },
        }


def generate_summary(intents: List[Intent]) -> str:
    """
    Generate a human-readable summary of intent analytics.

    Args:
        intents: List of intents to analyze

    Returns:
        Formatted summary string
    """
    analytics = IntentAnalytics(intents)
    report = analytics.generate_report()

    lines = [
        "=" * 50,
        "IntentLog Analytics Summary",
        "=" * 50,
        "",
        f"Total Intents: {report.intent_count}",
    ]

    if report._date_range[0]:
        lines.append(f"Date Range: {report._date_range[0].strftime('%Y-%m-%d')} to {report._date_range[1].strftime('%Y-%m-%d')}")

    lines.append("")

    # Latency
    if report.latency and report.latency.count > 0:
        lines.extend([
            "Latency Statistics:",
            f"  Mean: {report.latency.mean_ms:.1f}ms",
            f"  Median: {report.latency.median_ms:.1f}ms",
            f"  P95: {report.latency.p95_ms:.1f}ms",
            f"  Max: {report.latency.max_ms:.1f}ms",
            "",
        ])

    # Frequency
    if report.frequency:
        lines.extend([
            "Top Categories:",
        ])
        for cat, count in sorted(report.frequency.by_category.items(),
                                 key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"  {cat}: {count}")
        lines.append("")

    # Activity
    if report.activity:
        lines.extend([
            "Activity Patterns:",
            f"  Sessions: {report.activity.session_count}",
            f"  Avg/day: {report.activity.avg_intents_per_day:.1f} intents",
            f"  Most active: {', '.join(report.activity.active_days)}",
            "",
        ])

    # Errors
    if report.errors and report.errors.total_errors > 0:
        lines.extend([
            "Error Summary:",
            f"  Total errors: {report.errors.total_errors}",
            f"  Error rate: {report.errors.error_rate:.2%}",
            f"  High latency: {report.errors.high_latency_count}",
            "",
        ])

    lines.append("=" * 50)

    return "\n".join(lines)
