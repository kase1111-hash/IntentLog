"""
IntentLog auditing functionality

This module provides tools for auditing intent logs for quality issues
like empty reasoning, loops, prompt injection patterns, and constitutional
violation tracking.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .validation import scan_for_injection_patterns


class AuditError:
    """Represents an audit error"""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message

    def __str__(self):
        return f"{self.error_type}: {self.message}"


def audit_logs(file_path: str, max_repeats: int = 3) -> Tuple[bool, List[AuditError]]:
    """
    Audit intent logs for quality issues.

    Args:
        file_path: Path to the log file to audit
        max_repeats: Maximum number of times an intent can repeat before flagging

    Returns:
        Tuple of (passed: bool, errors: List[AuditError])
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        logs = f.read()

    errors = []

    # 1. Check for "Hallucination Risk" (Empty Intent reasoning)
    if re.search(r"intent_reasoning: ''", logs):
        errors.append(AuditError(
            "HALLUCINATION_RISK",
            "Empty reasoning found in logs."
        ))

    # 2. Check for "Cost/Loop Risk" (Same intent repeated > max_repeats times)
    intents = re.findall(r"intent_name: '(\w+)'", logs)
    for name in set(intents):
        count = intents.count(name)
        if count > max_repeats:
            errors.append(AuditError(
                "LOOP_RISK",
                f"Potential Loop: Intent '{name}' called {count} times."
            ))

    # 3. Check for prompt injection patterns in reasoning fields
    reasoning_blocks = re.findall(r"intent_reasoning: '([^']*)'", logs)
    for reasoning in reasoning_blocks:
        patterns = scan_for_injection_patterns(reasoning)
        if patterns:
            errors.append(AuditError(
                "INJECTION_RISK",
                f"Possible prompt injection detected: {', '.join(patterns)}"
            ))

    return len(errors) == 0, errors


def print_audit_results(passed: bool, errors: List[AuditError]) -> None:
    """Print audit results to console"""
    if errors:
        for error in errors:
            print(str(error))
    else:
        print("IntentLog Audit Passed!")


# ---- Constitutional Violation Tracking ----

# Well-known violation types
VIOLATION_SECRET_IN_OUTBOUND = "SECRET_IN_OUTBOUND"
VIOLATION_INJECTION_PATTERN = "INJECTION_PATTERN_DETECTED"
VIOLATION_PATH_TRAVERSAL = "PATH_TRAVERSAL_ATTEMPT"
VIOLATION_UNAUTHORIZED_URL = "UNAUTHORIZED_URL"
VIOLATION_CHAIN_INTEGRITY = "CHAIN_INTEGRITY_FAILURE"
VIOLATION_UNSIGNED_INTENT = "UNSIGNED_INTENT"


@dataclass
class ConstitutionalViolation:
    """Records a constitutional policy violation."""
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    message: str
    timestamp: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstitutionalViolation":
        return cls(
            violation_type=data["violation_type"],
            severity=data["severity"],
            message=data["message"],
            timestamp=data.get("timestamp", ""),
            context=data.get("context", {}),
        )


class ViolationTracker:
    """
    Tracks constitutional violations in an append-only JSONL file.

    Violations are written to .intentlog/violations.jsonl as one JSON
    object per line, making the file naturally append-only and easy to parse.
    """

    def __init__(self, intentlog_dir: Optional[Path] = None):
        """
        Initialize violation tracker.

        Args:
            intentlog_dir: Path to .intentlog directory. If None, tracking
                          is disabled (violations are only counted in memory).
        """
        self._intentlog_dir = Path(intentlog_dir) if intentlog_dir else None
        self._in_memory: List[ConstitutionalViolation] = []

    @property
    def _violations_file(self) -> Optional[Path]:
        if self._intentlog_dir:
            return self._intentlog_dir / "violations.jsonl"
        return None

    def record_violation(
        self,
        violation_type: str,
        severity: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstitutionalViolation:
        """
        Record a constitutional violation.

        Args:
            violation_type: One of the VIOLATION_* constants
            severity: "critical", "high", "medium", or "low"
            message: Human-readable description
            context: Optional additional context

        Returns:
            The recorded violation
        """
        violation = ConstitutionalViolation(
            violation_type=violation_type,
            severity=severity,
            message=message,
            context=context or {},
        )

        self._in_memory.append(violation)

        # Append to JSONL file
        vf = self._violations_file
        if vf:
            try:
                vf.parent.mkdir(parents=True, exist_ok=True)
                with open(vf, "a") as f:
                    f.write(json.dumps(violation.to_dict(), separators=(",", ":")) + "\n")
            except OSError:
                pass  # Best-effort; don't crash on write failure

        return violation

    def get_violations(
        self,
        since: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[ConstitutionalViolation]:
        """
        Retrieve recorded violations.

        Args:
            since: ISO timestamp — only return violations after this time
            severity: Filter by severity level

        Returns:
            List of matching violations
        """
        violations = []

        # Read from file if available
        vf = self._violations_file
        if vf and vf.exists():
            try:
                with open(vf, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            violations.append(ConstitutionalViolation.from_dict(data))
                        except (json.JSONDecodeError, KeyError):
                            continue
            except OSError:
                pass
        else:
            violations = list(self._in_memory)

        # Apply filters
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        if severity:
            violations = [v for v in violations if v.severity == severity]

        return violations

    def get_violation_count(self) -> Dict[str, int]:
        """
        Get counts of violations by type.

        Returns:
            Dict mapping violation_type to count
        """
        counts: Dict[str, int] = {}
        for v in self.get_violations():
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts
