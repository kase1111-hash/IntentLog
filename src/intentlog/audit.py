"""
IntentLog auditing functionality

This module provides tools for auditing intent logs for quality issues
like empty reasoning, loops, and other problems.
"""

import re
from typing import List, Tuple


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
    with open(file_path, 'r') as f:
        logs = f.read()

    errors = []

    # 1. Check for "Hallucination Risk" (Empty Intent reasoning)
    if re.search(r"intent_reasoning: ''", logs):
        errors.append(AuditError(
            "HALLUCINATION_RISK",
            "❌ Empty reasoning found in logs."
        ))

    # 2. Check for "Cost/Loop Risk" (Same intent repeated > max_repeats times)
    intents = re.findall(r"intent_name: '(\w+)'", logs)
    for name in set(intents):
        count = intents.count(name)
        if count > max_repeats:
            errors.append(AuditError(
                "LOOP_RISK",
                f"⚠️ Potential Loop: Intent '{name}' called {count} times."
            ))

    return len(errors) == 0, errors


def print_audit_results(passed: bool, errors: List[AuditError]) -> None:
    """Print audit results to console"""
    if errors:
        for error in errors:
            print(str(error))
    else:
        print("✅ IntentLog Audit Passed!")
