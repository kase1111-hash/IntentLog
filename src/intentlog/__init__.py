"""
IntentLog: Version Control for Human Reasoning

A conceptual framework and implementation for tracking intent and reasoning
as first-class artifacts in collaborative work.
"""

__version__ = "0.1.0"
__author__ = "IntentLog Contributors"
__license__ = "CC BY-SA 4.0"

from .core import IntentLog, Intent
from .audit import audit_logs

__all__ = ["IntentLog", "Intent", "audit_logs", "__version__"]
