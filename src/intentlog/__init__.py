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
from .storage import (
    IntentLogStorage,
    ProjectConfig,
    LLMSettings,
    compute_intent_hash,
    find_project_root,
    StorageError,
    ProjectNotFoundError,
    ProjectExistsError,
    BranchNotFoundError,
    BranchExistsError,
)
from .semantic import SemanticEngine, SemanticDiff, SemanticSearchResult, MergeResolution

# Phase 4: Analytics and Metrics
from .export import IntentExporter, ExportFilter, ExportFormat, AnonymizationConfig
from .analytics import IntentAnalytics, AnalyticsReport, LatencyStats, FrequencyStats
from .metrics import IntentMetrics, IntentDensity, AuditabilityScore, FraudResistance
from .sufficiency import SufficiencyTest, SufficiencyReport, run_sufficiency_test

__all__ = [
    "IntentLog",
    "Intent",
    "audit_logs",
    "IntentLogStorage",
    "ProjectConfig",
    "LLMSettings",
    "compute_intent_hash",
    "find_project_root",
    "StorageError",
    "ProjectNotFoundError",
    "ProjectExistsError",
    "BranchNotFoundError",
    "BranchExistsError",
    "SemanticEngine",
    "SemanticDiff",
    "SemanticSearchResult",
    "MergeResolution",
    # Phase 4: Analytics and Metrics
    "IntentExporter",
    "ExportFilter",
    "ExportFormat",
    "AnonymizationConfig",
    "IntentAnalytics",
    "AnalyticsReport",
    "LatencyStats",
    "FrequencyStats",
    "IntentMetrics",
    "IntentDensity",
    "AuditabilityScore",
    "FraudResistance",
    "SufficiencyTest",
    "SufficiencyReport",
    "run_sufficiency_test",
    "__version__",
]
