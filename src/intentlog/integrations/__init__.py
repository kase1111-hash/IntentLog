"""
IntentLog Integrations

This module provides integration points with external systems including:
- Memory Vault for secure classified storage
- Boundary-Daemon for security policy enforcement
- Boundary-SIEM for security event logging and compliance
"""

from .memory_vault import (
    MemoryVaultIntegration,
    MemoryVaultConfig,
    ClassificationLevel,
)
from .llm_classifier import (
    LLMIntentClassifier,
    IntentCategory,
    ClassificationResult,
    classify_intent_with_llm,
)
from .boundary_daemon import (
    BoundaryMode,
    PolicyDecision,
    PolicyRequest,
    PolicyResponse,
    AuditEvent,
    BoundaryDaemonConfig,
    BoundaryDaemonError,
    PolicyDeniedError,
    DaemonUnavailableError,
    BoundaryDaemonIntegration,
    policy_required,
    get_boundary_daemon_integration,
    configure_boundary_daemon,
)
from .boundary_siem import (
    EventSeverity,
    EventCategory,
    SIEMEvent,
    BoundarySIEMConfig,
    BoundarySIEMError,
    EventDeliveryError,
    BoundarySIEMIntegration,
    get_siem_integration,
    configure_siem,
    shutdown_siem,
)

__all__ = [
    # Memory Vault
    "MemoryVaultIntegration",
    "MemoryVaultConfig",
    "ClassificationLevel",
    # LLM Classification
    "LLMIntentClassifier",
    "IntentCategory",
    "ClassificationResult",
    "classify_intent_with_llm",
    # Boundary-Daemon
    "BoundaryMode",
    "PolicyDecision",
    "PolicyRequest",
    "PolicyResponse",
    "AuditEvent",
    "BoundaryDaemonConfig",
    "BoundaryDaemonError",
    "PolicyDeniedError",
    "DaemonUnavailableError",
    "BoundaryDaemonIntegration",
    "policy_required",
    "get_boundary_daemon_integration",
    "configure_boundary_daemon",
    # Boundary-SIEM
    "EventSeverity",
    "EventCategory",
    "SIEMEvent",
    "BoundarySIEMConfig",
    "BoundarySIEMError",
    "EventDeliveryError",
    "BoundarySIEMIntegration",
    "get_siem_integration",
    "configure_siem",
    "shutdown_siem",
]
