"""
IntentLog: Version Control for Human Reasoning

A conceptual framework and implementation for tracking intent and reasoning
as first-class artifacts in collaborative work.
"""

__version__ = "0.2.0"
__author__ = "IntentLog Contributors"
__license__ = "CC BY-SA 4.0"

# ---- Core imports (always available, no external deps) ----

from .core import IntentLog, Intent
from .audit import audit_logs
from .logging import (
    configure_logging,
    get_logger,
    log_context,
    log_function_call,
    LogFormat,
    LogLevel as StructuredLogLevel,
    LogContext,
    IntentLogLogger,
)
from .validation import (
    validate_project_name,
    validate_branch_name,
    validate_path_within_directory,
    sanitize_filename,
    ValidationError,
)
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
from .merkle import (
    ChainedIntent,
    MerkleChain,
    chain_intents,
    verify_chain,
    compute_root_hash,
    ChainVerificationResult,
    GENESIS_HASH,
)

# ---- Crypto availability flag (set by lazy loader or direct import) ----

CRYPTO_AVAILABLE = False
try:
    from .crypto import CRYPTO_AVAILABLE
except (ImportError, ModuleNotFoundError):
    pass

# ---- Lazy loading for non-core modules ----
# These are loaded on first access to avoid import-time failures
# (e.g., cryptography library Rust binding panics) and to keep
# package load time fast.

_LAZY_MODULES = {
    # Crypto (requires cryptography library)
    "KeyManager": ".crypto",
    "KeyPair": ".crypto",
    "Signature": ".crypto",
    "generate_key_pair": ".crypto",
    "sign_data": ".crypto",
    "verify_signature": ".crypto",
    "CryptoError": ".crypto",
    "KeyNotFoundError": ".crypto",
    "SignatureError": ".crypto",
    # Semantic (requires LLM providers)
    "SemanticEngine": ".semantic",
    "SemanticDiff": ".semantic",
    "SemanticSearchResult": ".semantic",
    "MergeResolution": ".semantic",
    "FormalizationType": ".semantic",
    "FormalizedOutput": ".semantic",
    "ProvenanceRecord": ".semantic",
    # Analytics & Metrics
    "IntentExporter": ".export",
    "ExportFilter": ".export",
    "ExportFormat": ".export",
    "AnonymizationConfig": ".export",
    "IntentAnalytics": ".analytics",
    "AnalyticsReport": ".analytics",
    "LatencyStats": ".analytics",
    "FrequencyStats": ".analytics",
    "IntentMetrics": ".metrics",
    "IntentDensity": ".metrics",
    "AuditabilityScore": ".metrics",
    "FraudResistance": ".metrics",
    # Context & Decorator
    "IntentContext": ".context",
    "IntentContextManager": ".context",
    "SessionContext": ".context",
    "SessionContextManager": ".context",
    "get_current_intent": ".context",
    "set_current_intent": ".context",
    "get_current_session": ".context",
    "set_current_session": ".context",
    "get_intent_chain": ".context",
    "get_current_depth": ".context",
    "get_session_id": ".context",
    "intent_scope": ".context",
    "session_scope": ".context",
    "ContextStatus": ".context",
    "EnhancedIntentContextManager": ".context",
    "intent_scope_enhanced": ".context",
    "register_on_enter_hook": ".context",
    "register_on_exit_hook": ".context",
    "unregister_on_enter_hook": ".context",
    "unregister_on_exit_hook": ".context",
    "clear_hooks": ".context",
    "propagate_context_to_env": ".context",
    "restore_context_from_env": ".context",
    "get_all_tags": ".context",
    "get_all_labels": ".context",
    "has_tag_in_chain": ".context",
    "get_root_context": ".context",
    "get_trace_id": ".context",
    "get_span_id": ".context",
    "with_tags": ".context",
    "with_labels": ".context",
    "INTENT_CONTEXT_ENV_VAR": ".context",
    "intent_logger": ".decorator",
    "intent_logger_class": ".decorator",
    "IntentLoggerConfig": ".decorator",
    "LogLevel": ".decorator",
    "set_log_level": ".decorator",
    "get_log_level": ".decorator",
    "should_log": ".decorator",
    "get_intent_log": ".decorator",
    "clear_intent_log": ".decorator",
    "log_intent": ".decorator",
    "trace": ".decorator",
    # Backup and Recovery
    "BackupManager": ".backup",
    "BackupMetadata": ".backup",
    "RestoreResult": ".backup",
    "BackupError": ".backup",
    "RestoreError": ".backup",
    "create_backup": ".backup",
    "restore_backup": ".backup",
    "list_backups": ".backup",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'intentlog' has no attribute {name!r}")


__all__ = [
    # Core
    "IntentLog",
    "Intent",
    "audit_logs",
    # Logging
    "configure_logging",
    "get_logger",
    "log_context",
    "log_function_call",
    "LogFormat",
    "StructuredLogLevel",
    "LogContext",
    "IntentLogLogger",
    # Validation
    "validate_project_name",
    "validate_branch_name",
    "validate_path_within_directory",
    "sanitize_filename",
    "ValidationError",
    # Storage
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
    # Merkle
    "ChainedIntent",
    "MerkleChain",
    "chain_intents",
    "verify_chain",
    "compute_root_hash",
    "ChainVerificationResult",
    "GENESIS_HASH",
    # Crypto (lazy)
    "CRYPTO_AVAILABLE",
    "KeyManager",
    "KeyPair",
    "Signature",
    "generate_key_pair",
    "sign_data",
    "verify_signature",
    "CryptoError",
    "KeyNotFoundError",
    "SignatureError",
    # Semantic (lazy)
    "SemanticEngine",
    "SemanticDiff",
    "SemanticSearchResult",
    "MergeResolution",
    "FormalizationType",
    "FormalizedOutput",
    "ProvenanceRecord",
    # Analytics (lazy)
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
    # Context & Decorator (lazy)
    "IntentContext",
    "IntentContextManager",
    "SessionContext",
    "SessionContextManager",
    "get_current_intent",
    "set_current_intent",
    "get_current_session",
    "set_current_session",
    "get_intent_chain",
    "get_current_depth",
    "get_session_id",
    "intent_scope",
    "session_scope",
    "ContextStatus",
    "EnhancedIntentContextManager",
    "intent_scope_enhanced",
    "register_on_enter_hook",
    "register_on_exit_hook",
    "unregister_on_enter_hook",
    "unregister_on_exit_hook",
    "clear_hooks",
    "propagate_context_to_env",
    "restore_context_from_env",
    "get_all_tags",
    "get_all_labels",
    "has_tag_in_chain",
    "get_root_context",
    "get_trace_id",
    "get_span_id",
    "with_tags",
    "with_labels",
    "INTENT_CONTEXT_ENV_VAR",
    "intent_logger",
    "intent_logger_class",
    "IntentLoggerConfig",
    "LogLevel",
    "set_log_level",
    "get_log_level",
    "should_log",
    "get_intent_log",
    "clear_intent_log",
    "log_intent",
    "trace",
    # Backup (lazy)
    "BackupManager",
    "BackupMetadata",
    "RestoreResult",
    "BackupError",
    "RestoreError",
    "create_backup",
    "restore_backup",
    "list_backups",
    "__version__",
]
