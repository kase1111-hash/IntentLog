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

# Phase 2: Cryptographic Integrity
from .merkle import (
    ChainedIntent,
    MerkleChain,
    chain_intents,
    verify_chain,
    compute_root_hash,
    ChainVerificationResult,
    GENESIS_HASH,
)

# Crypto imports are optional (requires cryptography library)
try:
    from .crypto import (
        KeyManager,
        KeyPair,
        Signature,
        generate_key_pair,
        sign_data,
        verify_signature,
        CryptoError,
        KeyNotFoundError,
        SignatureError,
        CRYPTO_AVAILABLE,
    )
except ImportError:
    # Crypto not available
    CRYPTO_AVAILABLE = False
    KeyManager = None
    KeyPair = None
    Signature = None
    generate_key_pair = None
    sign_data = None
    verify_signature = None
    CryptoError = Exception
    KeyNotFoundError = Exception
    SignatureError = Exception

# Phase 4: Analytics and Metrics
from .export import IntentExporter, ExportFilter, ExportFormat, AnonymizationConfig
from .analytics import IntentAnalytics, AnalyticsReport, LatencyStats, FrequencyStats
from .metrics import IntentMetrics, IntentDensity, AuditabilityScore, FraudResistance
from .sufficiency import SufficiencyTest, SufficiencyReport, run_sufficiency_test

# Phase 5: Context and Decorator
from .context import (
    IntentContext,
    IntentContextManager,
    SessionContext,
    SessionContextManager,
    get_current_intent,
    set_current_intent,
    get_current_session,
    set_current_session,
    get_intent_chain,
    get_current_depth,
    get_session_id,
    intent_scope,
    session_scope,
    # Extended context features
    ContextStatus,
    EnhancedIntentContextManager,
    intent_scope_enhanced,
    register_on_enter_hook,
    register_on_exit_hook,
    unregister_on_enter_hook,
    unregister_on_exit_hook,
    clear_hooks,
    propagate_context_to_env,
    restore_context_from_env,
    get_all_tags,
    get_all_labels,
    has_tag_in_chain,
    get_root_context,
    get_trace_id,
    get_span_id,
    with_tags,
    with_labels,
    INTENT_CONTEXT_ENV_VAR,
)
from .decorator import (
    intent_logger,
    intent_logger_class,
    IntentLoggerConfig,
    LogLevel,
    set_log_level,
    get_log_level,
    should_log,
    get_intent_log,
    clear_intent_log,
    log_intent,
    trace,
)

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
    # Phase 2: Cryptographic Integrity
    "ChainedIntent",
    "MerkleChain",
    "chain_intents",
    "verify_chain",
    "compute_root_hash",
    "ChainVerificationResult",
    "GENESIS_HASH",
    "KeyManager",
    "KeyPair",
    "Signature",
    "generate_key_pair",
    "sign_data",
    "verify_signature",
    "CryptoError",
    "KeyNotFoundError",
    "SignatureError",
    "CRYPTO_AVAILABLE",
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
    # Phase 5: Context and Decorator
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
    # Extended context features
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
    "__version__",
]
