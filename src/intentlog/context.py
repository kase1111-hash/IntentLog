"""
Context Management for IntentLog

Provides thread-local and async-safe context for tracking the current
intent during nested function calls. This enables automatic parent-child
relationships when using the @intent_logger decorator.

Features:
- Thread-local storage for sync code
- contextvars for async code
- Session tracking
- Automatic parent chain management
- Context serialization for distributed tracing
- Context hooks for callbacks
- Tags and labels for categorization
- Timeout support
"""

import contextvars
import threading
import uuid
import json
import base64
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .merkle import ChainedIntent


# Context variable for async-safe intent tracking
_current_intent: contextvars.ContextVar[Optional["IntentContext"]] = contextvars.ContextVar(
    "current_intent", default=None
)

# Thread-local storage for sync code fallback
_thread_local = threading.local()


class ContextStatus(Enum):
    """Status of an intent context."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class IntentContext:
    """
    Context for the currently executing intent.

    Tracks the intent being logged and provides parent chain info
    for nested intent logging.
    """
    intent_id: str
    intent_name: str
    start_time: datetime = field(default_factory=datetime.now)
    parent_context: Optional["IntentContext"] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # Child intent IDs
    tags: Set[str] = field(default_factory=set)  # Tags for categorization
    labels: Dict[str, str] = field(default_factory=dict)  # Key-value labels
    timeout_at: Optional[datetime] = None  # Auto-complete deadline
    status: ContextStatus = ContextStatus.ACTIVE
    end_time: Optional[datetime] = None
    trace_id: Optional[str] = None  # Distributed trace ID
    span_id: Optional[str] = None  # Span ID for distributed tracing

    def __post_init__(self):
        # Generate trace/span IDs if not provided
        if self.trace_id is None:
            if self.parent_context and self.parent_context.trace_id:
                self.trace_id = self.parent_context.trace_id
            else:
                self.trace_id = str(uuid.uuid4()).replace("-", "")[:32]
        if self.span_id is None:
            self.span_id = str(uuid.uuid4()).replace("-", "")[:16]

    @property
    def depth(self) -> int:
        """Get nesting depth (0 = root)"""
        if self.parent_context is None:
            return 0
        return self.parent_context.depth + 1

    @property
    def parent_id(self) -> Optional[str]:
        """Get parent intent ID if any"""
        if self.parent_context:
            return self.parent_context.intent_id
        return None

    @property
    def parent_span_id(self) -> Optional[str]:
        """Get parent span ID for distributed tracing"""
        if self.parent_context:
            return self.parent_context.span_id
        return None

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    @property
    def is_timed_out(self) -> bool:
        """Check if context has timed out"""
        if self.timeout_at is None:
            return False
        return datetime.now() > self.timeout_at

    @property
    def is_active(self) -> bool:
        """Check if context is still active"""
        return self.status == ContextStatus.ACTIVE and not self.is_timed_out

    def add_tag(self, tag: str) -> "IntentContext":
        """Add a tag to this context"""
        self.tags.add(tag)
        return self

    def add_tags(self, *tags: str) -> "IntentContext":
        """Add multiple tags to this context"""
        self.tags.update(tags)
        return self

    def has_tag(self, tag: str) -> bool:
        """Check if context has a specific tag"""
        return tag in self.tags

    def set_label(self, key: str, value: str) -> "IntentContext":
        """Set a label on this context"""
        self.labels[key] = value
        return self

    def get_label(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a label value"""
        return self.labels.get(key, default)

    def complete(self, status: ContextStatus = ContextStatus.COMPLETED) -> None:
        """Mark context as completed"""
        self.status = status
        self.end_time = datetime.now()

    def fail(self, error: Optional[str] = None) -> None:
        """Mark context as failed"""
        self.status = ContextStatus.FAILED
        self.end_time = datetime.now()
        if error:
            self.metadata["error"] = error

    def to_dict(self) -> Dict[str, Any]:
        """Export context to dictionary"""
        result = {
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "start_time": self.start_time.isoformat(),
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "depth": self.depth,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "labels": self.labels,
            "status": self.status.value,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        if self.timeout_at:
            result["timeout_at"] = self.timeout_at.isoformat()
        return result

    def serialize(self) -> str:
        """
        Serialize context for cross-process propagation.

        Returns:
            Base64-encoded JSON string
        """
        data = {
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "start_time": self.start_time.isoformat(),
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tags": list(self.tags),
            "labels": self.labels,
            "depth": self.depth,
        }
        json_str = json.dumps(data, separators=(",", ":"))
        return base64.b64encode(json_str.encode()).decode()

    @classmethod
    def deserialize(cls, encoded: str) -> "IntentContext":
        """
        Deserialize context from cross-process propagation.

        Args:
            encoded: Base64-encoded JSON string

        Returns:
            IntentContext restored from serialized data
        """
        json_str = base64.b64decode(encoded.encode()).decode()
        data = json.loads(json_str)
        return cls(
            intent_id=data["intent_id"],
            intent_name=data["intent_name"],
            start_time=datetime.fromisoformat(data["start_time"]),
            session_id=data.get("session_id"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            tags=set(data.get("tags", [])),
            labels=data.get("labels", {}),
        )

    def to_trace_headers(self) -> Dict[str, str]:
        """
        Generate W3C Trace Context compatible headers.

        Returns:
            Dict with traceparent and tracestate headers
        """
        # W3C Trace Context format: version-trace_id-span_id-flags
        traceparent = f"00-{self.trace_id}-{self.span_id}-01"
        headers = {
            "traceparent": traceparent,
            "x-intent-id": self.intent_id,
            "x-intent-name": self.intent_name,
        }
        if self.session_id:
            headers["x-session-id"] = self.session_id
        return headers

    @classmethod
    def from_trace_headers(
        cls,
        headers: Dict[str, str],
        intent_name: Optional[str] = None,
    ) -> "IntentContext":
        """
        Create context from incoming trace headers.

        Args:
            headers: Dict containing trace headers
            intent_name: Optional name for the new context

        Returns:
            New IntentContext with trace info from headers
        """
        trace_id = None
        parent_span_id = None

        # Parse W3C traceparent header
        if "traceparent" in headers:
            parts = headers["traceparent"].split("-")
            if len(parts) >= 3:
                trace_id = parts[1]
                parent_span_id = parts[2]

        # Get intent info from custom headers
        intent_id = headers.get("x-intent-id", str(uuid.uuid4()))
        name = intent_name or headers.get("x-intent-name", "remote-call")
        session_id = headers.get("x-session-id")

        ctx = cls(
            intent_id=intent_id,
            intent_name=name,
            trace_id=trace_id,
            session_id=session_id,
        )

        # Store parent span for reference
        if parent_span_id:
            ctx.metadata["parent_span_id"] = parent_span_id

        return ctx


@dataclass
class SessionContext:
    """
    Session-level context for tracking user journeys.

    A session groups multiple intents together across a user session.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
            "intent_count": self.intent_count,
        }


# Session context variable
_current_session: contextvars.ContextVar[Optional[SessionContext]] = contextvars.ContextVar(
    "current_session", default=None
)


def get_current_intent() -> Optional[IntentContext]:
    """
    Get the current intent context.

    Works in both sync and async code.

    Returns:
        Current IntentContext or None if not in an intent
    """
    # Try contextvar first (works for async)
    ctx = _current_intent.get()
    if ctx is not None:
        return ctx

    # Fall back to thread-local for sync code
    return getattr(_thread_local, "current_intent", None)


def set_current_intent(ctx: Optional[IntentContext]) -> None:
    """
    Set the current intent context.

    Args:
        ctx: IntentContext to set, or None to clear
    """
    _current_intent.set(ctx)
    _thread_local.current_intent = ctx


def get_current_session() -> Optional[SessionContext]:
    """Get the current session context."""
    return _current_session.get()


def set_current_session(session: Optional[SessionContext]) -> None:
    """Set the current session context."""
    _current_session.set(session)


class IntentContextManager:
    """
    Context manager for intent tracking.

    Usage:
        with IntentContextManager("my-intent", "Doing something") as ctx:
            # Code here runs within the intent context
            pass
    """

    def __init__(
        self,
        intent_name: str,
        intent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.intent_name = intent_name
        self.intent_id = intent_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.context: Optional[IntentContext] = None
        self.previous_context: Optional[IntentContext] = None
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> IntentContext:
        # Get parent context
        parent = get_current_intent()

        # Get session
        session = get_current_session()
        session_id = session.session_id if session else None

        # Create new context
        self.context = IntentContext(
            intent_id=self.intent_id,
            intent_name=self.intent_name,
            parent_context=parent,
            session_id=session_id,
            metadata=self.metadata,
        )

        # Track child in parent
        if parent:
            parent.children.append(self.intent_id)

        # Update session count
        if session:
            session.intent_count += 1

        # Save previous and set new
        self.previous_context = get_current_intent()
        set_current_intent(self.context)

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        set_current_intent(self.previous_context)
        return False

    async def __aenter__(self) -> IntentContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class SessionContextManager:
    """
    Context manager for session tracking.

    Usage:
        with SessionContextManager(user_id="user123") as session:
            # All intents here belong to this session
            pass
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.session: Optional[SessionContext] = None
        self.previous_session: Optional[SessionContext] = None

    def __enter__(self) -> SessionContext:
        self.previous_session = get_current_session()

        self.session = SessionContext(
            session_id=self.session_id or str(uuid.uuid4())[:8],
            user_id=self.user_id,
            metadata=self.metadata,
        )

        set_current_session(self.session)
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_current_session(self.previous_session)
        return False

    async def __aenter__(self) -> SessionContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def intent_scope(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> IntentContextManager:
    """
    Create an intent scope for tracking.

    Args:
        name: Name of the intent
        metadata: Optional metadata

    Returns:
        IntentContextManager for use with 'with' statement

    Example:
        with intent_scope("processing-data") as ctx:
            process_data()
            print(f"Depth: {ctx.depth}")
    """
    return IntentContextManager(name, metadata=metadata)


def session_scope(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionContextManager:
    """
    Create a session scope for tracking.

    Args:
        user_id: Optional user identifier
        session_id: Optional session ID (auto-generated if not provided)
        metadata: Optional metadata

    Returns:
        SessionContextManager for use with 'with' statement

    Example:
        with session_scope(user_id="user123") as session:
            with intent_scope("action1"):
                do_something()
            with intent_scope("action2"):
                do_something_else()
            print(f"Total intents: {session.intent_count}")
    """
    return SessionContextManager(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
    )


def get_intent_chain() -> List[IntentContext]:
    """
    Get the full chain of parent intents from root to current.

    Returns:
        List of IntentContext from root to current (inclusive)
    """
    chain = []
    current = get_current_intent()

    while current:
        chain.append(current)
        current = current.parent_context

    return list(reversed(chain))


def get_current_depth() -> int:
    """Get current nesting depth (0 if not in intent)"""
    ctx = get_current_intent()
    return ctx.depth if ctx else 0


def get_session_id() -> Optional[str]:
    """Get current session ID if in a session"""
    session = get_current_session()
    return session.session_id if session else None


# ============================================================================
# Context Hooks System
# ============================================================================

# Type aliases for hook callbacks
OnEnterHook = Callable[[IntentContext], None]
OnExitHook = Callable[[IntentContext, Optional[Exception]], None]

# Global hook registries
_on_enter_hooks: List[OnEnterHook] = []
_on_exit_hooks: List[OnExitHook] = []


def register_on_enter_hook(hook: OnEnterHook) -> None:
    """
    Register a callback to be called when entering an intent context.

    Args:
        hook: Callable that receives the IntentContext

    Example:
        def my_hook(ctx):
            print(f"Entering: {ctx.intent_name}")

        register_on_enter_hook(my_hook)
    """
    _on_enter_hooks.append(hook)


def register_on_exit_hook(hook: OnExitHook) -> None:
    """
    Register a callback to be called when exiting an intent context.

    Args:
        hook: Callable that receives IntentContext and optional exception

    Example:
        def my_hook(ctx, error):
            if error:
                print(f"Failed: {ctx.intent_name}")
            else:
                print(f"Completed: {ctx.intent_name} in {ctx.elapsed_ms}ms")

        register_on_exit_hook(my_hook)
    """
    _on_exit_hooks.append(hook)


def unregister_on_enter_hook(hook: OnEnterHook) -> bool:
    """Unregister an enter hook. Returns True if found and removed."""
    try:
        _on_enter_hooks.remove(hook)
        return True
    except ValueError:
        return False


def unregister_on_exit_hook(hook: OnExitHook) -> bool:
    """Unregister an exit hook. Returns True if found and removed."""
    try:
        _on_exit_hooks.remove(hook)
        return True
    except ValueError:
        return False


def clear_hooks() -> None:
    """Clear all registered hooks."""
    _on_enter_hooks.clear()
    _on_exit_hooks.clear()


def _invoke_enter_hooks(ctx: IntentContext) -> None:
    """Invoke all registered enter hooks."""
    for hook in _on_enter_hooks:
        try:
            hook(ctx)
        except Exception:
            pass  # Don't let hook errors break context management


def _invoke_exit_hooks(ctx: IntentContext, error: Optional[Exception] = None) -> None:
    """Invoke all registered exit hooks."""
    for hook in _on_exit_hooks:
        try:
            hook(ctx, error)
        except Exception:
            pass  # Don't let hook errors break context management


# ============================================================================
# Enhanced Context Managers with Hooks and Features
# ============================================================================

class EnhancedIntentContextManager:
    """
    Enhanced context manager with hooks, timeouts, and tags support.

    Usage:
        with EnhancedIntentContextManager(
            "my-intent",
            tags={"api", "critical"},
            timeout_seconds=30,
        ) as ctx:
            # Code here runs within the intent context
            ctx.add_tag("processed")
    """

    def __init__(
        self,
        intent_name: str,
        intent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        on_enter: Optional[OnEnterHook] = None,
        on_exit: Optional[OnExitHook] = None,
    ):
        self.intent_name = intent_name
        self.intent_id = intent_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.tags = tags or set()
        self.labels = labels or {}
        self.timeout_seconds = timeout_seconds
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.context: Optional[IntentContext] = None
        self.previous_context: Optional[IntentContext] = None
        self._exception: Optional[Exception] = None

    def __enter__(self) -> IntentContext:
        # Get parent context
        parent = get_current_intent()

        # Get session
        session = get_current_session()
        session_id = session.session_id if session else None

        # Calculate timeout
        timeout_at = None
        if self.timeout_seconds:
            timeout_at = datetime.now() + timedelta(seconds=self.timeout_seconds)

        # Create new context with enhanced features
        self.context = IntentContext(
            intent_id=self.intent_id,
            intent_name=self.intent_name,
            parent_context=parent,
            session_id=session_id,
            metadata=self.metadata,
            tags=self.tags.copy(),
            labels=self.labels.copy(),
            timeout_at=timeout_at,
        )

        # Track child in parent
        if parent:
            parent.children.append(self.intent_id)

        # Update session count
        if session:
            session.intent_count += 1

        # Save previous and set new
        self.previous_context = get_current_intent()
        set_current_intent(self.context)

        # Invoke hooks
        _invoke_enter_hooks(self.context)
        if self.on_enter:
            self.on_enter(self.context)

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Track exception
        self._exception = exc_val

        # Mark context status
        if self.context:
            if exc_val:
                self.context.fail(str(exc_val))
            elif self.context.is_timed_out:
                self.context.status = ContextStatus.TIMEOUT
                self.context.end_time = datetime.now()
            else:
                self.context.complete()

            # Invoke hooks
            _invoke_exit_hooks(self.context, exc_val)
            if self.on_exit:
                self.on_exit(self.context, exc_val)

        # Restore previous context
        set_current_intent(self.previous_context)
        return False

    async def __aenter__(self) -> IntentContext:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def intent_scope_enhanced(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[Set[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[float] = None,
    on_enter: Optional[OnEnterHook] = None,
    on_exit: Optional[OnExitHook] = None,
) -> EnhancedIntentContextManager:
    """
    Create an enhanced intent scope with hooks and features.

    Args:
        name: Name of the intent
        metadata: Optional metadata
        tags: Optional set of tags
        labels: Optional key-value labels
        timeout_seconds: Optional timeout in seconds
        on_enter: Optional callback when entering context
        on_exit: Optional callback when exiting context

    Returns:
        EnhancedIntentContextManager

    Example:
        with intent_scope_enhanced(
            "api-call",
            tags={"external", "critical"},
            timeout_seconds=30,
            on_exit=lambda ctx, err: log_latency(ctx.elapsed_ms)
        ) as ctx:
            response = call_api()
            ctx.set_label("status", str(response.status_code))
    """
    return EnhancedIntentContextManager(
        name,
        metadata=metadata,
        tags=tags,
        labels=labels,
        timeout_seconds=timeout_seconds,
        on_enter=on_enter,
        on_exit=on_exit,
    )


# ============================================================================
# Environment Variable Propagation
# ============================================================================

INTENT_CONTEXT_ENV_VAR = "INTENTLOG_CONTEXT"


def propagate_context_to_env() -> None:
    """
    Propagate current context to environment variable.

    Useful for subprocess spawning where context should be inherited.
    """
    ctx = get_current_intent()
    if ctx:
        os.environ[INTENT_CONTEXT_ENV_VAR] = ctx.serialize()
    elif INTENT_CONTEXT_ENV_VAR in os.environ:
        del os.environ[INTENT_CONTEXT_ENV_VAR]


def restore_context_from_env() -> Optional[IntentContext]:
    """
    Restore context from environment variable.

    Call this at subprocess startup to inherit parent context.

    Returns:
        IntentContext if found in environment, None otherwise
    """
    encoded = os.environ.get(INTENT_CONTEXT_ENV_VAR)
    if encoded:
        try:
            ctx = IntentContext.deserialize(encoded)
            set_current_intent(ctx)
            return ctx
        except Exception:
            pass
    return None


# ============================================================================
# Context Query Functions
# ============================================================================

def get_all_tags() -> Set[str]:
    """Get all tags from the current intent chain."""
    tags: Set[str] = set()
    for ctx in get_intent_chain():
        tags.update(ctx.tags)
    return tags


def get_all_labels() -> Dict[str, str]:
    """
    Get all labels from the current intent chain.

    Labels from deeper contexts override parent labels.
    """
    labels: Dict[str, str] = {}
    for ctx in get_intent_chain():
        labels.update(ctx.labels)
    return labels


def has_tag_in_chain(tag: str) -> bool:
    """Check if any context in the chain has the specified tag."""
    return tag in get_all_tags()


def get_root_context() -> Optional[IntentContext]:
    """Get the root (topmost) context in the current chain."""
    chain = get_intent_chain()
    return chain[0] if chain else None


def get_trace_id() -> Optional[str]:
    """Get the current trace ID if in a context."""
    ctx = get_current_intent()
    return ctx.trace_id if ctx else None


def get_span_id() -> Optional[str]:
    """Get the current span ID if in a context."""
    ctx = get_current_intent()
    return ctx.span_id if ctx else None


# ============================================================================
# Context Decorators for Tagging
# ============================================================================

def with_tags(*tags: str):
    """
    Decorator to add tags to the current context.

    Usage:
        @with_tags("api", "external")
        def call_external_api():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ctx = get_current_intent()
            if ctx:
                ctx.add_tags(*tags)
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def with_labels(**labels: str):
    """
    Decorator to add labels to the current context.

    Usage:
        @with_labels(service="payment", version="v2")
        def process_payment():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ctx = get_current_intent()
            if ctx:
                for key, value in labels.items():
                    ctx.set_label(key, value)
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
