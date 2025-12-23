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
"""

import contextvars
import threading
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .merkle import ChainedIntent


# Context variable for async-safe intent tracking
_current_intent: contextvars.ContextVar[Optional["IntentContext"]] = contextvars.ContextVar(
    "current_intent", default=None
)

# Thread-local storage for sync code fallback
_thread_local = threading.local()


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
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return (datetime.now() - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Export context to dictionary"""
        return {
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "start_time": self.start_time.isoformat(),
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "depth": self.depth,
            "metadata": self.metadata,
        }


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
