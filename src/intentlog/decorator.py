"""
Intent Logger Decorator for IntentLog

Provides the @intent_logger decorator for automatic nested intent tracing.
Wraps functions to automatically log entry/exit as intents with proper
parent-child relationships.

Features:
- Automatic function entry/exit logging
- Nested call tracking with parent relationships
- Latency measurement
- Exception tracking
- Support for sync and async functions
- Conditional logging levels
"""

import functools
import inspect
import time
import uuid
from datetime import datetime
from typing import (
    Optional,
    Dict,
    Any,
    Callable,
    TypeVar,
    Union,
    List,
    overload,
)
from enum import Enum

from .context import (
    IntentContext,
    IntentContextManager,
    get_current_intent,
    get_current_session,
    get_intent_chain,
)
from .core import Intent
from .storage import IntentLogStorage


F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(Enum):
    """Logging level for conditional intent logging"""
    DEBUG = 0      # Log everything
    INFO = 1       # Standard logging
    IMPORTANT = 2  # Only significant intents
    CRITICAL = 3   # Only critical intents
    OFF = 4        # No logging


# Global logging level
_log_level: LogLevel = LogLevel.INFO


def set_log_level(level: LogLevel) -> None:
    """Set the global logging level"""
    global _log_level
    _log_level = level


def get_log_level() -> LogLevel:
    """Get the current logging level"""
    return _log_level


def should_log(level: LogLevel) -> bool:
    """Check if a given level should be logged"""
    return level.value >= _log_level.value


class IntentLoggerConfig:
    """Configuration for @intent_logger decorator"""

    def __init__(
        self,
        name: Optional[str] = None,
        level: LogLevel = LogLevel.INFO,
        log_args: bool = False,
        log_result: bool = False,
        log_exceptions: bool = True,
        include_traceback: bool = False,
        persist: bool = False,
        sign: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        reasoning_template: Optional[str] = None,
    ):
        """
        Configure intent logger behavior.

        Args:
            name: Custom intent name (default: function name)
            level: Logging level for this intent
            log_args: Include function arguments in metadata
            log_result: Include return value in metadata
            log_exceptions: Log exceptions as failures
            include_traceback: Include traceback in exception metadata
            persist: Persist to storage (vs in-memory only)
            sign: Sign persisted intents
            metadata: Additional metadata to include
            reasoning_template: Template for reasoning (uses {func}, {args}, etc.)
        """
        self.name = name
        self.level = level
        self.log_args = log_args
        self.log_result = log_result
        self.log_exceptions = log_exceptions
        self.include_traceback = include_traceback
        self.persist = persist
        self.sign = sign
        self.metadata = metadata or {}
        self.reasoning_template = reasoning_template


# Storage for non-persisted intents (in-memory log)
_intent_log: List[Dict[str, Any]] = []


def get_intent_log() -> List[Dict[str, Any]]:
    """Get the in-memory intent log"""
    return _intent_log.copy()


def clear_intent_log() -> None:
    """Clear the in-memory intent log"""
    global _intent_log
    _intent_log = []


def _log_intent(
    intent_name: str,
    reasoning: str,
    context: IntentContext,
    status: str,
    config: IntentLoggerConfig,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Log an intent entry.

    Args:
        intent_name: Name of the intent
        reasoning: Reasoning text
        context: Current intent context
        status: Status (started, completed, failed)
        config: Logger configuration
        extra_metadata: Additional metadata

    Returns:
        Intent ID if persisted, None otherwise
    """
    metadata = {
        **config.metadata,
        **(extra_metadata or {}),
        "status": status,
        "depth": context.depth,
        "latency_ms": context.elapsed_ms if status != "started" else 0,
    }

    if context.session_id:
        metadata["session_id"] = context.session_id

    if context.parent_id:
        metadata["parent_intent_id"] = context.parent_id

    entry = {
        "intent_id": context.intent_id,
        "intent_name": intent_name,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "metadata": metadata,
    }

    # Add to in-memory log
    _intent_log.append(entry)

    # Persist if configured
    if config.persist and status == "completed":
        try:
            storage = IntentLogStorage()
            if storage.is_initialized():
                storage.add_chained_intent(
                    name=intent_name,
                    reasoning=reasoning,
                    metadata=metadata,
                    parent_id=context.parent_id,
                    sign=config.sign,
                )
                return context.intent_id
        except Exception:
            pass  # Silent failure for persistence

    return context.intent_id if status == "completed" else None


def _format_args(args: tuple, kwargs: dict) -> str:
    """Format function arguments for logging"""
    parts = []
    for i, arg in enumerate(args):
        parts.append(f"arg{i}={_safe_repr(arg)}")
    for key, value in kwargs.items():
        parts.append(f"{key}={_safe_repr(value)}")
    return ", ".join(parts)


def _safe_repr(obj: Any, max_len: int = 100) -> str:
    """Safely repr an object with length limit"""
    try:
        r = repr(obj)
        if len(r) > max_len:
            r = r[:max_len-3] + "..."
        return r
    except Exception:
        return "<unrepresentable>"


def _generate_reasoning(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: IntentLoggerConfig,
    result: Any = None,
    error: Optional[Exception] = None,
) -> str:
    """Generate reasoning text for the intent"""
    if config.reasoning_template:
        try:
            return config.reasoning_template.format(
                func=func.__name__,
                args=_format_args(args, kwargs),
                result=_safe_repr(result) if result is not None else "None",
                error=str(error) if error else "",
            )
        except Exception:
            pass

    # Default reasoning
    if error:
        return f"Function {func.__name__} raised {type(error).__name__}: {error}"
    elif result is not None:
        return f"Function {func.__name__} completed successfully"
    else:
        return f"Executing function {func.__name__}"


@overload
def intent_logger(func: F) -> F:
    ...


@overload
def intent_logger(
    *,
    name: Optional[str] = None,
    level: LogLevel = LogLevel.INFO,
    log_args: bool = False,
    log_result: bool = False,
    log_exceptions: bool = True,
    persist: bool = False,
    sign: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    reasoning_template: Optional[str] = None,
) -> Callable[[F], F]:
    ...


def intent_logger(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    level: LogLevel = LogLevel.INFO,
    log_args: bool = False,
    log_result: bool = False,
    log_exceptions: bool = True,
    include_traceback: bool = False,
    persist: bool = False,
    sign: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    reasoning_template: Optional[str] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for automatic intent logging.

    Logs function entry and exit as intents, with proper parent-child
    relationships for nested calls.

    Can be used with or without arguments:
        @intent_logger
        def my_function():
            pass

        @intent_logger(persist=True, log_args=True)
        def my_function():
            pass

    Args:
        func: Function to wrap (when used without parentheses)
        name: Custom intent name (default: function name)
        level: Logging level (DEBUG, INFO, IMPORTANT, CRITICAL)
        log_args: Include function arguments in metadata
        log_result: Include return value in metadata
        log_exceptions: Log exceptions as failures
        include_traceback: Include traceback in exception metadata
        persist: Persist intents to storage
        sign: Sign persisted intents
        metadata: Additional metadata to include
        reasoning_template: Template for reasoning text

    Returns:
        Decorated function or decorator
    """
    config = IntentLoggerConfig(
        name=name,
        level=level,
        log_args=log_args,
        log_result=log_result,
        log_exceptions=log_exceptions,
        include_traceback=include_traceback,
        persist=persist,
        sign=sign,
        metadata=metadata,
        reasoning_template=reasoning_template,
    )

    def decorator(fn: F) -> F:
        intent_name = config.name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                if not should_log(config.level):
                    return await fn(*args, **kwargs)

                extra_meta = {}
                if config.log_args:
                    extra_meta["args"] = _format_args(args, kwargs)

                with IntentContextManager(intent_name, metadata=config.metadata) as ctx:
                    reasoning = _generate_reasoning(fn, args, kwargs, config)
                    _log_intent(intent_name, reasoning, ctx, "started", config, extra_meta)

                    try:
                        result = await fn(*args, **kwargs)

                        if config.log_result:
                            extra_meta["result"] = _safe_repr(result)

                        reasoning = _generate_reasoning(fn, args, kwargs, config, result=result)
                        _log_intent(intent_name, reasoning, ctx, "completed", config, extra_meta)

                        return result

                    except Exception as e:
                        if config.log_exceptions:
                            extra_meta["error"] = str(e)
                            extra_meta["error_type"] = type(e).__name__
                            if config.include_traceback:
                                import traceback
                                extra_meta["traceback"] = traceback.format_exc()

                            reasoning = _generate_reasoning(fn, args, kwargs, config, error=e)
                            _log_intent(intent_name, reasoning, ctx, "failed", config, extra_meta)

                        raise

            return async_wrapper  # type: ignore
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                if not should_log(config.level):
                    return fn(*args, **kwargs)

                extra_meta = {}
                if config.log_args:
                    extra_meta["args"] = _format_args(args, kwargs)

                with IntentContextManager(intent_name, metadata=config.metadata) as ctx:
                    reasoning = _generate_reasoning(fn, args, kwargs, config)
                    _log_intent(intent_name, reasoning, ctx, "started", config, extra_meta)

                    try:
                        result = fn(*args, **kwargs)

                        if config.log_result:
                            extra_meta["result"] = _safe_repr(result)

                        reasoning = _generate_reasoning(fn, args, kwargs, config, result=result)
                        _log_intent(intent_name, reasoning, ctx, "completed", config, extra_meta)

                        return result

                    except Exception as e:
                        if config.log_exceptions:
                            extra_meta["error"] = str(e)
                            extra_meta["error_type"] = type(e).__name__
                            if config.include_traceback:
                                import traceback
                                extra_meta["traceback"] = traceback.format_exc()

                            reasoning = _generate_reasoning(fn, args, kwargs, config, error=e)
                            _log_intent(intent_name, reasoning, ctx, "failed", config, extra_meta)

                        raise

            return sync_wrapper  # type: ignore

    # Handle both @intent_logger and @intent_logger(...)
    if func is not None:
        return decorator(func)
    return decorator


def intent_logger_class(
    cls: Optional[type] = None,
    *,
    methods: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **kwargs,
) -> type:
    """
    Class decorator to add intent logging to all methods.

    Args:
        cls: Class to decorate
        methods: List of method names to log (None = all public methods)
        exclude: List of method names to exclude
        **kwargs: Arguments passed to intent_logger

    Returns:
        Decorated class

    Example:
        @intent_logger_class(persist=True, exclude=["_private"])
        class MyService:
            def process(self):
                pass
    """
    exclude = exclude or []
    exclude.extend(["__init__", "__new__", "__del__", "__repr__", "__str__"])

    def decorator(klass: type) -> type:
        for name, method in inspect.getmembers(klass, predicate=inspect.isfunction):
            # Skip excluded methods
            if name in exclude:
                continue

            # Skip private methods unless explicitly listed
            if name.startswith("_") and (methods is None or name not in methods):
                continue

            # Only process listed methods if specified
            if methods is not None and name not in methods:
                continue

            # Apply decorator
            decorated = intent_logger(**kwargs)(method)
            setattr(klass, name, decorated)

        return klass

    if cls is not None:
        return decorator(cls)
    return decorator  # type: ignore


# Convenience aliases
log_intent = intent_logger
trace = intent_logger
