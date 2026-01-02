"""
Structured Logging for IntentLog

Provides a consistent logging framework with:
- Structured JSON output for production
- Human-readable console output for development
- Context propagation (intent IDs, session IDs, trace IDs)
- Performance timing helpers
- Log level configuration
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# Default logger name
LOGGER_NAME = "intentlog"

# Environment variable for log level
LOG_LEVEL_ENV = "INTENTLOG_LOG_LEVEL"
LOG_FORMAT_ENV = "INTENTLOG_LOG_FORMAT"
LOG_FILE_ENV = "INTENTLOG_LOG_FILE"


class LogFormat(Enum):
    """Output format for logs."""
    CONSOLE = "console"     # Human-readable colored output
    JSON = "json"           # Structured JSON (one line per entry)
    PRETTY_JSON = "pretty"  # Indented JSON (for debugging)


class LogLevel(Enum):
    """Log levels matching Python's logging module."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogContext:
    """
    Context information attached to log entries.

    Provides structured fields for tracing and correlation.
    """
    intent_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    branch: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary, excluding None values."""
        result = {}
        if self.intent_id:
            result["intent_id"] = self.intent_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.branch:
            result["branch"] = self.branch
        if self.operation:
            result["operation"] = self.operation
        if self.user_id:
            result["user_id"] = self.user_id
        if self.extra:
            result.update(self.extra)
        return result

    def merge(self, other: "LogContext") -> "LogContext":
        """Merge with another context, preferring non-None values from other."""
        return LogContext(
            intent_id=other.intent_id or self.intent_id,
            session_id=other.session_id or self.session_id,
            trace_id=other.trace_id or self.trace_id,
            span_id=other.span_id or self.span_id,
            branch=other.branch or self.branch,
            operation=other.operation or self.operation,
            user_id=other.user_id or self.user_id,
            extra={**self.extra, **other.extra},
        )


# Thread-local context storage
import threading
_context_local = threading.local()


def get_current_log_context() -> LogContext:
    """Get the current logging context."""
    return getattr(_context_local, "context", LogContext())


def set_current_log_context(context: LogContext) -> None:
    """Set the current logging context."""
    _context_local.context = context


@contextmanager
def log_context(**kwargs):
    """
    Context manager for adding logging context.

    Usage:
        with log_context(intent_id="abc123", operation="commit"):
            logger.info("Processing intent")
            # Log entries will include intent_id and operation
    """
    old_context = get_current_log_context()
    new_context = old_context.merge(LogContext(**kwargs))
    set_current_log_context(new_context)
    try:
        yield new_context
    finally:
        set_current_log_context(old_context)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs one JSON object per line with:
    - timestamp (ISO 8601)
    - level
    - logger name
    - message
    - context fields
    - exception info (if any)
    """

    def __init__(self, pretty: bool = False):
        super().__init__()
        self.pretty = pretty

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context
        context = get_current_log_context()
        context_dict = context.to_dict()
        if context_dict:
            entry["context"] = context_dict

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            entry.update(record.extra_fields)

        # Add source location for debug
        if record.levelno <= logging.DEBUG:
            entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        if self.pretty:
            return json.dumps(entry, indent=2, default=str)
        return json.dumps(entry, separators=(",", ":"), default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Includes colors for different log levels (when terminal supports it).
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Get level with optional color
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level = f"{self.COLORS[level]}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"

        # Base message
        message = record.getMessage()

        # Add context summary if present
        context = get_current_log_context()
        context_parts = []
        if context.intent_id:
            context_parts.append(f"intent={context.intent_id[:8]}")
        if context.operation:
            context_parts.append(f"op={context.operation}")
        if context.session_id:
            context_parts.append(f"session={context.session_id[:8]}")

        context_str = ""
        if context_parts:
            context_str = f" [{', '.join(context_parts)}]"

        # Format output
        output = f"{timestamp} {level} {message}{context_str}"

        # Add exception if present
        if record.exc_info:
            output += "\n" + self.formatException(record.exc_info)

        return output


class IntentLogLogger:
    """
    Wrapper around Python's logging.Logger with structured logging support.

    Provides convenience methods and automatic context propagation.
    """

    def __init__(self, name: str = LOGGER_NAME):
        self._logger = logging.getLogger(name)
        self._configured = False

    def configure(
        self,
        level: Union[str, LogLevel] = LogLevel.INFO,
        format: Union[str, LogFormat] = LogFormat.CONSOLE,
        log_file: Optional[Path] = None,
        propagate: bool = False,
    ) -> None:
        """
        Configure the logger.

        Args:
            level: Minimum log level
            format: Output format (console, json, pretty)
            log_file: Optional file to write logs to
            propagate: Whether to propagate to parent loggers
        """
        # Parse level
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        self._logger.setLevel(level.value)
        self._logger.propagate = propagate

        # Clear existing handlers
        self._logger.handlers.clear()

        # Parse format
        if isinstance(format, str):
            format = LogFormat(format.lower())

        # Create formatter
        if format == LogFormat.JSON:
            formatter = StructuredFormatter(pretty=False)
        elif format == LogFormat.PRETTY_JSON:
            formatter = StructuredFormatter(pretty=True)
        else:
            formatter = ConsoleFormatter()

        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            # Always use JSON for file logging
            file_handler.setFormatter(StructuredFormatter(pretty=False))
            self._logger.addHandler(file_handler)

        self._configured = True

    def _ensure_configured(self) -> None:
        """Ensure logger is configured with defaults."""
        if not self._configured:
            # Check environment variables
            level = os.environ.get(LOG_LEVEL_ENV, "INFO")
            format_str = os.environ.get(LOG_FORMAT_ENV, "console")
            log_file = os.environ.get(LOG_FILE_ENV)

            self.configure(
                level=level,
                format=format_str,
                log_file=Path(log_file) if log_file else None,
            )

    def _log(
        self,
        level: int,
        msg: str,
        *args,
        exc_info: bool = False,
        **kwargs,
    ) -> None:
        """Internal logging method with extra fields support."""
        self._ensure_configured()

        # Create log record with extra fields
        extra = {"extra_fields": kwargs} if kwargs else {}
        self._logger.log(level, msg, *args, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log at ERROR level with exception info."""
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)

    @contextmanager
    def timed(self, operation: str, level: int = logging.DEBUG):
        """
        Context manager for timing operations.

        Usage:
            with logger.timed("database_query"):
                result = db.query(...)
        """
        start = time.perf_counter()
        self._log(level, f"Starting: {operation}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._log(level, f"Completed: {operation}", duration_ms=round(elapsed * 1000, 2))


# Global logger instance
_logger: Optional[IntentLogLogger] = None


def get_logger(name: str = LOGGER_NAME) -> IntentLogLogger:
    """
    Get the IntentLog logger.

    Returns the global logger instance, creating it if necessary.
    """
    global _logger
    if _logger is None:
        _logger = IntentLogLogger(name)
    return _logger


def configure_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    format: Union[str, LogFormat] = LogFormat.CONSOLE,
    log_file: Optional[Path] = None,
) -> IntentLogLogger:
    """
    Configure the global IntentLog logger.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format (console, json, pretty)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance

    Example:
        from intentlog.logging import configure_logging, LogLevel, LogFormat

        # Development
        configure_logging(level=LogLevel.DEBUG, format=LogFormat.CONSOLE)

        # Production
        configure_logging(
            level=LogLevel.INFO,
            format=LogFormat.JSON,
            log_file=Path("/var/log/intentlog.log")
        )
    """
    logger = get_logger()
    logger.configure(level=level, format=format, log_file=log_file)
    return logger


def log_function_call(
    level: int = logging.DEBUG,
    include_args: bool = True,
    include_result: bool = False,
):
    """
    Decorator for logging function calls.

    Args:
        level: Log level for the messages
        include_args: Include function arguments in log
        include_result: Include return value in log

    Usage:
        @log_function_call()
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            func_name = func.__qualname__

            # Log entry
            if include_args:
                logger._log(level, f"Calling {func_name}", args=args, kwargs=kwargs)
            else:
                logger._log(level, f"Calling {func_name}")

            # Execute function
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                # Log success
                if include_result:
                    logger._log(level, f"Completed {func_name}",
                               duration_ms=round(elapsed * 1000, 2),
                               result=repr(result)[:100])
                else:
                    logger._log(level, f"Completed {func_name}",
                               duration_ms=round(elapsed * 1000, 2))
                return result

            except Exception as e:
                elapsed = time.perf_counter() - start
                logger._log(logging.ERROR, f"Failed {func_name}: {e}",
                           duration_ms=round(elapsed * 1000, 2),
                           exc_info=True)
                raise

        return wrapper
    return decorator


# Convenience exports
__all__ = [
    "LogFormat",
    "LogLevel",
    "LogContext",
    "IntentLogLogger",
    "get_logger",
    "configure_logging",
    "log_context",
    "log_function_call",
    "get_current_log_context",
    "set_current_log_context",
]
