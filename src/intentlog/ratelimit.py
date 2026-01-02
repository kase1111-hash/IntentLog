"""
Rate Limiting and Retry Logic for IntentLog

Provides:
- Token bucket rate limiting for API calls
- Exponential backoff retry with jitter
- Circuit breaker pattern for fault tolerance
- Decorators for easy application to functions
"""

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, Optional, Type, Tuple, TypeVar, Union, List
from datetime import datetime, timedelta

from .logging import get_logger

# Type variable for generic function decoration
F = TypeVar('F', bound=Callable)


class RetryStrategy(Enum):
    """Retry strategies for handling failures."""
    EXPONENTIAL = "exponential"     # Exponential backoff with jitter
    LINEAR = "linear"               # Linear backoff
    CONSTANT = "constant"           # Fixed delay between retries


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0           # Initial delay in seconds
    max_delay: float = 60.0           # Maximum delay cap
    exponential_base: float = 2.0     # Base for exponential backoff
    jitter: float = 0.1               # Random jitter factor (0-1)
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL

    # Exceptions that should trigger a retry
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )

    # Exceptions that should NOT be retried (even if in retryable_exceptions)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=tuple
    )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0     # Max requests per second
    requests_per_minute: float = 100.0    # Max requests per minute
    burst_size: int = 10                  # Max burst size

    # Token bucket parameters (derived from above)
    @property
    def tokens_per_second(self) -> float:
        return min(self.requests_per_second, self.requests_per_minute / 60.0)

    @property
    def bucket_capacity(self) -> int:
        return self.burst_size


class CircuitState(Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout: float = 30.0               # Seconds before trying half-open

    # Exceptions that count as failures
    failure_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and cannot wait."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class TokenBucket:
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity, then limits to steady rate.
    Thread-safe implementation.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._tokens = float(config.bucket_capacity)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.config.bucket_capacity,
            self._tokens + elapsed * self.config.tokens_per_second
        )
        self._last_update = now

    def acquire(self, tokens: int = 1, block: bool = True, timeout: float = 30.0) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            block: If True, wait for tokens; if False, return immediately
            timeout: Maximum time to wait for tokens

        Returns:
            True if tokens acquired, False if not available and not blocking

        Raises:
            RateLimitExceeded: If blocking and timeout exceeded
        """
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                if not block:
                    return False

                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self.config.tokens_per_second

            # Check timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RateLimitExceeded(
                    f"Rate limit exceeded, need {wait_time:.2f}s",
                    retry_after=wait_time
                )

            # Wait for tokens (with small sleep intervals for responsiveness)
            time.sleep(min(wait_time, remaining, 0.1))

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascading failures by failing fast when a service is down.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CircuitState:
        """Get state (must hold lock)."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._last_failure_time:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
        return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        state = self.state
        return state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = 0

    def record_failure(self, exception: Exception) -> None:
        """Record a failed execution."""
        with self._lock:
            # Check if this exception counts as a failure
            if not isinstance(exception, self.config.failure_exceptions):
                return

            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN

    def get_retry_after(self) -> Optional[float]:
        """Get time until circuit might close."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return None
            if self._last_failure_time is None:
                return None
            elapsed = time.monotonic() - self._last_failure_time
            return max(0, self.config.timeout - elapsed)


class RateLimiter:
    """
    Combined rate limiter with retry logic and circuit breaker.

    Provides a complete solution for API rate limiting:
    - Token bucket for steady rate limiting
    - Exponential backoff for retries
    - Circuit breaker for fault tolerance
    """

    def __init__(
        self,
        rate_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.rate_config = rate_config or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config

        self._bucket = TokenBucket(self.rate_config)
        self._circuit = CircuitBreaker(self.circuit_config) if circuit_config else None
        self._logger = get_logger()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        config = self.retry_config

        if config.strategy == RetryStrategy.CONSTANT:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * (attempt + 1)
        else:  # EXPONENTIAL
            delay = config.base_delay * (config.exponential_base ** attempt)

        # Apply cap
        delay = min(delay, config.max_delay)

        # Apply jitter
        if config.jitter > 0:
            jitter_range = delay * config.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if an exception should be retried."""
        config = self.retry_config

        # Check attempt count
        if attempt >= config.max_retries:
            return False

        # Check non-retryable exceptions first
        if isinstance(exception, config.non_retryable_exceptions):
            return False

        # Check retryable exceptions
        return isinstance(exception, config.retryable_exceptions)

    def execute(self, func: Callable[[], F], operation: str = "operation") -> F:
        """
        Execute a function with rate limiting and retry logic.

        Args:
            func: Zero-argument callable to execute
            operation: Name of operation for logging

        Returns:
            Result of the function

        Raises:
            The last exception if all retries fail
        """
        logger = self._logger

        # Check circuit breaker
        if self._circuit and not self._circuit.can_execute():
            retry_after = self._circuit.get_retry_after()
            logger.warning(
                "Circuit breaker open",
                operation=operation,
                retry_after=retry_after
            )
            raise CircuitOpenError(
                f"Circuit breaker open for {operation}",
                retry_after=retry_after
            )

        last_exception: Optional[Exception] = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Acquire rate limit token
                self._bucket.acquire(tokens=1, block=True)

                # Execute function
                result = func()

                # Record success
                if self._circuit:
                    self._circuit.record_success()

                if attempt > 0:
                    logger.info(
                        "Operation succeeded after retry",
                        operation=operation,
                        attempt=attempt + 1
                    )

                return result

            except Exception as e:
                last_exception = e

                # Record failure
                if self._circuit:
                    self._circuit.record_failure(e)

                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.error(
                        "Operation failed (not retrying)",
                        operation=operation,
                        attempt=attempt + 1,
                        error=str(e),
                        exc_info=True
                    )
                    raise

                # Calculate delay
                delay = self.calculate_delay(attempt)

                # Check for retry_after hint from exception
                if hasattr(e, 'retry_after') and e.retry_after:
                    delay = max(delay, e.retry_after)

                logger.warning(
                    "Operation failed, retrying",
                    operation=operation,
                    attempt=attempt + 1,
                    max_attempts=self.retry_config.max_retries + 1,
                    delay=f"{delay:.2f}s",
                    error=str(e)
                )

                time.sleep(delay)

        # All retries exhausted
        logger.error(
            "Operation failed after all retries",
            operation=operation,
            attempts=self.retry_config.max_retries + 1
        )
        raise last_exception


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
):
    """
    Decorator for adding retry logic to a function.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap
        exponential_base: Base for exponential backoff
        jitter: Random jitter factor (0-1)
        strategy: Retry strategy to use
        retryable_exceptions: Exceptions that should trigger retry
        non_retryable_exceptions: Exceptions that should not be retried

    Usage:
        @with_retry(max_retries=3, base_delay=1.0)
        def call_api():
            return requests.get(url)
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        strategy=strategy,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(func: F) -> F:
        limiter = RateLimiter(retry_config=config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return limiter.execute(
                lambda: func(*args, **kwargs),
                operation=func.__qualname__
            )

        return wrapper

    return decorator


def with_rate_limit(
    requests_per_second: float = 10.0,
    requests_per_minute: float = 100.0,
    burst_size: int = 10,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Decorator for adding rate limiting with retry to a function.

    Args:
        requests_per_second: Maximum requests per second
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size
        max_retries: Maximum retry attempts
        base_delay: Initial retry delay

    Usage:
        @with_rate_limit(requests_per_second=5.0)
        def call_api():
            return requests.get(url)
    """
    rate_config = RateLimitConfig(
        requests_per_second=requests_per_second,
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
    )

    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
    )

    def decorator(func: F) -> F:
        limiter = RateLimiter(rate_config=rate_config, retry_config=retry_config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return limiter.execute(
                lambda: func(*args, **kwargs),
                operation=func.__qualname__
            )

        return wrapper

    return decorator


# Global rate limiter for LLM calls
_llm_rate_limiter: Optional[RateLimiter] = None


def get_llm_rate_limiter() -> RateLimiter:
    """Get the global LLM rate limiter."""
    global _llm_rate_limiter
    if _llm_rate_limiter is None:
        # Default configuration for LLM APIs
        # Conservative limits to avoid hitting rate limits
        _llm_rate_limiter = RateLimiter(
            rate_config=RateLimitConfig(
                requests_per_second=5.0,
                requests_per_minute=60.0,
                burst_size=10,
            ),
            retry_config=RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=0.1,
            ),
        )
    return _llm_rate_limiter


def configure_llm_rate_limit(
    requests_per_second: float = 5.0,
    requests_per_minute: float = 60.0,
    max_retries: int = 3,
    enable_circuit_breaker: bool = False,
) -> RateLimiter:
    """
    Configure the global LLM rate limiter.

    Args:
        requests_per_second: Max LLM requests per second
        requests_per_minute: Max LLM requests per minute
        max_retries: Max retry attempts for failed calls
        enable_circuit_breaker: Enable circuit breaker for fault tolerance

    Returns:
        Configured rate limiter
    """
    global _llm_rate_limiter

    circuit_config = None
    if enable_circuit_breaker:
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0,
        )

    _llm_rate_limiter = RateLimiter(
        rate_config=RateLimitConfig(
            requests_per_second=requests_per_second,
            requests_per_minute=requests_per_minute,
            burst_size=int(requests_per_second * 2),
        ),
        retry_config=RetryConfig(
            max_retries=max_retries,
            base_delay=1.0,
            max_delay=60.0,
        ),
        circuit_config=circuit_config,
    )

    return _llm_rate_limiter


# Convenience exports
__all__ = [
    "RetryStrategy",
    "RetryConfig",
    "RateLimitConfig",
    "CircuitState",
    "CircuitBreakerConfig",
    "RateLimitExceeded",
    "CircuitOpenError",
    "TokenBucket",
    "CircuitBreaker",
    "RateLimiter",
    "with_retry",
    "with_rate_limit",
    "get_llm_rate_limiter",
    "configure_llm_rate_limit",
]
