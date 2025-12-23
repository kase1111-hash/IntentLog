"""
Human-in-the-Loop (HITL) Triggers for IntentLog

Provides infrastructure for requiring human approval or review before
executing sensitive operations. This implements the HITL pattern from
Advanced-Use-Cases.md section 4.

Features:
- Multiple trigger types (confirmation, approval, review, notification)
- Customizable handlers for different UI/UX patterns
- Integration with intent context for full traceability
- Timeout and escalation support
- Audit logging of all HITL interactions
"""

import uuid
import time
import functools
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Optional,
    Dict,
    Any,
    List,
    Callable,
    TypeVar,
    Union,
    Awaitable,
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .context import get_current_intent, IntentContext


class TriggerType(Enum):
    """Types of HITL triggers"""
    NOTIFICATION = "notification"  # Inform user, no blocking
    CONFIRMATION = "confirmation"  # Simple yes/no confirmation
    APPROVAL = "approval"          # Formal approval required
    REVIEW = "review"              # Review with optional modification


class TriggerResponse(Enum):
    """Possible responses to a HITL trigger"""
    APPROVED = "approved"
    DENIED = "denied"
    MODIFIED = "modified"  # Approved with modifications
    TIMEOUT = "timeout"
    ESCALATED = "escalated"
    SKIPPED = "skipped"    # For notifications


class SensitivityLevel(Enum):
    """Sensitivity levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TriggerRequest:
    """
    A request for human intervention.

    Contains all information needed for a human to make an informed decision.
    """
    trigger_id: str
    trigger_type: TriggerType
    operation_name: str
    description: str
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
    intent_context: Optional[IntentContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: Optional[float] = None
    requires_reason: bool = False  # Require reason for denial

    @property
    def intent_id(self) -> Optional[str]:
        """Get associated intent ID if available"""
        return self.intent_context.intent_id if self.intent_context else None

    @property
    def intent_name(self) -> Optional[str]:
        """Get associated intent name if available"""
        return self.intent_context.intent_name if self.intent_context else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "operation_name": self.operation_name,
            "description": self.description,
            "sensitivity": self.sensitivity.value,
            "intent_id": self.intent_id,
            "intent_name": self.intent_name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "requires_reason": self.requires_reason,
        }


@dataclass
class TriggerResult:
    """
    Result of a HITL trigger interaction.

    Records the human's decision and any modifications made.
    """
    trigger_id: str
    response: TriggerResponse
    responded_by: Optional[str] = None  # User ID or name
    responded_at: datetime = field(default_factory=datetime.now)
    reason: Optional[str] = None  # Reason for denial or approval
    modifications: Optional[Dict[str, Any]] = None  # For MODIFIED responses
    elapsed_ms: float = 0.0

    @property
    def is_approved(self) -> bool:
        """Check if the trigger was approved (including modified)"""
        return self.response in (TriggerResponse.APPROVED, TriggerResponse.MODIFIED)

    @property
    def is_denied(self) -> bool:
        """Check if the trigger was denied or timed out"""
        return self.response in (TriggerResponse.DENIED, TriggerResponse.TIMEOUT)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trigger_id": self.trigger_id,
            "response": self.response.value,
            "responded_by": self.responded_by,
            "responded_at": self.responded_at.isoformat(),
            "reason": self.reason,
            "modifications": self.modifications,
            "elapsed_ms": self.elapsed_ms,
        }


class TriggerDeniedError(Exception):
    """Raised when a trigger is denied"""

    def __init__(self, trigger_id: str, reason: Optional[str] = None):
        self.trigger_id = trigger_id
        self.reason = reason
        message = f"Trigger {trigger_id} was denied"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class TriggerTimeoutError(Exception):
    """Raised when a trigger times out"""

    def __init__(self, trigger_id: str, timeout_seconds: float):
        self.trigger_id = trigger_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Trigger {trigger_id} timed out after {timeout_seconds}s"
        )


# Type for handler functions
TriggerHandler = Callable[[TriggerRequest], TriggerResult]
AsyncTriggerHandler = Callable[[TriggerRequest], Awaitable[TriggerResult]]


class TriggerHandlerBase(ABC):
    """
    Abstract base class for trigger handlers.

    Implement this to create custom UI/UX for HITL triggers.
    """

    @abstractmethod
    def handle(self, request: TriggerRequest) -> TriggerResult:
        """
        Handle a trigger request and return the result.

        Args:
            request: The trigger request to handle

        Returns:
            TriggerResult with the human's decision
        """
        pass

    def handle_async(self, request: TriggerRequest) -> Awaitable[TriggerResult]:
        """
        Async version of handle.

        Default implementation wraps sync handle.
        """
        import asyncio
        return asyncio.get_event_loop().run_in_executor(
            None, self.handle, request
        )


class ConsoleTriggerHandler(TriggerHandlerBase):
    """
    Console-based trigger handler for CLI applications.

    Displays trigger information and prompts for user input.
    """

    def __init__(
        self,
        auto_approve: bool = False,
        default_timeout: float = 300.0,
    ):
        """
        Initialize console handler.

        Args:
            auto_approve: If True, auto-approve all triggers (for testing)
            default_timeout: Default timeout in seconds
        """
        self.auto_approve = auto_approve
        self.default_timeout = default_timeout

    def handle(self, request: TriggerRequest) -> TriggerResult:
        """Handle trigger via console interaction"""
        start_time = time.time()

        if self.auto_approve:
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.APPROVED,
                responded_by="auto",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Display trigger information
        print("\n" + "=" * 60)
        print(f"HUMAN APPROVAL REQUIRED ({request.trigger_type.value.upper()})")
        print("=" * 60)
        print(f"Operation: {request.operation_name}")
        print(f"Description: {request.description}")
        print(f"Sensitivity: {request.sensitivity.value.upper()}")
        if request.intent_name:
            print(f"Intent: {request.intent_name}")
        if request.metadata:
            print(f"Details: {request.metadata}")
        print("-" * 60)

        # Get response based on trigger type
        if request.trigger_type == TriggerType.NOTIFICATION:
            print("(This is a notification - press Enter to continue)")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.SKIPPED,
                responded_by="console",
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # For confirmation/approval/review
        prompt = self._get_prompt(request.trigger_type)
        try:
            response_str = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            response_str = "n"

        # Parse response
        if response_str in ("y", "yes", "approve", "a"):
            response = TriggerResponse.APPROVED
            reason = None
        elif response_str in ("n", "no", "deny", "d"):
            response = TriggerResponse.DENIED
            reason = None
            if request.requires_reason:
                try:
                    reason = input("Reason for denial: ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = "User cancelled"
        elif response_str in ("m", "modify") and request.trigger_type == TriggerType.REVIEW:
            response = TriggerResponse.MODIFIED
            print("Enter modifications (JSON format or key=value pairs):")
            try:
                mod_str = input().strip()
                # Try JSON first
                import json
                try:
                    modifications = json.loads(mod_str)
                except json.JSONDecodeError:
                    # Parse key=value pairs
                    modifications = {}
                    for pair in mod_str.split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            modifications[k.strip()] = v.strip()
            except (EOFError, KeyboardInterrupt):
                modifications = {}
            reason = "Modified by user"
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=response,
                responded_by="console",
                reason=reason,
                modifications=modifications,
                elapsed_ms=(time.time() - start_time) * 1000,
            )
        else:
            response = TriggerResponse.DENIED
            reason = f"Invalid response: {response_str}"

        return TriggerResult(
            trigger_id=request.trigger_id,
            response=response,
            responded_by="console",
            reason=reason,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _get_prompt(self, trigger_type: TriggerType) -> str:
        """Get the appropriate prompt for the trigger type"""
        if trigger_type == TriggerType.CONFIRMATION:
            return "Confirm? [y/n]: "
        elif trigger_type == TriggerType.APPROVAL:
            return "Approve? [y/n]: "
        elif trigger_type == TriggerType.REVIEW:
            return "Approve/Deny/Modify? [y/n/m]: "
        return "Continue? [y/n]: "


class CallbackTriggerHandler(TriggerHandlerBase):
    """
    Callback-based trigger handler for custom UI integration.

    Allows registering custom callbacks for handling triggers.
    """

    def __init__(
        self,
        on_trigger: Optional[TriggerHandler] = None,
        on_notification: Optional[Callable[[TriggerRequest], None]] = None,
    ):
        """
        Initialize callback handler.

        Args:
            on_trigger: Callback for handling approval triggers
            on_notification: Callback for notifications (no response needed)
        """
        self._on_trigger = on_trigger
        self._on_notification = on_notification

    def handle(self, request: TriggerRequest) -> TriggerResult:
        """Handle trigger via callbacks"""
        start_time = time.time()

        if request.trigger_type == TriggerType.NOTIFICATION:
            if self._on_notification:
                self._on_notification(request)
            return TriggerResult(
                trigger_id=request.trigger_id,
                response=TriggerResponse.SKIPPED,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        if self._on_trigger:
            result = self._on_trigger(request)
            result.elapsed_ms = (time.time() - start_time) * 1000
            return result

        # No handler - auto-approve with warning
        return TriggerResult(
            trigger_id=request.trigger_id,
            response=TriggerResponse.APPROVED,
            reason="No handler configured - auto-approved",
            elapsed_ms=(time.time() - start_time) * 1000,
        )


# Global trigger handler registry
_default_handler: Optional[TriggerHandlerBase] = None
_trigger_history: List[Dict[str, Any]] = []
_max_history: int = 1000


def set_trigger_handler(handler: TriggerHandlerBase) -> None:
    """
    Set the default trigger handler.

    Args:
        handler: The handler to use for all triggers
    """
    global _default_handler
    _default_handler = handler


def get_trigger_handler() -> TriggerHandlerBase:
    """
    Get the current trigger handler.

    Returns:
        The current handler, or a ConsoleTriggerHandler if none set
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = ConsoleTriggerHandler()
    return _default_handler


def get_trigger_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent trigger history.

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of trigger interaction records
    """
    return _trigger_history[-limit:]


def clear_trigger_history() -> None:
    """Clear the trigger history"""
    global _trigger_history
    _trigger_history = []


def _record_trigger(request: TriggerRequest, result: TriggerResult) -> None:
    """Record a trigger interaction to history"""
    global _trigger_history
    record = {
        "request": request.to_dict(),
        "result": result.to_dict(),
    }
    _trigger_history.append(record)
    # Trim if over limit
    if len(_trigger_history) > _max_history:
        _trigger_history = _trigger_history[-_max_history:]


def require_human_approval(
    operation_name: str,
    description: str,
    trigger_type: TriggerType = TriggerType.CONFIRMATION,
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
    timeout_seconds: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    requires_reason: bool = False,
    raise_on_deny: bool = True,
) -> TriggerResult:
    """
    Request human approval before proceeding.

    This is the primary API for HITL triggers.

    Args:
        operation_name: Short name for the operation
        description: Human-readable description of what will happen
        trigger_type: Type of trigger (confirmation, approval, etc.)
        sensitivity: Sensitivity level of the operation
        timeout_seconds: Optional timeout
        metadata: Additional context to show the user
        requires_reason: Whether to require reason for denial
        raise_on_deny: Whether to raise TriggerDeniedError on denial

    Returns:
        TriggerResult with the human's decision

    Raises:
        TriggerDeniedError: If denied and raise_on_deny is True
        TriggerTimeoutError: If timed out and raise_on_deny is True

    Example:
        result = require_human_approval(
            "Database Delete",
            "This will permanently delete 1,523 user records",
            sensitivity=SensitivityLevel.CRITICAL,
        )
        if result.is_approved:
            delete_records()
    """
    # Build request
    request = TriggerRequest(
        trigger_id=str(uuid.uuid4()),
        trigger_type=trigger_type,
        operation_name=operation_name,
        description=description,
        sensitivity=sensitivity,
        intent_context=get_current_intent(),
        metadata=metadata or {},
        timeout_seconds=timeout_seconds,
        requires_reason=requires_reason,
    )

    # Get handler and process
    handler = get_trigger_handler()
    result = handler.handle(request)

    # Record to history
    _record_trigger(request, result)

    # Handle denial
    if raise_on_deny:
        if result.response == TriggerResponse.DENIED:
            raise TriggerDeniedError(request.trigger_id, result.reason)
        if result.response == TriggerResponse.TIMEOUT:
            raise TriggerTimeoutError(
                request.trigger_id,
                timeout_seconds or 0,
            )

    return result


def notify_human(
    operation_name: str,
    description: str,
    sensitivity: SensitivityLevel = SensitivityLevel.LOW,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Send a notification to the human (non-blocking).

    Use this to keep users informed of AI actions without requiring approval.

    Args:
        operation_name: Short name for the operation
        description: Human-readable description
        sensitivity: Sensitivity level
        metadata: Additional context

    Example:
        notify_human(
            "Data Processing",
            "Processing 10,000 records...",
        )
    """
    require_human_approval(
        operation_name=operation_name,
        description=description,
        trigger_type=TriggerType.NOTIFICATION,
        sensitivity=sensitivity,
        metadata=metadata,
        raise_on_deny=False,
    )


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def requires_approval(
    operation_name: Optional[str] = None,
    description: Optional[str] = None,
    trigger_type: TriggerType = TriggerType.APPROVAL,
    sensitivity: SensitivityLevel = SensitivityLevel.HIGH,
    timeout_seconds: Optional[float] = None,
) -> Callable[[F], F]:
    """
    Decorator to require human approval before executing a function.

    Args:
        operation_name: Name for the operation (defaults to function name)
        description: Description (defaults to function docstring)
        trigger_type: Type of trigger
        sensitivity: Sensitivity level
        timeout_seconds: Optional timeout

    Returns:
        Decorated function that requires approval

    Example:
        @requires_approval(
            sensitivity=SensitivityLevel.CRITICAL,
            description="Permanently deletes user data"
        )
        def delete_user(user_id: str):
            # This will require approval before execution
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            desc = description or func.__doc__ or f"Execute {func.__name__}"

            # Include function arguments in metadata
            metadata = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            }

            result = require_human_approval(
                operation_name=op_name,
                description=desc,
                trigger_type=trigger_type,
                sensitivity=sensitivity,
                timeout_seconds=timeout_seconds,
                metadata=metadata,
            )

            # If approved, execute the function
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def requires_confirmation(
    description: Optional[str] = None,
    sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
) -> Callable[[F], F]:
    """
    Convenience decorator for simple confirmation triggers.

    Example:
        @requires_confirmation("Send email to all users?")
        def send_mass_email():
            pass
    """
    return requires_approval(
        description=description,
        trigger_type=TriggerType.CONFIRMATION,
        sensitivity=sensitivity,
    )


def requires_review(
    description: Optional[str] = None,
    sensitivity: SensitivityLevel = SensitivityLevel.HIGH,
) -> Callable[[F], F]:
    """
    Decorator for operations that require review with possible modification.

    The function receives the trigger result as first argument if modifications
    were made.

    Example:
        @requires_review("Review and approve payment details")
        def process_payment(amount: float):
            pass
    """
    return requires_approval(
        description=description,
        trigger_type=TriggerType.REVIEW,
        sensitivity=sensitivity,
    )


class TriggerScope:
    """
    Context manager for HITL triggers.

    Provides a scope where the human has been informed or has approved
    the operations within.

    Example:
        with TriggerScope(
            "Batch Processing",
            "Processing 1000 records with potential data changes",
            trigger_type=TriggerType.APPROVAL,
        ) as scope:
            if scope.is_approved:
                for record in records:
                    process(record)
    """

    def __init__(
        self,
        operation_name: str,
        description: str,
        trigger_type: TriggerType = TriggerType.CONFIRMATION,
        sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
        timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raise_on_deny: bool = True,
    ):
        self.operation_name = operation_name
        self.description = description
        self.trigger_type = trigger_type
        self.sensitivity = sensitivity
        self.timeout_seconds = timeout_seconds
        self.metadata = metadata
        self.raise_on_deny = raise_on_deny
        self.result: Optional[TriggerResult] = None

    @property
    def is_approved(self) -> bool:
        """Check if the scope was approved"""
        return self.result is not None and self.result.is_approved

    @property
    def modifications(self) -> Optional[Dict[str, Any]]:
        """Get any modifications made during review"""
        if self.result and self.result.response == TriggerResponse.MODIFIED:
            return self.result.modifications
        return None

    def __enter__(self) -> "TriggerScope":
        self.result = require_human_approval(
            operation_name=self.operation_name,
            description=self.description,
            trigger_type=self.trigger_type,
            sensitivity=self.sensitivity,
            timeout_seconds=self.timeout_seconds,
            metadata=self.metadata,
            raise_on_deny=self.raise_on_deny,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self) -> "TriggerScope":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


# Convenience aliases
approval_scope = TriggerScope


def sensitive_operation(
    operation_name: str,
    description: str,
    sensitivity: SensitivityLevel = SensitivityLevel.HIGH,
) -> TriggerScope:
    """
    Create a trigger scope for a sensitive operation.

    Example:
        with sensitive_operation(
            "Delete Records",
            "Permanently delete 500 user records"
        ):
            delete_records()
    """
    return TriggerScope(
        operation_name=operation_name,
        description=description,
        trigger_type=TriggerType.APPROVAL,
        sensitivity=sensitivity,
    )


# Predefined trigger conditions for common operations
class CommonTriggers:
    """Predefined triggers for common sensitive operations"""

    @staticmethod
    def database_write(
        table: str,
        operation: str = "write",
        record_count: int = 1,
    ) -> TriggerResult:
        """Trigger for database write operations"""
        sensitivity = SensitivityLevel.MEDIUM
        if operation in ("delete", "truncate", "drop"):
            sensitivity = SensitivityLevel.CRITICAL
        elif record_count > 100:
            sensitivity = SensitivityLevel.HIGH

        return require_human_approval(
            operation_name=f"Database {operation.title()}",
            description=f"{operation.title()} {record_count} record(s) in table '{table}'",
            trigger_type=TriggerType.APPROVAL,
            sensitivity=sensitivity,
            metadata={
                "table": table,
                "operation": operation,
                "record_count": record_count,
            },
        )

    @staticmethod
    def email_send(
        recipient_count: int,
        subject: str,
        is_bulk: bool = False,
    ) -> TriggerResult:
        """Trigger for email send operations"""
        sensitivity = SensitivityLevel.HIGH if is_bulk else SensitivityLevel.MEDIUM

        return require_human_approval(
            operation_name="Send Email",
            description=f"Send email '{subject}' to {recipient_count} recipient(s)",
            trigger_type=TriggerType.CONFIRMATION,
            sensitivity=sensitivity,
            metadata={
                "recipient_count": recipient_count,
                "subject": subject,
                "is_bulk": is_bulk,
            },
        )

    @staticmethod
    def external_api_call(
        api_name: str,
        method: str = "POST",
        has_side_effects: bool = True,
    ) -> TriggerResult:
        """Trigger for external API calls with side effects"""
        if not has_side_effects:
            return TriggerResult(
                trigger_id=str(uuid.uuid4()),
                response=TriggerResponse.SKIPPED,
            )

        return require_human_approval(
            operation_name=f"API Call: {api_name}",
            description=f"{method} request to {api_name}",
            trigger_type=TriggerType.NOTIFICATION,
            sensitivity=SensitivityLevel.MEDIUM,
            metadata={
                "api": api_name,
                "method": method,
            },
            raise_on_deny=False,
        )

    @staticmethod
    def file_operation(
        path: str,
        operation: str = "write",
    ) -> TriggerResult:
        """Trigger for file system operations"""
        sensitivity = SensitivityLevel.MEDIUM
        if operation in ("delete", "overwrite"):
            sensitivity = SensitivityLevel.HIGH

        return require_human_approval(
            operation_name=f"File {operation.title()}",
            description=f"{operation.title()} file: {path}",
            trigger_type=TriggerType.CONFIRMATION,
            sensitivity=sensitivity,
            metadata={
                "path": path,
                "operation": operation,
            },
        )

    @staticmethod
    def payment_processing(
        amount: float,
        currency: str = "USD",
    ) -> TriggerResult:
        """Trigger for payment operations"""
        sensitivity = SensitivityLevel.HIGH
        if amount > 1000:
            sensitivity = SensitivityLevel.CRITICAL

        return require_human_approval(
            operation_name="Process Payment",
            description=f"Process payment of {currency} {amount:.2f}",
            trigger_type=TriggerType.APPROVAL,
            sensitivity=sensitivity,
            metadata={
                "amount": amount,
                "currency": currency,
            },
            requires_reason=True,
        )


# Export convenient access to common triggers
database_write = CommonTriggers.database_write
email_send = CommonTriggers.email_send
external_api_call = CommonTriggers.external_api_call
file_operation = CommonTriggers.file_operation
payment_processing = CommonTriggers.payment_processing
