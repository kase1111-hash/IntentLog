"""
Boundary-Daemon Integration for IntentLog

This module provides integration between IntentLog and Boundary-Daemon
(Agent Smith) for security policy enforcement and audit logging.

Boundary-Daemon is a security policy and audit layer for AI agent systems
that provides:
- Policy decisions (authorization checks)
- Tamper-evident, hash-chained audit logs
- Boundary modes (OPEN, RESTRICTED, TRUSTED, AIRGAP, COLDROOM, LOCKDOWN)
- AI/Agent protections (prompt injection detection, etc.)

See: https://github.com/kase1111-hash/boundary-daemon-
"""

import json
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..logging import get_logger, log_context


class BoundaryMode(Enum):
    """Boundary-Daemon security modes."""
    OPEN = "OPEN"               # Full network access
    RESTRICTED = "RESTRICTED"   # Limited network access
    TRUSTED = "TRUSTED"         # Whitelisted endpoints only
    AIRGAP = "AIRGAP"           # No external network
    COLDROOM = "COLDROOM"       # Isolated processing
    LOCKDOWN = "LOCKDOWN"       # Emergency lockdown


class PolicyDecision(Enum):
    """Policy decision results from Boundary-Daemon."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    DEFER = "DEFER"         # Requires human approval
    AUDIT_ONLY = "AUDIT_ONLY"  # Allow but log


@dataclass
class PolicyRequest:
    """Request to Boundary-Daemon for policy decision."""
    action: str                 # e.g., "intent.create", "intent.read"
    resource: str               # e.g., "intent:<id>", "branch:<name>"
    subject: Optional[str] = None  # e.g., "user:alice", "agent:gpt4"
    context: Dict[str, Any] = field(default_factory=dict)
    justification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "resource": self.resource,
            "subject": self.subject,
            "context": self.context,
            "justification": self.justification,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


@dataclass
class PolicyResponse:
    """Response from Boundary-Daemon policy check."""
    decision: PolicyDecision
    reason: Optional[str] = None
    boundary_mode: Optional[BoundaryMode] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    audit_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyResponse":
        return cls(
            decision=PolicyDecision(data.get("decision", "DENY")),
            reason=data.get("reason"),
            boundary_mode=BoundaryMode(data["boundary_mode"]) if data.get("boundary_mode") else None,
            constraints=data.get("constraints", {}),
            audit_id=data.get("audit_id"),
        )


@dataclass
class AuditEvent:
    """Audit event to send to Boundary-Daemon."""
    event_type: str             # e.g., "intent.created", "branch.switched"
    severity: str = "INFO"      # DEBUG, INFO, WARNING, ERROR, CRITICAL
    resource: Optional[str] = None
    subject: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "SUCCESS"    # SUCCESS, FAILURE, PENDING
    details: Dict[str, Any] = field(default_factory=dict)
    intent_id: Optional[str] = None
    chain_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        event = {
            "event_type": self.event_type,
            "severity": self.severity,
            "outcome": self.outcome,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "IntentLog",
        }
        if self.resource:
            event["resource"] = self.resource
        if self.subject:
            event["subject"] = self.subject
        if self.action:
            event["action"] = self.action
        if self.details:
            event["details"] = self.details
        if self.intent_id:
            event["intent_id"] = self.intent_id
        if self.chain_hash:
            event["chain_hash"] = self.chain_hash
        return event


@dataclass
class BoundaryDaemonConfig:
    """Configuration for Boundary-Daemon integration."""
    socket_path: str = "/var/run/boundary-daemon/boundary.sock"
    http_endpoint: Optional[str] = None  # Alternative: "http://localhost:9500"
    api_key: Optional[str] = None
    api_key_env: str = "BOUNDARY_DAEMON_API_KEY"
    timeout: float = 5.0
    fail_open: bool = False     # If True, allow on daemon unavailability
    enable_audit: bool = True
    enable_policy_check: bool = True
    default_subject: Optional[str] = None

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


class BoundaryDaemonError(Exception):
    """Base exception for Boundary-Daemon errors."""
    pass


class PolicyDeniedError(BoundaryDaemonError):
    """Raised when policy check denies an action."""
    def __init__(self, message: str, response: Optional[PolicyResponse] = None):
        super().__init__(message)
        self.response = response


class DaemonUnavailableError(BoundaryDaemonError):
    """Raised when Boundary-Daemon is not available."""
    pass


class BoundaryDaemonIntegration:
    """
    Integration layer between IntentLog and Boundary-Daemon.

    This class provides:
    - Policy checks before sensitive operations
    - Audit event emission for all intent operations
    - Boundary mode awareness for adaptive behavior
    - Human override ceremony support
    """

    def __init__(self, config: Optional[BoundaryDaemonConfig] = None):
        self.config = config or BoundaryDaemonConfig()
        self._boundary_mode: Optional[BoundaryMode] = None
        self._socket: Optional[socket.socket] = None

    def _get_connection(self) -> socket.socket:
        """Get or create socket connection to daemon."""
        if self._socket is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            try:
                sock.connect(self.config.socket_path)
                self._socket = sock
            except (socket.error, FileNotFoundError) as e:
                raise DaemonUnavailableError(
                    f"Cannot connect to Boundary-Daemon at {self.config.socket_path}: {e}"
                )
        return self._socket

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to daemon and get response."""
        logger = get_logger()

        # Try HTTP endpoint first if configured
        if self.config.http_endpoint:
            return self._send_http_request(request)

        # Use Unix socket
        try:
            sock = self._get_connection()
            request_data = json.dumps(request).encode("utf-8")

            # Send length-prefixed message
            length = len(request_data)
            sock.sendall(length.to_bytes(4, "big") + request_data)

            # Read response length
            length_bytes = sock.recv(4)
            if len(length_bytes) < 4:
                raise DaemonUnavailableError("Incomplete response from daemon")

            response_length = int.from_bytes(length_bytes, "big")
            response_data = sock.recv(response_length)

            return json.loads(response_data.decode("utf-8"))

        except socket.timeout:
            logger.warning("Boundary-Daemon request timeout")
            raise DaemonUnavailableError("Daemon request timeout")
        except (socket.error, ConnectionError) as e:
            self._socket = None  # Reset connection
            logger.warning("Boundary-Daemon connection error", error=str(e))
            raise DaemonUnavailableError(f"Daemon connection error: {e}")

    def _send_http_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request via HTTP endpoint."""
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        url = f"{self.config.http_endpoint}/api/v1/query"
        headers = {"Content-Type": "application/json"}

        api_key = self.config.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = Request(
                url,
                data=json.dumps(request).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            raise BoundaryDaemonError(f"HTTP error from daemon: {e.code}")
        except URLError as e:
            raise DaemonUnavailableError(f"Cannot reach daemon: {e.reason}")

    def check_policy(self, request: PolicyRequest) -> PolicyResponse:
        """
        Check policy with Boundary-Daemon.

        Args:
            request: The policy request

        Returns:
            PolicyResponse with decision

        Raises:
            PolicyDeniedError: If policy denies the action
            DaemonUnavailableError: If daemon is unavailable (and fail_open=False)
        """
        logger = get_logger()

        if not self.config.enable_policy_check:
            logger.debug("Policy check disabled, allowing by default")
            return PolicyResponse(decision=PolicyDecision.ALLOW)

        # Add default subject if not provided
        if not request.subject and self.config.default_subject:
            request.subject = self.config.default_subject

        try:
            response_data = self._send_request({
                "type": "policy_check",
                "request": request.to_dict(),
            })

            response = PolicyResponse.from_dict(response_data)

            # Update cached boundary mode
            if response.boundary_mode:
                self._boundary_mode = response.boundary_mode

            logger.debug(
                "Policy check result",
                action=request.action,
                resource=request.resource,
                decision=response.decision.value,
            )

            return response

        except DaemonUnavailableError as e:
            if self.config.fail_open:
                logger.warning("Daemon unavailable, fail-open allowing action")
                return PolicyResponse(
                    decision=PolicyDecision.ALLOW,
                    reason="Daemon unavailable, fail-open policy",
                )
            raise

    def require_policy(self, request: PolicyRequest) -> PolicyResponse:
        """
        Check policy and raise if denied.

        Args:
            request: The policy request

        Returns:
            PolicyResponse if allowed

        Raises:
            PolicyDeniedError: If policy denies the action
        """
        response = self.check_policy(request)

        if response.decision == PolicyDecision.DENY:
            raise PolicyDeniedError(
                f"Policy denied: {request.action} on {request.resource}. "
                f"Reason: {response.reason or 'No reason provided'}",
                response=response,
            )

        if response.decision == PolicyDecision.DEFER:
            raise PolicyDeniedError(
                f"Policy requires human approval: {request.action} on {request.resource}",
                response=response,
            )

        return response

    def emit_audit_event(self, event: AuditEvent) -> Optional[str]:
        """
        Emit an audit event to Boundary-Daemon.

        Args:
            event: The audit event to emit

        Returns:
            Audit ID if successful, None if audit disabled or daemon unavailable
        """
        logger = get_logger()

        if not self.config.enable_audit:
            logger.debug("Audit disabled, skipping event emission")
            return None

        try:
            response = self._send_request({
                "type": "audit_event",
                "event": event.to_dict(),
            })

            audit_id = response.get("audit_id")
            logger.debug(
                "Audit event emitted",
                event_type=event.event_type,
                audit_id=audit_id,
            )
            return audit_id

        except DaemonUnavailableError:
            logger.warning(
                "Daemon unavailable for audit, event not recorded",
                event_type=event.event_type,
            )
            return None

    def get_boundary_mode(self) -> Optional[BoundaryMode]:
        """
        Get current boundary mode from daemon.

        Returns:
            Current BoundaryMode or None if unavailable
        """
        try:
            response = self._send_request({"type": "get_mode"})
            mode_str = response.get("mode")
            if mode_str:
                self._boundary_mode = BoundaryMode(mode_str)
            return self._boundary_mode
        except (DaemonUnavailableError, BoundaryDaemonError):
            return self._boundary_mode

    def is_available(self) -> bool:
        """Check if Boundary-Daemon is available."""
        try:
            self._send_request({"type": "health"})
            return True
        except (DaemonUnavailableError, BoundaryDaemonError):
            return False

    # Convenience methods for common IntentLog operations

    def check_intent_create(
        self,
        intent_name: str,
        branch: str = "main",
        subject: Optional[str] = None,
    ) -> PolicyResponse:
        """Check policy for creating an intent."""
        return self.check_policy(PolicyRequest(
            action="intent.create",
            resource=f"branch:{branch}",
            subject=subject,
            context={"intent_name": intent_name},
        ))

    def check_intent_read(
        self,
        intent_id: str,
        subject: Optional[str] = None,
    ) -> PolicyResponse:
        """Check policy for reading an intent."""
        return self.check_policy(PolicyRequest(
            action="intent.read",
            resource=f"intent:{intent_id}",
            subject=subject,
        ))

    def check_branch_create(
        self,
        branch_name: str,
        subject: Optional[str] = None,
    ) -> PolicyResponse:
        """Check policy for creating a branch."""
        return self.check_policy(PolicyRequest(
            action="branch.create",
            resource=f"branch:{branch_name}",
            subject=subject,
        ))

    def check_crypto_operation(
        self,
        operation: str,
        subject: Optional[str] = None,
        justification: Optional[str] = None,
    ) -> PolicyResponse:
        """Check policy for cryptographic operations."""
        return self.check_policy(PolicyRequest(
            action=f"crypto.{operation}",
            resource="crypto:keys",
            subject=subject,
            justification=justification,
        ))

    def audit_intent_created(
        self,
        intent_id: str,
        intent_name: str,
        branch: str,
        chain_hash: Optional[str] = None,
    ) -> Optional[str]:
        """Emit audit event for intent creation."""
        return self.emit_audit_event(AuditEvent(
            event_type="intent.created",
            severity="INFO",
            resource=f"intent:{intent_id}",
            action="create",
            outcome="SUCCESS",
            intent_id=intent_id,
            chain_hash=chain_hash,
            details={
                "intent_name": intent_name,
                "branch": branch,
            },
        ))

    def audit_branch_created(
        self,
        branch_name: str,
        from_branch: str,
    ) -> Optional[str]:
        """Emit audit event for branch creation."""
        return self.emit_audit_event(AuditEvent(
            event_type="branch.created",
            severity="INFO",
            resource=f"branch:{branch_name}",
            action="create",
            outcome="SUCCESS",
            details={
                "from_branch": from_branch,
            },
        ))

    def audit_key_generated(
        self,
        key_id: str,
        algorithm: str = "ed25519",
    ) -> Optional[str]:
        """Emit audit event for key generation."""
        return self.emit_audit_event(AuditEvent(
            event_type="crypto.key_generated",
            severity="WARNING",  # Key generation is security-sensitive
            resource=f"key:{key_id}",
            action="generate",
            outcome="SUCCESS",
            details={
                "algorithm": algorithm,
            },
        ))

    def audit_signature_verified(
        self,
        intent_id: str,
        valid: bool,
        signer_id: Optional[str] = None,
    ) -> Optional[str]:
        """Emit audit event for signature verification."""
        return self.emit_audit_event(AuditEvent(
            event_type="crypto.signature_verified",
            severity="INFO" if valid else "WARNING",
            resource=f"intent:{intent_id}",
            action="verify",
            outcome="SUCCESS" if valid else "FAILURE",
            intent_id=intent_id,
            details={
                "valid": valid,
                "signer_id": signer_id,
            },
        ))

    def audit_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
    ) -> Optional[str]:
        """Emit a generic security audit event."""
        return self.emit_audit_event(AuditEvent(
            event_type=event_type,
            severity=severity,
            outcome="ALERT",
            details=details,
        ))


def policy_required(
    action: str,
    resource_fn: Optional[Callable[..., str]] = None,
):
    """
    Decorator for requiring policy check before function execution.

    Args:
        action: The policy action (e.g., "intent.create")
        resource_fn: Function to extract resource from function args

    Usage:
        @policy_required("intent.create", lambda name, **kw: f"branch:{kw.get('branch', 'main')}")
        def create_intent(name: str, branch: str = "main"):
            ...
    """
    from functools import wraps

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get integration from global or create default
            integration = get_boundary_daemon_integration()

            if integration:
                resource = resource_fn(*args, **kwargs) if resource_fn else "unknown"
                integration.require_policy(PolicyRequest(
                    action=action,
                    resource=resource,
                ))

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global integration instance
_integration: Optional[BoundaryDaemonIntegration] = None


def get_boundary_daemon_integration() -> Optional[BoundaryDaemonIntegration]:
    """Get the global Boundary-Daemon integration instance."""
    return _integration


def configure_boundary_daemon(
    config: Optional[BoundaryDaemonConfig] = None,
    **kwargs,
) -> BoundaryDaemonIntegration:
    """
    Configure the global Boundary-Daemon integration.

    Args:
        config: Configuration object, or pass kwargs
        **kwargs: Configuration options

    Returns:
        Configured integration instance
    """
    global _integration

    if config is None:
        config = BoundaryDaemonConfig(**kwargs)

    _integration = BoundaryDaemonIntegration(config)
    return _integration


__all__ = [
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
]
