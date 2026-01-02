"""
Boundary-SIEM Integration for IntentLog

This module provides integration between IntentLog and Boundary-SIEM
for security event logging, compliance reporting, and threat detection.

Boundary-SIEM is a comprehensive SIEM platform with:
- CEF/JSON event ingestion
- Real-time search and correlation
- Rule-based detection and alerting
- Compliance reporting (SOC 2, ISO 27001, NIST CSF, PCI DSS)

See: https://github.com/kase1111-hash/Boundary-SIEM
"""

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from ..logging import get_logger


class EventSeverity(Enum):
    """CEF severity levels (0-10 scale)."""
    UNKNOWN = 0
    LOW = 3
    MEDIUM = 5
    HIGH = 7
    VERY_HIGH = 8
    CRITICAL = 10


class EventCategory(Enum):
    """IntentLog event categories for SIEM classification."""
    INTENT = "intent"
    BRANCH = "branch"
    CRYPTO = "crypto"
    AUTH = "auth"
    PRIVACY = "privacy"
    CHAIN = "chain"
    AUDIT = "audit"
    SECURITY = "security"


@dataclass
class SIEMEvent:
    """
    Event structure for Boundary-SIEM ingestion.

    Supports both CEF (Common Event Format) and JSON output.
    """
    event_type: str             # e.g., "IntentCreated", "ChainVerified"
    category: EventCategory
    severity: EventSeverity = EventSeverity.LOW
    message: str = ""
    source_host: str = ""
    source_user: Optional[str] = None
    destination: Optional[str] = None
    intent_id: Optional[str] = None
    branch: Optional[str] = None
    chain_hash: Optional[str] = None
    signature_id: Optional[str] = None
    outcome: str = "Success"    # Success, Failure, Unknown
    reason: Optional[str] = None
    extension: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if not self.source_host:
            import socket
            self.source_host = socket.gethostname()

    def to_cef(self) -> str:
        """
        Convert to CEF (Common Event Format) string.

        CEF format:
        CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension

        Returns:
            CEF-formatted string
        """
        # CEF header fields
        version = "0"
        vendor = "IntentLog"
        product = "IntentLog"
        device_version = "0.1.0"
        sig_id = self.signature_id or f"IL-{self.category.value.upper()}-{self.event_type}"
        name = self.event_type
        severity = str(self.severity.value)

        # Build extension fields
        ext_parts = []

        # Standard CEF extension fields
        ext_parts.append(f"rt={int(self.timestamp.timestamp() * 1000)}")
        ext_parts.append(f"msg={self._escape_cef_value(self.message)}")
        ext_parts.append(f"src={self.source_host}")
        ext_parts.append(f"outcome={self.outcome}")

        if self.source_user:
            ext_parts.append(f"suser={self._escape_cef_value(self.source_user)}")
        if self.destination:
            ext_parts.append(f"dst={self._escape_cef_value(self.destination)}")

        # Custom IntentLog fields (cs1-cs6 for custom strings)
        if self.intent_id:
            ext_parts.append(f"cs1={self.intent_id}")
            ext_parts.append("cs1Label=IntentID")
        if self.branch:
            ext_parts.append(f"cs2={self._escape_cef_value(self.branch)}")
            ext_parts.append("cs2Label=Branch")
        if self.chain_hash:
            ext_parts.append(f"cs3={self.chain_hash}")
            ext_parts.append("cs3Label=ChainHash")
        if self.reason:
            ext_parts.append(f"reason={self._escape_cef_value(self.reason)}")

        # Category
        ext_parts.append(f"cat={self.category.value}")

        # Additional extension fields
        for key, value in self.extension.items():
            ext_parts.append(f"{key}={self._escape_cef_value(str(value))}")

        extension = " ".join(ext_parts)

        return f"CEF:{version}|{vendor}|{product}|{device_version}|{sig_id}|{name}|{severity}|{extension}"

    def to_leef(self) -> str:
        """
        Convert to LEEF (Log Event Extended Format) string.

        LEEF format (IBM QRadar):
        LEEF:Version|Vendor|Product|Version|EventID|Key1=Value1\tKey2=Value2

        Returns:
            LEEF-formatted string
        """
        version = "2.0"
        vendor = "IntentLog"
        product = "IntentLog"
        product_version = "0.1.0"
        event_id = self.signature_id or f"{self.category.value}_{self.event_type}"

        # Build attributes (tab-separated)
        attrs = []
        attrs.append(f"devTime={self.timestamp.strftime('%b %d %Y %H:%M:%S')}")
        attrs.append(f"sev={self.severity.value}")
        attrs.append(f"cat={self.category.value}")
        attrs.append(f"src={self.source_host}")
        attrs.append(f"msg={self._escape_leef_value(self.message)}")

        if self.source_user:
            attrs.append(f"usrName={self._escape_leef_value(self.source_user)}")
        if self.intent_id:
            attrs.append(f"intentId={self.intent_id}")
        if self.branch:
            attrs.append(f"branch={self._escape_leef_value(self.branch)}")
        if self.chain_hash:
            attrs.append(f"chainHash={self.chain_hash}")
        if self.outcome:
            attrs.append(f"outcome={self.outcome}")

        for key, value in self.extension.items():
            attrs.append(f"{key}={self._escape_leef_value(str(value))}")

        attributes = "\t".join(attrs)

        return f"LEEF:{version}|{vendor}|{product}|{product_version}|{event_id}|{attributes}"

    def to_json(self) -> str:
        """
        Convert to JSON format for Boundary-SIEM HTTP ingestion.

        Returns:
            JSON-formatted string
        """
        event_dict = {
            "timestamp": self.timestamp.isoformat() + "Z",
            "event_type": self.event_type,
            "category": self.category.value,
            "severity": self.severity.value,
            "severity_name": self.severity.name,
            "message": self.message,
            "source": {
                "host": self.source_host,
                "user": self.source_user,
                "application": "IntentLog",
            },
            "outcome": self.outcome,
        }

        if self.destination:
            event_dict["destination"] = self.destination
        if self.intent_id:
            event_dict["intent_id"] = self.intent_id
        if self.branch:
            event_dict["branch"] = self.branch
        if self.chain_hash:
            event_dict["chain_hash"] = self.chain_hash
        if self.signature_id:
            event_dict["signature_id"] = self.signature_id
        if self.reason:
            event_dict["reason"] = self.reason
        if self.extension:
            event_dict["extension"] = self.extension

        return json.dumps(event_dict, separators=(",", ":"))

    def _escape_cef_value(self, value: str) -> str:
        """Escape special characters for CEF format."""
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("=", "\\=")

    def _escape_leef_value(self, value: str) -> str:
        """Escape special characters for LEEF format."""
        return value.replace("\t", " ").replace("\n", " ").replace("\r", "")


@dataclass
class BoundarySIEMConfig:
    """Configuration for Boundary-SIEM integration."""
    # HTTP endpoint configuration
    endpoint: str = "http://localhost:8080"
    api_path: str = "/api/v1/events"
    api_key: Optional[str] = None
    api_key_env: str = "BOUNDARY_SIEM_API_KEY"

    # Output format
    output_format: str = "json"  # "cef", "leef", "json"

    # Syslog configuration (alternative to HTTP)
    syslog_host: Optional[str] = None
    syslog_port: int = 514
    syslog_protocol: str = "udp"  # "udp", "tcp", "tls"

    # Batching configuration
    batch_size: int = 100
    batch_interval_seconds: float = 5.0
    max_queue_size: int = 10000

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Feature flags
    enable_async: bool = True
    enable_compression: bool = True
    include_chain_hashes: bool = True

    # Compliance tagging
    compliance_tags: List[str] = field(default_factory=lambda: ["SOC2", "NIST-CSF"])

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


class BoundarySIEMError(Exception):
    """Base exception for Boundary-SIEM errors."""
    pass


class EventDeliveryError(BoundarySIEMError):
    """Raised when event delivery fails."""
    pass


class BoundarySIEMIntegration:
    """
    Integration layer between IntentLog and Boundary-SIEM.

    This class provides:
    - Event emission in CEF, LEEF, or JSON format
    - Batched async delivery for performance
    - Retry logic with exponential backoff
    - Compliance tagging for regulatory reporting
    """

    def __init__(self, config: Optional[BoundarySIEMConfig] = None):
        self.config = config or BoundarySIEMConfig()
        self._event_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        """Start the async event delivery worker."""
        if self._started:
            return

        if self.config.enable_async:
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._event_worker,
                daemon=True,
                name="BoundarySIEM-Worker",
            )
            self._worker_thread.start()

        self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the async event delivery worker."""
        if not self._started:
            return

        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        self._started = False

    def _event_worker(self) -> None:
        """Background worker for batch event delivery."""
        logger = get_logger()
        batch: List[SIEMEvent] = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                # Get event with timeout
                try:
                    event = self._event_queue.get(timeout=0.5)
                    batch.append(event)
                except queue.Empty:
                    pass

                # Flush if batch is full or interval elapsed
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and time.time() - last_flush >= self.config.batch_interval_seconds)
                )

                if should_flush:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"SIEM worker error: {e}", exc_info=True)

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, events: List[SIEMEvent]) -> None:
        """Flush a batch of events to SIEM."""
        if not events:
            return

        logger = get_logger()

        for attempt in range(self.config.max_retries):
            try:
                if self.config.syslog_host:
                    self._send_syslog_batch(events)
                else:
                    self._send_http_batch(events)

                logger.debug(f"Flushed {len(events)} events to SIEM")
                return

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(f"SIEM delivery failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"SIEM delivery failed after {self.config.max_retries} attempts: {e}")

    def _send_http_batch(self, events: List[SIEMEvent]) -> None:
        """Send events via HTTP to Boundary-SIEM."""
        url = f"{self.config.endpoint}{self.config.api_path}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "IntentLog/0.1.0",
        }

        api_key = self.config.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Format events based on config
        if self.config.output_format == "cef":
            body = {"events": [{"cef": e.to_cef()} for e in events]}
        elif self.config.output_format == "leef":
            body = {"events": [{"leef": e.to_leef()} for e in events]}
        else:
            body = {"events": [json.loads(e.to_json()) for e in events]}

        # Add compliance tags
        if self.config.compliance_tags:
            body["compliance_tags"] = self.config.compliance_tags

        request_data = json.dumps(body).encode("utf-8")

        # Compress if enabled
        if self.config.enable_compression and len(request_data) > 1024:
            import gzip
            request_data = gzip.compress(request_data)
            headers["Content-Encoding"] = "gzip"

        try:
            req = Request(url, data=request_data, headers=headers, method="POST")
            with urlopen(req, timeout=10) as response:
                if response.status >= 400:
                    raise EventDeliveryError(f"HTTP {response.status}")
        except HTTPError as e:
            raise EventDeliveryError(f"HTTP error: {e.code}")
        except URLError as e:
            raise EventDeliveryError(f"Connection error: {e.reason}")

    def _send_syslog_batch(self, events: List[SIEMEvent]) -> None:
        """Send events via syslog."""
        import socket

        if self.config.syslog_protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.syslog_host, self.config.syslog_port))

        try:
            for event in events:
                # Format as syslog message
                if self.config.output_format == "cef":
                    message = event.to_cef()
                elif self.config.output_format == "leef":
                    message = event.to_leef()
                else:
                    message = event.to_json()

                # Add syslog priority (facility=local0, severity from event)
                facility = 16  # local0
                priority = (facility * 8) + min(event.severity.value, 7)
                syslog_msg = f"<{priority}>{message}\n"

                if self.config.syslog_protocol == "udp":
                    sock.sendto(
                        syslog_msg.encode("utf-8"),
                        (self.config.syslog_host, self.config.syslog_port),
                    )
                else:
                    sock.sendall(syslog_msg.encode("utf-8"))
        finally:
            sock.close()

    def emit(self, event: SIEMEvent) -> None:
        """
        Emit a single event to SIEM.

        Args:
            event: The SIEM event to emit

        If async mode is enabled, event is queued for batch delivery.
        Otherwise, event is sent immediately.
        """
        if not self._started:
            self.start()

        if self.config.enable_async:
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                get_logger().warning("SIEM event queue full, dropping event")
        else:
            self._flush_batch([event])

    def emit_sync(self, event: SIEMEvent) -> None:
        """Emit a single event synchronously (bypasses queue)."""
        self._flush_batch([event])

    # Convenience methods for common IntentLog events

    def emit_intent_created(
        self,
        intent_id: str,
        intent_name: str,
        branch: str = "main",
        chain_hash: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        """Emit event for intent creation."""
        self.emit(SIEMEvent(
            event_type="IntentCreated",
            category=EventCategory.INTENT,
            severity=EventSeverity.LOW,
            message=f"Intent created: {intent_name}",
            intent_id=intent_id,
            branch=branch,
            chain_hash=chain_hash,
            source_user=user,
            outcome="Success",
        ))

    def emit_intent_signed(
        self,
        intent_id: str,
        signer: str,
        chain_hash: Optional[str] = None,
    ) -> None:
        """Emit event for intent signing."""
        self.emit(SIEMEvent(
            event_type="IntentSigned",
            category=EventCategory.CRYPTO,
            severity=EventSeverity.MEDIUM,
            message=f"Intent signed by {signer}",
            intent_id=intent_id,
            chain_hash=chain_hash,
            source_user=signer,
            outcome="Success",
        ))

    def emit_chain_verified(
        self,
        branch: str,
        chain_hash: str,
        valid: bool,
        intent_count: int,
    ) -> None:
        """Emit event for chain verification."""
        self.emit(SIEMEvent(
            event_type="ChainVerified",
            category=EventCategory.CHAIN,
            severity=EventSeverity.LOW if valid else EventSeverity.HIGH,
            message=f"Chain verification {'passed' if valid else 'FAILED'} for branch {branch}",
            branch=branch,
            chain_hash=chain_hash,
            outcome="Success" if valid else "Failure",
            extension={"intent_count": intent_count},
        ))

    def emit_signature_invalid(
        self,
        intent_id: str,
        reason: str,
    ) -> None:
        """Emit event for invalid signature (security alert)."""
        self.emit(SIEMEvent(
            event_type="SignatureInvalid",
            category=EventCategory.SECURITY,
            severity=EventSeverity.CRITICAL,
            message=f"Invalid signature detected on intent {intent_id}",
            intent_id=intent_id,
            outcome="Failure",
            reason=reason,
            signature_id="IL-SEC-001",
        ))

    def emit_privacy_redaction(
        self,
        intent_id: str,
        redaction_type: str,
        user: Optional[str] = None,
    ) -> None:
        """Emit event for privacy redaction."""
        self.emit(SIEMEvent(
            event_type="PrivacyRedaction",
            category=EventCategory.PRIVACY,
            severity=EventSeverity.MEDIUM,
            message=f"Privacy redaction applied: {redaction_type}",
            intent_id=intent_id,
            source_user=user,
            outcome="Success",
            extension={"redaction_type": redaction_type},
        ))

    def emit_key_generated(
        self,
        key_id: str,
        algorithm: str,
        user: Optional[str] = None,
    ) -> None:
        """Emit event for key generation."""
        self.emit(SIEMEvent(
            event_type="KeyGenerated",
            category=EventCategory.CRYPTO,
            severity=EventSeverity.MEDIUM,
            message=f"Cryptographic key generated: {algorithm}",
            source_user=user,
            outcome="Success",
            extension={
                "key_id": key_id,
                "algorithm": algorithm,
            },
        ))

    def emit_access_denied(
        self,
        resource: str,
        user: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Emit event for access denial."""
        self.emit(SIEMEvent(
            event_type="AccessDenied",
            category=EventCategory.AUTH,
            severity=EventSeverity.HIGH,
            message=f"Access denied to resource: {resource}",
            destination=resource,
            source_user=user,
            outcome="Failure",
            reason=reason,
            signature_id="IL-AUTH-001",
        ))

    def emit_compliance_event(
        self,
        event_type: str,
        message: str,
        compliance_framework: str,
        control_id: str,
        status: str = "Compliant",
    ) -> None:
        """Emit a compliance-related event."""
        self.emit(SIEMEvent(
            event_type=event_type,
            category=EventCategory.AUDIT,
            severity=EventSeverity.LOW if status == "Compliant" else EventSeverity.HIGH,
            message=message,
            outcome="Success" if status == "Compliant" else "Failure",
            extension={
                "compliance_framework": compliance_framework,
                "control_id": control_id,
                "compliance_status": status,
            },
        ))


# Global integration instance
_siem_integration: Optional[BoundarySIEMIntegration] = None


def get_siem_integration() -> Optional[BoundarySIEMIntegration]:
    """Get the global Boundary-SIEM integration instance."""
    return _siem_integration


def configure_siem(
    config: Optional[BoundarySIEMConfig] = None,
    **kwargs,
) -> BoundarySIEMIntegration:
    """
    Configure the global Boundary-SIEM integration.

    Args:
        config: Configuration object, or pass kwargs
        **kwargs: Configuration options

    Returns:
        Configured integration instance
    """
    global _siem_integration

    if config is None:
        config = BoundarySIEMConfig(**kwargs)

    _siem_integration = BoundarySIEMIntegration(config)
    _siem_integration.start()
    return _siem_integration


def shutdown_siem(timeout: float = 5.0) -> None:
    """Shutdown the global SIEM integration."""
    global _siem_integration
    if _siem_integration:
        _siem_integration.stop(timeout)
        _siem_integration = None


__all__ = [
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
