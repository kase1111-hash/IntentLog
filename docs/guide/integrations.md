# External Integrations

IntentLog provides integrations with external security and infrastructure systems to enhance its capabilities in enterprise environments.

## Boundary-Daemon Integration

[Boundary-Daemon](https://github.com/kase1111-hash/boundary-daemon-) (Agent Smith) is a security policy and audit layer for AI agent systems. IntentLog integrates with Boundary-Daemon for:

- **Policy Enforcement**: Check authorization before sensitive operations
- **Audit Logging**: Emit tamper-evident audit events
- **Boundary Mode Awareness**: Adapt behavior based on security posture

### Configuration

```python
from intentlog.integrations import (
    BoundaryDaemonConfig,
    configure_boundary_daemon,
)

# Configure the integration
daemon = configure_boundary_daemon(
    socket_path="/var/run/boundary-daemon/boundary.sock",
    enable_policy_check=True,
    enable_audit=True,
    fail_open=False,  # Deny on daemon unavailability
)

# Or use HTTP endpoint
daemon = configure_boundary_daemon(
    http_endpoint="http://localhost:9500",
    api_key_env="BOUNDARY_DAEMON_API_KEY",
)
```

### Policy Checks

```python
from intentlog.integrations import (
    PolicyRequest,
    get_boundary_daemon_integration,
)

daemon = get_boundary_daemon_integration()

# Check policy before creating intent
response = daemon.check_policy(PolicyRequest(
    action="intent.create",
    resource="branch:main",
    subject="user:alice",
    justification="Recording design decision",
))

if response.decision == PolicyDecision.ALLOW:
    # Proceed with operation
    storage.add_intent(name="Design Decision", reasoning="...")
```

### Audit Events

```python
# Emit audit events for security-relevant operations
daemon.audit_intent_created(
    intent_id="abc123",
    intent_name="Security Policy Update",
    branch="main",
    chain_hash="def456",
)

daemon.audit_key_generated(
    key_id="key-001",
    algorithm="ed25519",
)
```

### Policy Decorator

```python
from intentlog.integrations import policy_required

@policy_required("intent.create", lambda name, **kw: f"branch:{kw.get('branch', 'main')}")
def create_intent(name: str, reasoning: str, branch: str = "main"):
    # This function will only execute if policy allows
    ...
```

### Boundary Modes

IntentLog can query and respond to Boundary-Daemon's security modes:

| Mode | Description | IntentLog Behavior |
|------|-------------|-------------------|
| OPEN | Full access | Normal operation |
| RESTRICTED | Limited network | Disable LLM features |
| TRUSTED | Whitelisted only | Require signed intents |
| AIRGAP | No external network | Local-only operations |
| COLDROOM | Isolated processing | Read-only mode |
| LOCKDOWN | Emergency | Deny all writes |

```python
mode = daemon.get_boundary_mode()
if mode == BoundaryMode.LOCKDOWN:
    raise SecurityError("System is in lockdown mode")
```

---

## Boundary-SIEM Integration

[Boundary-SIEM](https://github.com/kase1111-hash/Boundary-SIEM) is a comprehensive SIEM platform for security event management. IntentLog integrates with Boundary-SIEM for:

- **Event Logging**: Emit events in CEF, LEEF, or JSON format
- **Compliance Reporting**: Tag events for SOC 2, ISO 27001, NIST CSF
- **Threat Detection**: Enable SIEM correlation rules

### Configuration

```python
from intentlog.integrations import (
    BoundarySIEMConfig,
    configure_siem,
)

# Configure HTTP endpoint
siem = configure_siem(
    endpoint="http://siem.example.com:8080",
    api_path="/api/v1/events",
    api_key_env="BOUNDARY_SIEM_API_KEY",
    output_format="json",  # or "cef", "leef"
    enable_async=True,
    batch_size=100,
    compliance_tags=["SOC2", "NIST-CSF"],
)

# Or configure syslog
siem = configure_siem(
    syslog_host="syslog.example.com",
    syslog_port=514,
    syslog_protocol="tcp",
    output_format="cef",
)
```

### Emitting Events

```python
from intentlog.integrations import get_siem_integration

siem = get_siem_integration()

# Emit intent events
siem.emit_intent_created(
    intent_id="abc123",
    intent_name="Architecture Decision",
    branch="main",
    chain_hash="def456",
    user="alice",
)

# Emit security events
siem.emit_signature_invalid(
    intent_id="xyz789",
    reason="Signature verification failed: key not found",
)

# Emit compliance events
siem.emit_compliance_event(
    event_type="DataRetentionCheck",
    message="Intent retention policy verified",
    compliance_framework="SOC2",
    control_id="CC6.1",
    status="Compliant",
)
```

### Custom Events

```python
from intentlog.integrations import SIEMEvent, EventCategory, EventSeverity

# Create custom event
event = SIEMEvent(
    event_type="CustomOperation",
    category=EventCategory.AUDIT,
    severity=EventSeverity.MEDIUM,
    message="Custom operation performed",
    intent_id="abc123",
    extension={
        "custom_field": "custom_value",
    },
)

siem.emit(event)
```

### CEF Format Example

IntentLog events are formatted according to the Common Event Format (CEF) specification:

```
CEF:0|IntentLog|IntentLog|0.1.0|IL-INTENT-IntentCreated|IntentCreated|3|rt=1704067200000 msg=Intent created: Architecture Decision src=hostname outcome=Success cs1=abc123 cs1Label=IntentID cs2=main cs2Label=Branch cs3=def456 cs3Label=ChainHash cat=intent
```

### Shutdown

```python
from intentlog.integrations import shutdown_siem

# Flush remaining events and stop worker
shutdown_siem(timeout=5.0)
```

---

## Combined Usage

For maximum security, use both integrations together:

```python
from intentlog.integrations import (
    configure_boundary_daemon,
    configure_siem,
    get_boundary_daemon_integration,
    get_siem_integration,
    PolicyRequest,
    PolicyDecision,
)

# Configure both integrations
daemon = configure_boundary_daemon(
    http_endpoint="http://localhost:9500",
)
siem = configure_siem(
    endpoint="http://siem.example.com:8080",
)

def create_secure_intent(name: str, reasoning: str, branch: str = "main"):
    # Check policy first
    response = daemon.check_policy(PolicyRequest(
        action="intent.create",
        resource=f"branch:{branch}",
    ))

    if response.decision != PolicyDecision.ALLOW:
        # Log denial to SIEM
        siem.emit_access_denied(
            resource=f"branch:{branch}",
            reason=response.reason,
        )
        raise PermissionError(f"Policy denied: {response.reason}")

    # Create intent
    intent = storage.add_intent(name=name, reasoning=reasoning, branch=branch)

    # Audit to daemon
    daemon.audit_intent_created(
        intent_id=intent.intent_id,
        intent_name=name,
        branch=branch,
    )

    # Log to SIEM
    siem.emit_intent_created(
        intent_id=intent.intent_id,
        intent_name=name,
        branch=branch,
    )

    return intent
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BOUNDARY_DAEMON_API_KEY` | API key for Boundary-Daemon HTTP endpoint |
| `BOUNDARY_SIEM_API_KEY` | API key for Boundary-SIEM HTTP endpoint |

---

## Error Handling

```python
from intentlog.integrations import (
    PolicyDeniedError,
    DaemonUnavailableError,
    EventDeliveryError,
)

try:
    daemon.require_policy(request)
except PolicyDeniedError as e:
    print(f"Access denied: {e.response.reason}")
except DaemonUnavailableError:
    print("Daemon unavailable, using fallback policy")

try:
    siem.emit_sync(event)
except EventDeliveryError as e:
    print(f"Failed to deliver event: {e}")
```
