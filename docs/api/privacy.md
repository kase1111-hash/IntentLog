# Privacy Module

The privacy module implements MP-02 Section 12 privacy controls.

## PrivacyManager

Main class for managing privacy controls.

::: intentlog.privacy.PrivacyManager
    options:
      show_root_heading: true
      members:
        - encrypt_intent
        - decrypt_intent
        - revoke_future_observation
        - is_revoked

## Privacy Levels

::: intentlog.privacy.PrivacyLevel
    options:
      show_root_heading: true

## RevocationRecord

Records consent revocation per MP-02 Section 12.

::: intentlog.privacy.RevocationRecord
    options:
      show_root_heading: true

## AccessPolicy

Defines access control for intents.

::: intentlog.privacy.AccessPolicy
    options:
      show_root_heading: true

## Usage Examples

### Encrypting Intents

```python
from intentlog.privacy import PrivacyManager
from pathlib import Path

privacy = PrivacyManager(Path("."))

# Encrypt an intent
encrypted = privacy.encrypt_intent(intent)

# Decrypt when needed
decrypted = privacy.decrypt_intent(encrypted)
```

### Revoking Observation

```python
# Revoke future observation (MP-02 Section 12)
record = privacy.revoke_future_observation(
    user_id="user-123",
    reason="Privacy request"
)

print(f"Revocation ID: {record.record_id}")
print("Note: Past receipts remain immutable")
```

### Checking Revocation Status

```python
if privacy.is_revoked():
    print("Observation has been revoked")
else:
    print("Observation is active")
```

## MP-02 Section 12 Compliance

The privacy module implements these MP-02 requirements:

1. **Raw signals MAY be encrypted** - Fernet symmetric encryption
2. **Receipts MUST NOT expose raw content** - Only hashes stored
3. **Humans MAY revoke future observation** - Revocation mechanism
4. **Past receipts remain immutable** - Cryptographic integrity preserved
