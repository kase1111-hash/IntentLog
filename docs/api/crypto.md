# Crypto Module

The crypto module provides Ed25519 signing and key management.

!!! note "Optional Dependency"
    This module requires the `cryptography` package:
    ```bash
    pip install intentlog[crypto]
    ```

## KeyManager

Manages Ed25519 signing keys.

::: intentlog.crypto.KeyManager
    options:
      show_root_heading: true
      members:
        - generate_key
        - has_keys
        - list_keys
        - get_default_key_name
        - set_default_key
        - load_key
        - load_public_key
        - export_public_key

## Signing Functions

Intents are signed and verified with module-level helpers that take a
`KeyManager`.

::: intentlog.crypto.sign_intent
    options:
      show_root_heading: true

::: intentlog.crypto.verify_intent_signature
    options:
      show_root_heading: true

## KeyPair

Represents an Ed25519 key pair.

::: intentlog.crypto.KeyPair
    options:
      show_root_heading: true

## Exceptions

::: intentlog.crypto.CryptoNotAvailableError
    options:
      show_root_heading: true

::: intentlog.crypto.KeyNotFoundError
    options:
      show_root_heading: true

## Usage Examples

### Key Generation

```python
from intentlog.crypto import KeyManager
from pathlib import Path

# Initialize key manager
key_manager = KeyManager(Path(".intentlog"))

# Generate a new key pair
key_pair = key_manager.generate_key(
    name="signing-key",
    password="optional-password"
)

print(f"Key ID: {key_pair.key_id}")
print(f"Created: {key_pair.created_at}")
```

### Signing and Verification

```python
from intentlog.crypto import sign_intent, verify_intent_signature

# Sign a dictionary of intent data
data = {"intent_id": "abc123", "intent_hash": "..."}
signature = sign_intent(data, key_manager, key_name="signing-key")

# Verify the signature (raises SignatureError if invalid)
is_valid = verify_intent_signature(data, signature, key_manager)
print(f"Signature valid: {is_valid}")
```

### Exporting Public Keys

```python
# Export public key in PEM format
public_pem = key_manager.export_public_key("signing-key")
print(public_pem)
```
