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
        - list_keys
        - get_default_key_name
        - set_default_key
        - export_public_key
        - sign
        - verify

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
# Sign data
data = b"intent data to sign"
signature = key_manager.sign(data, key_name="signing-key")

# Verify signature
is_valid = key_manager.verify(data, signature, key_name="signing-key")
print(f"Signature valid: {is_valid}")
```

### Exporting Public Keys

```python
# Export public key in PEM format
public_pem = key_manager.export_public_key("signing-key")
print(public_pem)
```
