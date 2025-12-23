"""
Cryptographic Module for IntentLog

Provides Ed25519 digital signatures and key management for
tamper-evident intent verification per MP-02 specification.

Features:
- Ed25519 key pair generation
- Intent signing and verification
- Key serialization (PEM format)
- Key storage and loading from .intentlog/keys/
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Use cryptography library for Ed25519
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Ed25519PrivateKey = None
    Ed25519PublicKey = None


class CryptoError(Exception):
    """Base exception for cryptographic errors"""
    pass


class KeyNotFoundError(CryptoError):
    """Raised when a key file is not found"""
    pass


class SignatureError(CryptoError):
    """Raised when signature verification fails"""
    pass


class CryptoNotAvailableError(CryptoError):
    """Raised when cryptography library is not installed"""
    pass


@dataclass
class KeyPair:
    """Ed25519 key pair container"""
    private_key: Any  # Ed25519PrivateKey
    public_key: Any   # Ed25519PublicKey
    key_id: str       # Short identifier for the key
    created_at: str   # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Export key metadata (not the actual keys)"""
        return {
            "key_id": self.key_id,
            "created_at": self.created_at,
            "algorithm": "Ed25519",
        }


@dataclass
class Signature:
    """Digital signature with metadata"""
    signature: bytes      # Raw signature bytes
    key_id: str           # ID of signing key
    algorithm: str = "Ed25519"
    timestamp: str = ""   # When signature was created

    def to_dict(self) -> Dict[str, Any]:
        """Export signature as dictionary"""
        return {
            "signature": self.signature.hex(),
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        """Create Signature from dictionary"""
        return cls(
            signature=bytes.fromhex(data["signature"]),
            key_id=data["key_id"],
            algorithm=data.get("algorithm", "Ed25519"),
            timestamp=data.get("timestamp", ""),
        )

    def to_hex(self) -> str:
        """Return signature as hex string"""
        return self.signature.hex()


def check_crypto_available() -> None:
    """Raise error if cryptography library not available"""
    if not CRYPTO_AVAILABLE:
        raise CryptoNotAvailableError(
            "Cryptography library not installed. "
            "Install with: pip install cryptography"
        )


def generate_key_pair() -> KeyPair:
    """
    Generate a new Ed25519 key pair.

    Returns:
        KeyPair with private and public keys

    Raises:
        CryptoNotAvailableError: If cryptography library not installed
    """
    check_crypto_available()

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Generate key ID from public key hash
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_id = hashlib.sha256(public_bytes).hexdigest()[:8]

    return KeyPair(
        private_key=private_key,
        public_key=public_key,
        key_id=key_id,
        created_at=datetime.now().isoformat(),
    )


def serialize_private_key(
    private_key: Any,
    password: Optional[str] = None,
) -> bytes:
    """
    Serialize private key to PEM format.

    Args:
        private_key: Ed25519PrivateKey to serialize
        password: Optional password for encryption

    Returns:
        PEM-encoded private key bytes
    """
    check_crypto_available()

    if password:
        encryption = serialization.BestAvailableEncryption(password.encode())
    else:
        encryption = serialization.NoEncryption()

    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )


def serialize_public_key(public_key: Any) -> bytes:
    """
    Serialize public key to PEM format.

    Args:
        public_key: Ed25519PublicKey to serialize

    Returns:
        PEM-encoded public key bytes
    """
    check_crypto_available()

    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def load_private_key(
    pem_data: bytes,
    password: Optional[str] = None,
) -> Any:
    """
    Load private key from PEM data.

    Args:
        pem_data: PEM-encoded private key
        password: Password if key is encrypted

    Returns:
        Ed25519PrivateKey

    Raises:
        CryptoError: If key cannot be loaded
    """
    check_crypto_available()

    try:
        pwd = password.encode() if password else None
        return serialization.load_pem_private_key(pem_data, password=pwd)
    except Exception as e:
        raise CryptoError(f"Failed to load private key: {e}")


def load_public_key(pem_data: bytes) -> Any:
    """
    Load public key from PEM data.

    Args:
        pem_data: PEM-encoded public key

    Returns:
        Ed25519PublicKey

    Raises:
        CryptoError: If key cannot be loaded
    """
    check_crypto_available()

    try:
        return serialization.load_pem_public_key(pem_data)
    except Exception as e:
        raise CryptoError(f"Failed to load public key: {e}")


def compute_canonical_hash(data: Dict[str, Any]) -> bytes:
    """
    Compute canonical hash of data for signing.

    Uses sorted keys and compact JSON for deterministic hashing.

    Args:
        data: Dictionary to hash

    Returns:
        SHA-256 hash bytes
    """
    canonical_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode()).digest()


def sign_data(
    data: Dict[str, Any],
    private_key: Any,
    key_id: str,
) -> Signature:
    """
    Sign data with Ed25519 private key.

    Args:
        data: Dictionary to sign
        private_key: Ed25519PrivateKey for signing
        key_id: Identifier for the signing key

    Returns:
        Signature object

    Raises:
        CryptoError: If signing fails
    """
    check_crypto_available()

    try:
        data_hash = compute_canonical_hash(data)
        signature_bytes = private_key.sign(data_hash)

        return Signature(
            signature=signature_bytes,
            key_id=key_id,
            algorithm="Ed25519",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise CryptoError(f"Signing failed: {e}")


def verify_signature(
    data: Dict[str, Any],
    signature: Signature,
    public_key: Any,
) -> bool:
    """
    Verify Ed25519 signature on data.

    Args:
        data: Dictionary that was signed
        signature: Signature to verify
        public_key: Ed25519PublicKey for verification

    Returns:
        True if signature is valid

    Raises:
        SignatureError: If signature is invalid
    """
    check_crypto_available()

    try:
        data_hash = compute_canonical_hash(data)
        public_key.verify(signature.signature, data_hash)
        return True
    except InvalidSignature:
        raise SignatureError("Invalid signature")
    except Exception as e:
        raise CryptoError(f"Verification failed: {e}")


class KeyManager:
    """
    Manages cryptographic keys for an IntentLog project.

    Keys are stored in .intentlog/keys/ directory:
        .intentlog/keys/
        ├── default.key      # Private key (encrypted or not)
        ├── default.pub      # Public key
        └── keys.json        # Key metadata
    """

    def __init__(self, intentlog_dir: Path):
        """
        Initialize key manager.

        Args:
            intentlog_dir: Path to .intentlog directory
        """
        self.intentlog_dir = Path(intentlog_dir)
        self.keys_dir = self.intentlog_dir / "keys"
        self.metadata_file = self.keys_dir / "keys.json"

    def _ensure_keys_dir(self) -> None:
        """Create keys directory if it doesn't exist"""
        self.keys_dir.mkdir(exist_ok=True)

        # Add to .gitignore if not already there
        gitignore = self.intentlog_dir / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            if "keys/*.key" not in content:
                with open(gitignore, "a") as f:
                    f.write("\n# Private keys - never commit!\nkeys/*.key\n")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load key metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"keys": {}, "default_key": None}

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save key metadata"""
        self._ensure_keys_dir()
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def generate_key(
        self,
        name: str = "default",
        password: Optional[str] = None,
        set_default: bool = True,
    ) -> KeyPair:
        """
        Generate and store a new key pair.

        Args:
            name: Name for the key pair
            password: Optional password for private key encryption
            set_default: If True, set as default signing key

        Returns:
            Generated KeyPair
        """
        check_crypto_available()
        self._ensure_keys_dir()

        # Generate key pair
        key_pair = generate_key_pair()

        # Save private key
        private_pem = serialize_private_key(key_pair.private_key, password)
        private_path = self.keys_dir / f"{name}.key"
        with open(private_path, "wb") as f:
            f.write(private_pem)
        # Set restrictive permissions on private key
        os.chmod(private_path, 0o600)

        # Save public key
        public_pem = serialize_public_key(key_pair.public_key)
        public_path = self.keys_dir / f"{name}.pub"
        with open(public_path, "wb") as f:
            f.write(public_pem)

        # Update metadata
        metadata = self._load_metadata()
        metadata["keys"][name] = {
            "key_id": key_pair.key_id,
            "created_at": key_pair.created_at,
            "algorithm": "Ed25519",
            "encrypted": password is not None,
        }
        if set_default or metadata["default_key"] is None:
            metadata["default_key"] = name
        self._save_metadata(metadata)

        return key_pair

    def load_key(
        self,
        name: str = "default",
        password: Optional[str] = None,
    ) -> KeyPair:
        """
        Load a key pair by name.

        Args:
            name: Name of the key pair
            password: Password if private key is encrypted

        Returns:
            KeyPair with loaded keys

        Raises:
            KeyNotFoundError: If key doesn't exist
        """
        check_crypto_available()

        private_path = self.keys_dir / f"{name}.key"
        public_path = self.keys_dir / f"{name}.pub"

        if not private_path.exists():
            raise KeyNotFoundError(f"Key '{name}' not found")

        # Load metadata
        metadata = self._load_metadata()
        key_meta = metadata.get("keys", {}).get(name, {})

        # Load keys
        with open(private_path, "rb") as f:
            private_key = load_private_key(f.read(), password)

        with open(public_path, "rb") as f:
            public_key = load_public_key(f.read())

        return KeyPair(
            private_key=private_key,
            public_key=public_key,
            key_id=key_meta.get("key_id", name),
            created_at=key_meta.get("created_at", ""),
        )

    def load_public_key(self, name: str = "default") -> Tuple[Any, str]:
        """
        Load only the public key (for verification).

        Args:
            name: Name of the key pair

        Returns:
            Tuple of (public_key, key_id)

        Raises:
            KeyNotFoundError: If key doesn't exist
        """
        check_crypto_available()

        public_path = self.keys_dir / f"{name}.pub"

        if not public_path.exists():
            raise KeyNotFoundError(f"Public key '{name}' not found")

        metadata = self._load_metadata()
        key_meta = metadata.get("keys", {}).get(name, {})

        with open(public_path, "rb") as f:
            public_key = load_public_key(f.read())

        return public_key, key_meta.get("key_id", name)

    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available keys.

        Returns:
            Dictionary of key names to metadata
        """
        metadata = self._load_metadata()
        return metadata.get("keys", {})

    def get_default_key_name(self) -> Optional[str]:
        """Get the name of the default signing key"""
        metadata = self._load_metadata()
        return metadata.get("default_key")

    def set_default_key(self, name: str) -> None:
        """Set the default signing key"""
        metadata = self._load_metadata()
        if name not in metadata.get("keys", {}):
            raise KeyNotFoundError(f"Key '{name}' not found")
        metadata["default_key"] = name
        self._save_metadata(metadata)

    def export_public_key(self, name: str = "default") -> str:
        """
        Export public key as PEM string.

        Args:
            name: Name of the key pair

        Returns:
            PEM-encoded public key string
        """
        public_path = self.keys_dir / f"{name}.pub"

        if not public_path.exists():
            raise KeyNotFoundError(f"Public key '{name}' not found")

        with open(public_path, "rb") as f:
            return f.read().decode()

    def has_keys(self) -> bool:
        """Check if any keys are configured"""
        metadata = self._load_metadata()
        return bool(metadata.get("keys"))


def sign_intent(
    intent_data: Dict[str, Any],
    key_manager: KeyManager,
    key_name: Optional[str] = None,
    password: Optional[str] = None,
) -> Signature:
    """
    Sign an intent using the project's keys.

    Args:
        intent_data: Intent data dictionary to sign
        key_manager: KeyManager instance
        key_name: Key to use (None = default)
        password: Password if key is encrypted

    Returns:
        Signature object
    """
    key_name = key_name or key_manager.get_default_key_name()
    if not key_name:
        raise KeyNotFoundError("No signing key available")

    key_pair = key_manager.load_key(key_name, password)
    return sign_data(intent_data, key_pair.private_key, key_pair.key_id)


def verify_intent_signature(
    intent_data: Dict[str, Any],
    signature: Signature,
    key_manager: KeyManager,
    key_name: Optional[str] = None,
) -> bool:
    """
    Verify an intent's signature.

    Args:
        intent_data: Intent data dictionary that was signed
        signature: Signature to verify
        key_manager: KeyManager instance
        key_name: Key that signed (None = search by key_id)

    Returns:
        True if valid

    Raises:
        SignatureError: If invalid
        KeyNotFoundError: If key not found
    """
    # If key_name not provided, try to find by key_id
    if not key_name:
        keys = key_manager.list_keys()
        for name, meta in keys.items():
            if meta.get("key_id") == signature.key_id:
                key_name = name
                break

    if not key_name:
        raise KeyNotFoundError(f"Key with ID '{signature.key_id}' not found")

    public_key, _ = key_manager.load_public_key(key_name)
    return verify_signature(intent_data, signature, public_key)
