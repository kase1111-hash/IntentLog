"""
Privacy Controls for IntentLog

Implements MP-02 Section 12 privacy requirements:
- Raw signals MAY be encrypted or access-controlled
- Receipts MUST NOT expose raw content by default
- Humans MAY revoke future observation
- Past receipts remain immutable

Features:
- Symmetric encryption (Fernet/AES-128-CBC)
- Access control levels and access lists
- Revocation support for future access
- Encryption key management
"""

import json
import os
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import base64

# Use cryptography library for Fernet encryption
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception


class PrivacyError(Exception):
    """Base exception for privacy-related errors"""
    pass


class EncryptionError(PrivacyError):
    """Raised when encryption/decryption fails"""
    pass


class AccessDeniedError(PrivacyError):
    """Raised when access is denied"""
    pass


class RevocationError(PrivacyError):
    """Raised when revocation fails"""
    pass


class KeyManagementError(PrivacyError):
    """Raised for key management issues"""
    pass


def check_encryption_available():
    """Check if encryption is available, raise if not."""
    if not ENCRYPTION_AVAILABLE:
        raise EncryptionError(
            "Encryption not available. Install 'cryptography' package: "
            "pip install cryptography"
        )


class PrivacyLevel(Enum):
    """Access control levels for intent content."""
    PUBLIC = "public"           # Anyone can read
    INTERNAL = "internal"       # Authenticated users only
    CONFIDENTIAL = "confidential"  # Specific access list only
    SECRET = "secret"           # Encrypted, specific access list
    TOP_SECRET = "top_secret"   # Encrypted, hardware key required


@dataclass
class AccessPolicy:
    """
    Access control policy for an intent or collection.

    Defines who can read, write, and manage access to content.
    """
    level: PrivacyLevel = PrivacyLevel.INTERNAL
    owner_id: Optional[str] = None
    read_access: Set[str] = field(default_factory=set)  # User/role IDs
    write_access: Set[str] = field(default_factory=set)
    admin_access: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None

    def can_read(self, user_id: str) -> bool:
        """Check if user can read content."""
        if self.revoked:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        if self.level == PrivacyLevel.PUBLIC:
            return True
        if user_id == self.owner_id:
            return True
        if user_id in self.admin_access:
            return True
        if user_id in self.read_access:
            return True
        return False

    def can_write(self, user_id: str) -> bool:
        """Check if user can write/modify content."""
        if self.revoked:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        if user_id == self.owner_id:
            return True
        if user_id in self.admin_access:
            return True
        if user_id in self.write_access:
            return True
        return False

    def can_admin(self, user_id: str) -> bool:
        """Check if user can manage access."""
        if self.revoked:
            return False
        if user_id == self.owner_id:
            return True
        if user_id in self.admin_access:
            return True
        return False

    def grant_read(self, user_id: str) -> None:
        """Grant read access to user."""
        self.read_access.add(user_id)

    def grant_write(self, user_id: str) -> None:
        """Grant write access to user."""
        self.write_access.add(user_id)

    def grant_admin(self, user_id: str) -> None:
        """Grant admin access to user."""
        self.admin_access.add(user_id)

    def revoke_user(self, user_id: str) -> None:
        """Revoke all access for a user."""
        self.read_access.discard(user_id)
        self.write_access.discard(user_id)
        self.admin_access.discard(user_id)

    def revoke_all(self, revoked_by: str) -> None:
        """Revoke all future access (past receipts remain)."""
        self.revoked = True
        self.revoked_at = datetime.now()
        self.revoked_by = revoked_by

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "level": self.level.value,
            "owner_id": self.owner_id,
            "read_access": list(self.read_access),
            "write_access": list(self.write_access),
            "admin_access": list(self.admin_access),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_by": self.revoked_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccessPolicy":
        """Deserialize from dictionary."""
        return cls(
            level=PrivacyLevel(data["level"]),
            owner_id=data.get("owner_id"),
            read_access=set(data.get("read_access", [])),
            write_access=set(data.get("write_access", [])),
            admin_access=set(data.get("admin_access", [])),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            revoked=data.get("revoked", False),
            revoked_at=datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None,
            revoked_by=data.get("revoked_by"),
        )


@dataclass
class EncryptionKey:
    """
    Encryption key for protecting intent content.

    Uses Fernet (AES-128-CBC with HMAC) for symmetric encryption.
    """
    key_id: str
    key_bytes: bytes  # 32 bytes for Fernet
    created_at: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None
    expires_at: Optional[datetime] = None
    revoked: bool = False

    @classmethod
    def generate(cls, name: Optional[str] = None) -> "EncryptionKey":
        """Generate a new encryption key."""
        check_encryption_available()
        key_bytes = Fernet.generate_key()
        key_id = hashlib.sha256(key_bytes).hexdigest()[:16]
        return cls(
            key_id=key_id,
            key_bytes=key_bytes,
            name=name,
        )

    @classmethod
    def from_password(
        cls,
        password: str,
        salt: Optional[bytes] = None,
        name: Optional[str] = None,
    ) -> Tuple["EncryptionKey", bytes]:
        """
        Derive encryption key from password.

        Args:
            password: User password
            salt: Optional salt (generated if not provided)
            name: Optional key name

        Returns:
            Tuple of (EncryptionKey, salt)
        """
        check_encryption_available()
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key_bytes = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        key_id = hashlib.sha256(key_bytes).hexdigest()[:16]

        return cls(
            key_id=key_id,
            key_bytes=key_bytes,
            name=name,
        ), salt

    def get_fernet(self) -> "Fernet":
        """Get Fernet instance for encryption/decryption."""
        check_encryption_available()
        return Fernet(self.key_bytes)

    def is_valid(self) -> bool:
        """Check if key is still valid."""
        if self.revoked:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def export_public(self) -> Dict[str, Any]:
        """Export key metadata (without the key bytes)."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked": self.revoked,
        }


@dataclass
class EncryptedContent:
    """
    Encrypted content wrapper.

    Stores encrypted data with metadata needed for decryption.
    """
    ciphertext: bytes
    key_id: str
    algorithm: str = "fernet"  # AES-128-CBC with HMAC
    encrypted_at: datetime = field(default_factory=datetime.now)
    content_hash: Optional[str] = None  # Hash of plaintext for verification

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (base64 encoded ciphertext)."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "encrypted_at": self.encrypted_at.isoformat(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedContent":
        """Deserialize from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            key_id=data["key_id"],
            algorithm=data.get("algorithm", "fernet"),
            encrypted_at=datetime.fromisoformat(data["encrypted_at"]),
            content_hash=data.get("content_hash"),
        )


class IntentEncryptor:
    """
    Encrypts and decrypts intent content.

    Handles encryption of reasoning, metadata, and attachments
    while preserving intent structure for indexing.
    """

    def __init__(self, key: EncryptionKey):
        """Initialize with encryption key."""
        self.key = key
        self._fernet = key.get_fernet()

    def encrypt_string(self, plaintext: str) -> EncryptedContent:
        """
        Encrypt a string.

        Args:
            plaintext: String to encrypt

        Returns:
            EncryptedContent with ciphertext
        """
        check_encryption_available()
        if not self.key.is_valid():
            raise EncryptionError("Encryption key is not valid")

        plaintext_bytes = plaintext.encode("utf-8")
        content_hash = hashlib.sha256(plaintext_bytes).hexdigest()
        ciphertext = self._fernet.encrypt(plaintext_bytes)

        return EncryptedContent(
            ciphertext=ciphertext,
            key_id=self.key.key_id,
            content_hash=content_hash,
        )

    def decrypt_string(self, encrypted: EncryptedContent) -> str:
        """
        Decrypt encrypted content to string.

        Args:
            encrypted: EncryptedContent to decrypt

        Returns:
            Decrypted string

        Raises:
            EncryptionError: If decryption fails or hash mismatch
        """
        check_encryption_available()
        if encrypted.key_id != self.key.key_id:
            raise EncryptionError(
                f"Key mismatch: content encrypted with {encrypted.key_id}, "
                f"but using key {self.key.key_id}"
            )

        try:
            plaintext_bytes = self._fernet.decrypt(encrypted.ciphertext)
        except InvalidToken:
            raise EncryptionError("Decryption failed: invalid key or corrupted data")

        # Verify hash if present
        if encrypted.content_hash:
            actual_hash = hashlib.sha256(plaintext_bytes).hexdigest()
            if actual_hash != encrypted.content_hash:
                raise EncryptionError("Content hash mismatch: data may be corrupted")

        return plaintext_bytes.decode("utf-8")

    def encrypt_dict(self, data: Dict[str, Any]) -> EncryptedContent:
        """Encrypt a dictionary as JSON."""
        json_str = json.dumps(data, separators=(",", ":"), default=str)
        return self.encrypt_string(json_str)

    def decrypt_dict(self, encrypted: EncryptedContent) -> Dict[str, Any]:
        """Decrypt to dictionary."""
        json_str = self.decrypt_string(encrypted)
        return json.loads(json_str)

    def encrypt_intent_fields(
        self,
        intent_dict: Dict[str, Any],
        fields: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt specific fields in an intent dictionary.

        By default encrypts: reasoning, metadata
        Preserves: intent_id, intent_name, timestamp (for indexing)

        Args:
            intent_dict: Intent as dictionary
            fields: Fields to encrypt (default: reasoning, metadata)

        Returns:
            Intent dict with encrypted fields
        """
        if fields is None:
            fields = ["intent_reasoning", "reasoning", "metadata"]

        result = intent_dict.copy()
        encrypted_fields = {}

        for field_name in fields:
            if field_name in result and result[field_name]:
                value = result[field_name]
                if isinstance(value, dict):
                    encrypted = self.encrypt_dict(value)
                else:
                    encrypted = self.encrypt_string(str(value))
                encrypted_fields[field_name] = encrypted.to_dict()
                result[field_name] = f"[ENCRYPTED:{self.key.key_id}]"

        result["_encrypted_fields"] = encrypted_fields
        result["_encryption_key_id"] = self.key.key_id

        return result

    def decrypt_intent_fields(
        self,
        intent_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in an intent dictionary.

        Args:
            intent_dict: Intent dict with encrypted fields

        Returns:
            Intent dict with decrypted fields
        """
        if "_encrypted_fields" not in intent_dict:
            return intent_dict

        result = intent_dict.copy()
        encrypted_fields = result.pop("_encrypted_fields", {})
        result.pop("_encryption_key_id", None)

        for field_name, encrypted_data in encrypted_fields.items():
            encrypted = EncryptedContent.from_dict(encrypted_data)

            # Try to decrypt as dict first, fall back to string
            try:
                result[field_name] = self.decrypt_dict(encrypted)
            except json.JSONDecodeError:
                result[field_name] = self.decrypt_string(encrypted)

        return result


class EncryptionKeyManager:
    """
    Manages encryption keys for privacy controls.

    Stores keys in .intentlog/keys/encryption/
    """

    def __init__(self, project_root: Path):
        """Initialize key manager."""
        self.project_root = project_root
        self.keys_dir = project_root / ".intentlog" / "keys" / "encryption"
        self._keys: Dict[str, EncryptionKey] = {}
        self._default_key: Optional[str] = None

    def _ensure_dir(self) -> None:
        """Ensure keys directory exists."""
        self.keys_dir.mkdir(parents=True, exist_ok=True)

    def generate_key(
        self,
        name: Optional[str] = None,
        set_default: bool = True,
    ) -> EncryptionKey:
        """
        Generate a new encryption key.

        Args:
            name: Optional key name
            set_default: Set as default key

        Returns:
            Generated EncryptionKey
        """
        check_encryption_available()
        self._ensure_dir()

        key = EncryptionKey.generate(name=name)
        self._save_key(key)
        self._keys[key.key_id] = key

        if set_default or self._default_key is None:
            self._default_key = key.key_id
            self._save_default()

        return key

    def generate_from_password(
        self,
        password: str,
        name: Optional[str] = None,
        set_default: bool = True,
    ) -> EncryptionKey:
        """
        Generate encryption key from password.

        Args:
            password: User password
            name: Optional key name
            set_default: Set as default key

        Returns:
            Generated EncryptionKey
        """
        check_encryption_available()
        self._ensure_dir()

        key, salt = EncryptionKey.from_password(password, name=name)
        self._save_key(key, salt=salt)
        self._keys[key.key_id] = key

        if set_default or self._default_key is None:
            self._default_key = key.key_id
            self._save_default()

        return key

    def _save_key(self, key: EncryptionKey, salt: Optional[bytes] = None) -> None:
        """Save key to disk."""
        key_file = self.keys_dir / f"{key.key_id}.key"
        meta_file = self.keys_dir / f"{key.key_id}.meta"

        # Save key bytes (in production, use secure storage)
        key_file.write_bytes(key.key_bytes)
        key_file.chmod(0o600)  # Owner read/write only

        # Save metadata
        meta = {
            "key_id": key.key_id,
            "name": key.name,
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "revoked": key.revoked,
        }
        if salt:
            meta["salt"] = base64.b64encode(salt).decode()

        meta_file.write_text(json.dumps(meta, indent=2))

    def _load_key(self, key_id: str) -> EncryptionKey:
        """Load key from disk."""
        key_file = self.keys_dir / f"{key_id}.key"
        meta_file = self.keys_dir / f"{key_id}.meta"

        if not key_file.exists():
            raise KeyManagementError(f"Key not found: {key_id}")

        key_bytes = key_file.read_bytes()

        meta = {}
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())

        return EncryptionKey(
            key_id=key_id,
            key_bytes=key_bytes,
            name=meta.get("name"),
            created_at=datetime.fromisoformat(meta["created_at"]) if "created_at" in meta else datetime.now(),
            expires_at=datetime.fromisoformat(meta["expires_at"]) if meta.get("expires_at") else None,
            revoked=meta.get("revoked", False),
        )

    def _save_default(self) -> None:
        """Save default key ID."""
        default_file = self.keys_dir / "default"
        if self._default_key:
            default_file.write_text(self._default_key)

    def _load_default(self) -> Optional[str]:
        """Load default key ID."""
        default_file = self.keys_dir / "default"
        if default_file.exists():
            return default_file.read_text().strip()
        return None

    def get_key(self, key_id: str) -> EncryptionKey:
        """Get key by ID."""
        if key_id not in self._keys:
            self._keys[key_id] = self._load_key(key_id)
        return self._keys[key_id]

    def get_default_key(self) -> Optional[EncryptionKey]:
        """Get default encryption key."""
        if self._default_key is None:
            self._default_key = self._load_default()
        if self._default_key:
            return self.get_key(self._default_key)
        return None

    def set_default_key(self, key_id: str) -> None:
        """Set default encryption key."""
        # Verify key exists
        self.get_key(key_id)
        self._default_key = key_id
        self._save_default()

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all encryption keys (metadata only)."""
        keys = []
        if self.keys_dir.exists():
            for meta_file in self.keys_dir.glob("*.meta"):
                meta = json.loads(meta_file.read_text())
                meta["is_default"] = meta["key_id"] == self._default_key
                keys.append(meta)
        return sorted(keys, key=lambda x: x.get("created_at", ""))

    def revoke_key(self, key_id: str) -> None:
        """Revoke an encryption key."""
        key = self.get_key(key_id)
        key.revoked = True
        self._save_key(key)

    def delete_key(self, key_id: str) -> bool:
        """
        Delete an encryption key.

        WARNING: Data encrypted with this key will be unrecoverable!

        Returns:
            True if deleted, False if not found
        """
        key_file = self.keys_dir / f"{key_id}.key"
        meta_file = self.keys_dir / f"{key_id}.meta"

        deleted = False
        if key_file.exists():
            key_file.unlink()
            deleted = True
        if meta_file.exists():
            meta_file.unlink()
            deleted = True

        self._keys.pop(key_id, None)

        if self._default_key == key_id:
            self._default_key = None
            default_file = self.keys_dir / "default"
            if default_file.exists():
                default_file.unlink()

        return deleted


@dataclass
class RevocationRecord:
    """
    Record of access revocation.

    Per MP-02 Section 12: Humans MAY revoke future observation.
    Past receipts remain immutable.
    """
    record_id: str
    target_type: str  # "intent", "session", "user", "all"
    target_id: Optional[str]  # ID of revoked target
    revoked_by: str
    revoked_at: datetime = field(default_factory=datetime.now)
    reason: Optional[str] = None
    scope: str = "future"  # "future" (default per MP-02) or "immediate"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "revoked_by": self.revoked_by,
            "revoked_at": self.revoked_at.isoformat(),
            "reason": self.reason,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RevocationRecord":
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            target_type=data["target_type"],
            target_id=data.get("target_id"),
            revoked_by=data["revoked_by"],
            revoked_at=datetime.fromisoformat(data["revoked_at"]),
            reason=data.get("reason"),
            scope=data.get("scope", "future"),
        )


class RevocationManager:
    """
    Manages access revocations per MP-02 Section 12.

    Stores revocations in .intentlog/revocations/
    """

    def __init__(self, project_root: Path):
        """Initialize revocation manager."""
        self.project_root = project_root
        self.revocations_dir = project_root / ".intentlog" / "revocations"
        self._revocations: Dict[str, RevocationRecord] = {}

    def _ensure_dir(self) -> None:
        """Ensure revocations directory exists."""
        self.revocations_dir.mkdir(parents=True, exist_ok=True)

    def revoke_observation(
        self,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> RevocationRecord:
        """
        Revoke all future observation (MP-02 Section 12).

        Past receipts remain immutable.

        Args:
            revoked_by: User ID revoking observation
            reason: Optional reason for revocation

        Returns:
            RevocationRecord
        """
        self._ensure_dir()

        record = RevocationRecord(
            record_id=secrets.token_hex(8),
            target_type="all",
            target_id=None,
            revoked_by=revoked_by,
            reason=reason,
            scope="future",
        )

        self._save_revocation(record)
        return record

    def revoke_intent(
        self,
        intent_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> RevocationRecord:
        """Revoke future access to a specific intent."""
        self._ensure_dir()

        record = RevocationRecord(
            record_id=secrets.token_hex(8),
            target_type="intent",
            target_id=intent_id,
            revoked_by=revoked_by,
            reason=reason,
        )

        self._save_revocation(record)
        return record

    def revoke_session(
        self,
        session_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> RevocationRecord:
        """Revoke future access to a session's intents."""
        self._ensure_dir()

        record = RevocationRecord(
            record_id=secrets.token_hex(8),
            target_type="session",
            target_id=session_id,
            revoked_by=revoked_by,
            reason=reason,
        )

        self._save_revocation(record)
        return record

    def revoke_user_access(
        self,
        user_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> RevocationRecord:
        """Revoke a user's future access."""
        self._ensure_dir()

        record = RevocationRecord(
            record_id=secrets.token_hex(8),
            target_type="user",
            target_id=user_id,
            revoked_by=revoked_by,
            reason=reason,
        )

        self._save_revocation(record)
        return record

    def _save_revocation(self, record: RevocationRecord) -> None:
        """Save revocation record."""
        file_path = self.revocations_dir / f"{record.record_id}.json"
        file_path.write_text(json.dumps(record.to_dict(), indent=2))
        self._revocations[record.record_id] = record

    def _load_revocations(self) -> None:
        """Load all revocation records."""
        if not self.revocations_dir.exists():
            return

        for file_path in self.revocations_dir.glob("*.json"):
            data = json.loads(file_path.read_text())
            record = RevocationRecord.from_dict(data)
            self._revocations[record.record_id] = record

    def is_revoked(
        self,
        intent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Check if access is revoked.

        Args:
            intent_id: Intent to check
            session_id: Session to check
            user_id: User to check

        Returns:
            True if revoked
        """
        if not self._revocations:
            self._load_revocations()

        for record in self._revocations.values():
            if record.target_type == "all":
                return True
            if record.target_type == "intent" and record.target_id == intent_id:
                return True
            if record.target_type == "session" and record.target_id == session_id:
                return True
            if record.target_type == "user" and record.target_id == user_id:
                return True

        return False

    def list_revocations(self) -> List[RevocationRecord]:
        """List all revocation records."""
        if not self._revocations:
            self._load_revocations()
        return list(self._revocations.values())

    def get_revocation(self, record_id: str) -> Optional[RevocationRecord]:
        """Get revocation record by ID."""
        if not self._revocations:
            self._load_revocations()
        return self._revocations.get(record_id)


class PrivacyManager:
    """
    High-level privacy manager combining encryption, access control, and revocation.

    Implements MP-02 Section 12 requirements.
    """

    def __init__(self, project_root: Path):
        """Initialize privacy manager."""
        self.project_root = project_root
        self.key_manager = EncryptionKeyManager(project_root)
        self.revocation_manager = RevocationManager(project_root)
        self._policies: Dict[str, AccessPolicy] = {}
        self._policies_dir = project_root / ".intentlog" / "policies"

    def _ensure_policies_dir(self) -> None:
        """Ensure policies directory exists."""
        self._policies_dir.mkdir(parents=True, exist_ok=True)

    def set_policy(self, target_id: str, policy: AccessPolicy) -> None:
        """Set access policy for a target."""
        self._ensure_policies_dir()
        file_path = self._policies_dir / f"{target_id}.json"
        file_path.write_text(json.dumps(policy.to_dict(), indent=2))
        self._policies[target_id] = policy

    def get_policy(self, target_id: str) -> Optional[AccessPolicy]:
        """Get access policy for a target."""
        if target_id in self._policies:
            return self._policies[target_id]

        file_path = self._policies_dir / f"{target_id}.json"
        if file_path.exists():
            data = json.loads(file_path.read_text())
            policy = AccessPolicy.from_dict(data)
            self._policies[target_id] = policy
            return policy

        return None

    def encrypt_intent(
        self,
        intent_dict: Dict[str, Any],
        key: Optional[EncryptionKey] = None,
        fields: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt an intent's sensitive fields.

        Args:
            intent_dict: Intent as dictionary
            key: Encryption key (uses default if not provided)
            fields: Fields to encrypt

        Returns:
            Intent dict with encrypted fields
        """
        if key is None:
            key = self.key_manager.get_default_key()
            if key is None:
                key = self.key_manager.generate_key()

        encryptor = IntentEncryptor(key)
        return encryptor.encrypt_intent_fields(intent_dict, fields)

    def decrypt_intent(
        self,
        intent_dict: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decrypt an intent's encrypted fields.

        Args:
            intent_dict: Intent with encrypted fields
            user_id: User requesting decryption (for access check)

        Returns:
            Intent dict with decrypted fields

        Raises:
            AccessDeniedError: If access is denied
            EncryptionError: If decryption fails
        """
        if "_encryption_key_id" not in intent_dict:
            return intent_dict  # Not encrypted

        # Check revocation
        intent_id = intent_dict.get("intent_id")
        session_id = intent_dict.get("session_id")
        if self.revocation_manager.is_revoked(intent_id=intent_id, session_id=session_id, user_id=user_id):
            raise AccessDeniedError("Access has been revoked")

        # Check access policy if present
        if intent_id and user_id:
            policy = self.get_policy(intent_id)
            if policy and not policy.can_read(user_id):
                raise AccessDeniedError(f"User {user_id} does not have read access")

        # Decrypt
        key_id = intent_dict["_encryption_key_id"]
        key = self.key_manager.get_key(key_id)
        encryptor = IntentEncryptor(key)
        return encryptor.decrypt_intent_fields(intent_dict)

    def check_access(
        self,
        target_id: str,
        user_id: str,
        action: str = "read",
    ) -> bool:
        """
        Check if user has access to perform action.

        Args:
            target_id: Intent or resource ID
            user_id: User ID
            action: "read", "write", or "admin"

        Returns:
            True if access allowed
        """
        # Check revocation first
        if self.revocation_manager.is_revoked(intent_id=target_id, user_id=user_id):
            return False

        # Check policy
        policy = self.get_policy(target_id)
        if policy is None:
            return True  # No policy = public access

        if action == "read":
            return policy.can_read(user_id)
        elif action == "write":
            return policy.can_write(user_id)
        elif action == "admin":
            return policy.can_admin(user_id)

        return False

    def revoke_future_observation(self, user_id: str, reason: Optional[str] = None) -> RevocationRecord:
        """
        Revoke all future observation per MP-02 Section 12.

        Past receipts remain immutable.
        """
        return self.revocation_manager.revoke_observation(user_id, reason)
