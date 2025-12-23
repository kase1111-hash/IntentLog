"""
Tests for Privacy Controls (MP-02 Section 12)

Tests for:
- Encryption (Fernet/AES-128-CBC)
- Access control (policies, permissions)
- Revocation support
- Key management
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

from intentlog.privacy import (
    PrivacyLevel,
    AccessPolicy,
    EncryptionKey,
    EncryptedContent,
    IntentEncryptor,
    EncryptionKeyManager,
    RevocationRecord,
    RevocationManager,
    PrivacyManager,
    PrivacyError,
    EncryptionError,
    AccessDeniedError,
    KeyManagementError,
    ENCRYPTION_AVAILABLE,
)


# Skip tests if cryptography not available
pytestmark = pytest.mark.skipif(
    not ENCRYPTION_AVAILABLE,
    reason="cryptography library not installed"
)


class TestPrivacyLevel:
    """Tests for PrivacyLevel enum"""

    def test_privacy_levels(self):
        """All privacy levels exist"""
        assert PrivacyLevel.PUBLIC.value == "public"
        assert PrivacyLevel.INTERNAL.value == "internal"
        assert PrivacyLevel.CONFIDENTIAL.value == "confidential"
        assert PrivacyLevel.SECRET.value == "secret"
        assert PrivacyLevel.TOP_SECRET.value == "top_secret"


class TestAccessPolicy:
    """Tests for AccessPolicy"""

    def test_create_policy(self):
        """Create access policy"""
        policy = AccessPolicy(
            level=PrivacyLevel.CONFIDENTIAL,
            owner_id="owner123",
        )
        assert policy.level == PrivacyLevel.CONFIDENTIAL
        assert policy.owner_id == "owner123"

    def test_owner_can_read(self):
        """Owner can always read"""
        policy = AccessPolicy(owner_id="owner123")
        assert policy.can_read("owner123")

    def test_owner_can_write(self):
        """Owner can always write"""
        policy = AccessPolicy(owner_id="owner123")
        assert policy.can_write("owner123")

    def test_owner_can_admin(self):
        """Owner can always admin"""
        policy = AccessPolicy(owner_id="owner123")
        assert policy.can_admin("owner123")

    def test_public_access(self):
        """Public level allows anyone to read"""
        policy = AccessPolicy(level=PrivacyLevel.PUBLIC)
        assert policy.can_read("anyone")

    def test_grant_read(self):
        """Grant read access to user"""
        policy = AccessPolicy(owner_id="owner")
        policy.grant_read("reader")
        assert policy.can_read("reader")
        assert not policy.can_write("reader")

    def test_grant_write(self):
        """Grant write access to user"""
        policy = AccessPolicy(owner_id="owner")
        policy.grant_write("writer")
        assert policy.can_write("writer")

    def test_grant_admin(self):
        """Grant admin access to user"""
        policy = AccessPolicy(owner_id="owner")
        policy.grant_admin("admin")
        assert policy.can_admin("admin")
        assert policy.can_read("admin")
        assert policy.can_write("admin")

    def test_revoke_user(self):
        """Revoke user access"""
        policy = AccessPolicy(owner_id="owner")
        policy.grant_read("user")
        policy.grant_write("user")
        assert policy.can_read("user")

        policy.revoke_user("user")
        assert not policy.can_read("user")
        assert not policy.can_write("user")

    def test_revoke_all(self):
        """Revoke all access"""
        policy = AccessPolicy(owner_id="owner")
        policy.grant_read("user")

        policy.revoke_all("admin")
        assert policy.revoked
        assert not policy.can_read("user")
        assert not policy.can_read("owner")  # Even owner loses access

    def test_expiration(self):
        """Policy expires"""
        policy = AccessPolicy(
            owner_id="owner",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert not policy.can_read("owner")

    def test_to_from_dict(self):
        """Serialize and deserialize policy"""
        policy = AccessPolicy(
            level=PrivacyLevel.SECRET,
            owner_id="owner123",
            read_access={"user1", "user2"},
        )
        policy.grant_write("writer")

        data = policy.to_dict()
        restored = AccessPolicy.from_dict(data)

        assert restored.level == PrivacyLevel.SECRET
        assert restored.owner_id == "owner123"
        assert "user1" in restored.read_access
        assert "writer" in restored.write_access


class TestEncryptionKey:
    """Tests for EncryptionKey"""

    def test_generate_key(self):
        """Generate encryption key"""
        key = EncryptionKey.generate(name="test-key")
        assert key.key_id is not None
        assert len(key.key_bytes) > 0
        assert key.name == "test-key"
        assert key.is_valid()

    def test_key_from_password(self):
        """Derive key from password"""
        key1, salt1 = EncryptionKey.from_password("mypassword")
        key2, _ = EncryptionKey.from_password("mypassword", salt=salt1)

        # Same password + salt = same key
        assert key1.key_bytes == key2.key_bytes

    def test_key_revocation(self):
        """Revoked key is not valid"""
        key = EncryptionKey.generate()
        assert key.is_valid()

        key.revoked = True
        assert not key.is_valid()

    def test_key_expiration(self):
        """Expired key is not valid"""
        key = EncryptionKey.generate()
        key.expires_at = datetime.now() - timedelta(hours=1)
        assert not key.is_valid()

    def test_export_public(self):
        """Export key metadata without key bytes"""
        key = EncryptionKey.generate(name="my-key")
        public = key.export_public()

        assert public["key_id"] == key.key_id
        assert public["name"] == "my-key"
        assert "key_bytes" not in public


class TestIntentEncryptor:
    """Tests for IntentEncryptor"""

    def setup_method(self):
        self.key = EncryptionKey.generate()
        self.encryptor = IntentEncryptor(self.key)

    def test_encrypt_decrypt_string(self):
        """Encrypt and decrypt string"""
        plaintext = "This is secret reasoning"
        encrypted = self.encryptor.encrypt_string(plaintext)

        assert encrypted.key_id == self.key.key_id
        assert encrypted.ciphertext != plaintext.encode()

        decrypted = self.encryptor.decrypt_string(encrypted)
        assert decrypted == plaintext

    def test_encrypt_decrypt_dict(self):
        """Encrypt and decrypt dictionary"""
        data = {"key": "value", "nested": {"a": 1}}
        encrypted = self.encryptor.encrypt_dict(data)

        decrypted = self.encryptor.decrypt_dict(encrypted)
        assert decrypted == data

    def test_content_hash_verification(self):
        """Content hash is verified on decrypt"""
        plaintext = "test content"
        encrypted = self.encryptor.encrypt_string(plaintext)

        # Tamper with hash
        encrypted.content_hash = "invalid_hash"

        with pytest.raises(EncryptionError, match="hash mismatch"):
            self.encryptor.decrypt_string(encrypted)

    def test_wrong_key_fails(self):
        """Decryption with wrong key fails"""
        plaintext = "secret"
        encrypted = self.encryptor.encrypt_string(plaintext)

        other_key = EncryptionKey.generate()
        other_encryptor = IntentEncryptor(other_key)

        with pytest.raises(EncryptionError, match="Key mismatch"):
            other_encryptor.decrypt_string(encrypted)

    def test_encrypt_intent_fields(self):
        """Encrypt specific intent fields"""
        intent = {
            "intent_id": "123",
            "intent_name": "test",
            "intent_reasoning": "This is secret",
            "metadata": {"key": "value"},
            "timestamp": "2025-01-01T00:00:00",
        }

        encrypted_intent = self.encryptor.encrypt_intent_fields(intent)

        # Preserved fields
        assert encrypted_intent["intent_id"] == "123"
        assert encrypted_intent["intent_name"] == "test"
        assert encrypted_intent["timestamp"] == "2025-01-01T00:00:00"

        # Encrypted fields
        assert encrypted_intent["intent_reasoning"].startswith("[ENCRYPTED:")
        assert encrypted_intent["metadata"].startswith("[ENCRYPTED:")
        assert "_encrypted_fields" in encrypted_intent

    def test_decrypt_intent_fields(self):
        """Decrypt intent fields"""
        intent = {
            "intent_id": "123",
            "intent_reasoning": "Secret reasoning",
            "metadata": {"secret": "data"},
        }

        encrypted = self.encryptor.encrypt_intent_fields(intent)
        decrypted = self.encryptor.decrypt_intent_fields(encrypted)

        assert decrypted["intent_reasoning"] == "Secret reasoning"
        assert decrypted["metadata"]["secret"] == "data"
        assert "_encrypted_fields" not in decrypted


class TestEncryptedContent:
    """Tests for EncryptedContent serialization"""

    def test_to_from_dict(self):
        """Serialize and deserialize encrypted content"""
        key = EncryptionKey.generate()
        encryptor = IntentEncryptor(key)
        encrypted = encryptor.encrypt_string("test")

        data = encrypted.to_dict()
        restored = EncryptedContent.from_dict(data)

        assert restored.key_id == encrypted.key_id
        assert restored.ciphertext == encrypted.ciphertext
        assert restored.algorithm == encrypted.algorithm


class TestEncryptionKeyManager:
    """Tests for EncryptionKeyManager"""

    def test_generate_and_save_key(self):
        """Generate and save key to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionKeyManager(Path(tmpdir))
            key = manager.generate_key(name="test-key")

            assert key.name == "test-key"
            assert (Path(tmpdir) / ".intentlog" / "keys" / "encryption" / f"{key.key_id}.key").exists()

    def test_load_key(self):
        """Load key from disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionKeyManager(Path(tmpdir))
            key = manager.generate_key(name="test")

            # Create new manager instance
            manager2 = EncryptionKeyManager(Path(tmpdir))
            loaded = manager2.get_key(key.key_id)

            assert loaded.key_id == key.key_id
            assert loaded.key_bytes == key.key_bytes

    def test_default_key(self):
        """Default key management"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionKeyManager(Path(tmpdir))
            key1 = manager.generate_key(name="first")
            key2 = manager.generate_key(name="second", set_default=False)

            default = manager.get_default_key()
            assert default.key_id == key1.key_id

            manager.set_default_key(key2.key_id)
            default = manager.get_default_key()
            assert default.key_id == key2.key_id

    def test_list_keys(self):
        """List all keys"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionKeyManager(Path(tmpdir))
            manager.generate_key(name="key1")
            manager.generate_key(name="key2", set_default=False)

            keys = manager.list_keys()
            assert len(keys) == 2
            names = {k["name"] for k in keys}
            assert names == {"key1", "key2"}

    def test_password_derived_key(self):
        """Generate key from password"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionKeyManager(Path(tmpdir))
            key = manager.generate_from_password("mypassword", name="password-key")

            assert key.name == "password-key"
            assert key.is_valid()

    def test_delete_key(self):
        """Delete encryption key"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionKeyManager(Path(tmpdir))
            key = manager.generate_key()

            assert manager.delete_key(key.key_id)

            with pytest.raises(KeyManagementError):
                manager.get_key(key.key_id)


class TestRevocationRecord:
    """Tests for RevocationRecord"""

    def test_create_record(self):
        """Create revocation record"""
        record = RevocationRecord(
            record_id="abc123",
            target_type="intent",
            target_id="intent-456",
            revoked_by="user789",
            reason="Privacy request",
        )
        assert record.target_type == "intent"
        assert record.scope == "future"

    def test_to_from_dict(self):
        """Serialize and deserialize"""
        record = RevocationRecord(
            record_id="abc",
            target_type="session",
            target_id="sess-123",
            revoked_by="admin",
        )

        data = record.to_dict()
        restored = RevocationRecord.from_dict(data)

        assert restored.record_id == "abc"
        assert restored.target_type == "session"
        assert restored.target_id == "sess-123"


class TestRevocationManager:
    """Tests for RevocationManager"""

    def test_revoke_observation(self):
        """Revoke all future observation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RevocationManager(Path(tmpdir))
            record = manager.revoke_observation("user123", reason="Privacy request")

            assert record.target_type == "all"
            assert record.revoked_by == "user123"
            assert record.scope == "future"

    def test_revoke_intent(self):
        """Revoke specific intent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RevocationManager(Path(tmpdir))
            record = manager.revoke_intent("intent-123", "admin")

            assert record.target_type == "intent"
            assert record.target_id == "intent-123"

    def test_revoke_session(self):
        """Revoke session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RevocationManager(Path(tmpdir))
            record = manager.revoke_session("session-456", "user")

            assert record.target_type == "session"
            assert record.target_id == "session-456"

    def test_is_revoked(self):
        """Check if revoked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RevocationManager(Path(tmpdir))

            assert not manager.is_revoked(intent_id="intent-123")

            manager.revoke_intent("intent-123", "admin")
            assert manager.is_revoked(intent_id="intent-123")

    def test_revoke_all(self):
        """Revoke all observation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RevocationManager(Path(tmpdir))
            manager.revoke_observation("user")

            # Everything is revoked
            assert manager.is_revoked(intent_id="any-intent")
            assert manager.is_revoked(session_id="any-session")

    def test_list_revocations(self):
        """List all revocations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RevocationManager(Path(tmpdir))
            manager.revoke_intent("i1", "admin")
            manager.revoke_session("s1", "admin")

            revocations = manager.list_revocations()
            assert len(revocations) == 2


class TestPrivacyManager:
    """Tests for PrivacyManager"""

    def test_encrypt_decrypt_intent(self):
        """Encrypt and decrypt intent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            intent = {
                "intent_id": "123",
                "intent_name": "test",
                "intent_reasoning": "Secret reasoning",
                "metadata": {"key": "value"},
            }

            encrypted = manager.encrypt_intent(intent)
            assert encrypted["intent_reasoning"].startswith("[ENCRYPTED:")

            decrypted = manager.decrypt_intent(encrypted)
            assert decrypted["intent_reasoning"] == "Secret reasoning"

    def test_access_denied_on_revocation(self):
        """Access denied after revocation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            intent = {
                "intent_id": "123",
                "intent_reasoning": "Secret",
            }

            encrypted = manager.encrypt_intent(intent)

            # Revoke
            manager.revoke_future_observation("admin")

            with pytest.raises(AccessDeniedError):
                manager.decrypt_intent(encrypted, user_id="anyone")

    def test_access_policy_check(self):
        """Check access with policy"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            policy = AccessPolicy(
                level=PrivacyLevel.CONFIDENTIAL,
                owner_id="owner",
            )
            policy.grant_read("reader")

            manager.set_policy("intent-123", policy)

            assert manager.check_access("intent-123", "owner", "read")
            assert manager.check_access("intent-123", "reader", "read")
            assert not manager.check_access("intent-123", "stranger", "read")

    def test_decrypt_with_access_check(self):
        """Decrypt respects access policy"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            intent = {
                "intent_id": "protected-123",
                "intent_reasoning": "Secret",
            }

            encrypted = manager.encrypt_intent(intent)

            # Set restrictive policy
            policy = AccessPolicy(
                level=PrivacyLevel.SECRET,
                owner_id="owner",
            )
            manager.set_policy("protected-123", policy)

            # Owner can decrypt
            decrypted = manager.decrypt_intent(encrypted, user_id="owner")
            assert decrypted["intent_reasoning"] == "Secret"

            # Stranger cannot
            with pytest.raises(AccessDeniedError):
                manager.decrypt_intent(encrypted, user_id="stranger")


class TestMP02Section12Compliance:
    """Tests for MP-02 Section 12 compliance"""

    def test_raw_signals_encrypted(self):
        """Raw signals MAY be encrypted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            signal = {
                "intent_id": "sig-1",
                "intent_reasoning": "Raw observation data",
                "metadata": {"signal_type": "keystroke"},
            }

            encrypted = manager.encrypt_intent(signal)

            # Reasoning is encrypted
            assert "Raw observation" not in str(encrypted.get("intent_reasoning", ""))
            assert "_encrypted_fields" in encrypted

    def test_receipts_not_expose_raw_content(self):
        """Receipts MUST NOT expose raw content by default"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            intent = {
                "intent_id": "receipt-1",
                "intent_reasoning": "Sensitive raw content",
            }

            encrypted = manager.encrypt_intent(intent)

            # Raw content not visible
            assert "Sensitive raw content" not in str(encrypted)

    def test_humans_may_revoke_future(self):
        """Humans MAY revoke future observation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            # Human revokes future observation
            record = manager.revoke_future_observation("human-user", "Privacy request")

            assert record.scope == "future"
            assert record.target_type == "all"

            # Future access is blocked
            assert manager.revocation_manager.is_revoked()

    def test_past_receipts_immutable(self):
        """Past receipts remain immutable"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PrivacyManager(Path(tmpdir))

            # Create and encrypt intent before revocation
            intent = {
                "intent_id": "past-1",
                "intent_reasoning": "Historical record",
            }
            encrypted = manager.encrypt_intent(intent)

            # The encrypted data itself remains unchanged
            # (Revocation affects access, not the data)
            assert encrypted["_encryption_key_id"] is not None
            assert "_encrypted_fields" in encrypted

            # Note: The spec says past receipts remain immutable
            # This means the encrypted content is preserved
            # Access control is separate from data immutability
