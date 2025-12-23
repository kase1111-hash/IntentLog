"""
Tests for Phase 2: Cryptographic Integrity

Tests for:
- Merkle tree hash chain (merkle.py)
- Ed25519 cryptographic signatures (crypto.py)
- Key management
- Chain verification
- Storage integration
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from intentlog.core import Intent
from intentlog.merkle import (
    ChainedIntent,
    MerkleChain,
    chain_intents,
    append_to_chain,
    verify_chain,
    compute_root_hash,
    compute_intent_content_hash,
    compute_chain_hash,
    find_divergence_point,
    generate_inclusion_proof,
    verify_inclusion_proof,
    ChainVerificationResult,
    GENESIS_HASH,
)

# Only run crypto tests if cryptography is available
try:
    from intentlog.crypto import (
        KeyManager,
        KeyPair,
        Signature,
        generate_key_pair,
        sign_data,
        verify_signature,
        serialize_private_key,
        serialize_public_key,
        load_private_key,
        load_public_key,
        CryptoError,
        KeyNotFoundError,
        SignatureError,
        CRYPTO_AVAILABLE,
    )
    HAS_CRYPTO = CRYPTO_AVAILABLE
except ImportError:
    HAS_CRYPTO = False


# ============================================================================
# Merkle Chain Tests
# ============================================================================

class TestMerkleChainBasics:
    """Basic Merkle chain functionality tests"""

    def test_compute_intent_content_hash(self):
        """Test content hash computation"""
        intent = Intent(
            intent_name="Test intent",
            intent_reasoning="This is a test",
        )
        hash1 = compute_intent_content_hash(intent)

        # Hash should be deterministic
        hash2 = compute_intent_content_hash(intent)
        assert hash1 == hash2

        # Hash should be 64 chars (full SHA-256)
        assert len(hash1) == 64

    def test_compute_intent_content_hash_different_content(self):
        """Different content produces different hashes"""
        intent1 = Intent(intent_name="Test 1", intent_reasoning="Reason 1")
        intent2 = Intent(intent_name="Test 2", intent_reasoning="Reason 2")

        hash1 = compute_intent_content_hash(intent1)
        hash2 = compute_intent_content_hash(intent2)

        assert hash1 != hash2

    def test_compute_chain_hash(self):
        """Test chain hash computation"""
        intent_hash = "a" * 64
        prev_hash = "b" * 64

        chain_hash = compute_chain_hash(intent_hash, prev_hash)

        assert len(chain_hash) == 64
        assert chain_hash != intent_hash
        assert chain_hash != prev_hash

    def test_genesis_hash(self):
        """Genesis hash should be all zeros"""
        assert GENESIS_HASH == "0" * 64


class TestChainIntents:
    """Tests for chaining intents together"""

    def test_chain_empty_list(self):
        """Chaining empty list returns empty list"""
        result = chain_intents([])
        assert result == []

    def test_chain_single_intent(self):
        """Chain single intent"""
        intent = Intent(intent_name="Test", intent_reasoning="Reason")
        chained = chain_intents([intent])

        assert len(chained) == 1
        assert chained[0].intent == intent
        assert chained[0].prev_hash == GENESIS_HASH
        assert chained[0].sequence == 0

    def test_chain_multiple_intents(self):
        """Chain multiple intents"""
        intents = [
            Intent(intent_name=f"Intent {i}", intent_reasoning=f"Reason {i}")
            for i in range(5)
        ]
        chained = chain_intents(intents)

        assert len(chained) == 5

        # Check sequence numbers
        for i, c in enumerate(chained):
            assert c.sequence == i

        # Check chain linking
        assert chained[0].prev_hash == GENESIS_HASH
        for i in range(1, 5):
            assert chained[i].prev_hash == chained[i-1].chain_hash

    def test_append_to_empty_chain(self):
        """Append to empty chain"""
        intent = Intent(intent_name="Test", intent_reasoning="Reason")
        chained = append_to_chain([], intent)

        assert chained.prev_hash == GENESIS_HASH
        assert chained.sequence == 0

    def test_append_to_existing_chain(self):
        """Append to existing chain"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(3)]
        chain = chain_intents(intents)

        new_intent = Intent(intent_name="New", intent_reasoning="New reason")
        new_chained = append_to_chain(chain, new_intent)

        assert new_chained.prev_hash == chain[-1].chain_hash
        assert new_chained.sequence == 3


class TestChainVerification:
    """Tests for chain verification"""

    def test_verify_empty_chain(self):
        """Empty chain is valid"""
        result = verify_chain([])
        assert result.valid
        assert result.entries_checked == 0
        assert result.root_hash == GENESIS_HASH

    def test_verify_valid_chain(self):
        """Valid chain passes verification"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(5)]
        chain = chain_intents(intents)

        result = verify_chain(chain)

        assert result.valid
        assert result.entries_checked == 5
        assert len(result.errors) == 0
        assert result.broken_at is None

    def test_verify_tampered_content(self):
        """Tampered content is detected"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(3)]
        chain = chain_intents(intents)

        # Tamper with content
        chain[1].intent.intent_reasoning = "Tampered!"

        result = verify_chain(chain)

        assert not result.valid
        assert result.broken_at == 1
        assert any("hash mismatch" in e.lower() for e in result.errors)

    def test_verify_broken_chain_link(self):
        """Broken chain link is detected"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(3)]
        chain = chain_intents(intents)

        # Break chain link
        chain[2].prev_hash = "x" * 64

        result = verify_chain(chain)

        assert not result.valid
        assert result.broken_at == 2


class TestRootHash:
    """Tests for root hash computation"""

    def test_root_hash_empty(self):
        """Root hash of empty chain is genesis"""
        assert compute_root_hash([]) == GENESIS_HASH

    def test_root_hash_single(self):
        """Root hash of single intent"""
        intent = Intent(intent_name="Test", intent_reasoning="Reason")
        chain = chain_intents([intent])

        root = compute_root_hash(chain)
        assert root == chain[0].chain_hash

    def test_root_hash_multiple(self):
        """Root hash is last chain hash"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(5)]
        chain = chain_intents(intents)

        root = compute_root_hash(chain)
        assert root == chain[-1].chain_hash


class TestDivergencePoint:
    """Tests for finding chain divergence"""

    def test_identical_chains(self):
        """Identical chains have no divergence"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(3)]
        chain_a = chain_intents(intents)
        chain_b = chain_intents(intents)

        assert find_divergence_point(chain_a, chain_b) is None

    def test_different_length_chains(self):
        """Different length chains diverge at length difference"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(3)]
        chain_a = chain_intents(intents)
        chain_b = chain_intents(intents[:2])

        assert find_divergence_point(chain_a, chain_b) == 2

    def test_divergent_chains(self):
        """Find divergence point"""
        # Create shared intents first
        shared = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(2)]
        chain_a = chain_intents(shared + [Intent(intent_name="A", intent_reasoning="RA")])

        # Chain B shares same base but diverges at position 2
        # We need to manually create chain_b from same shared intents
        chain_b = chain_intents(shared)
        new_chained = append_to_chain(chain_b, Intent(intent_name="B", intent_reasoning="RB"))
        chain_b.append(new_chained)

        # Both chains share first 2 intents, diverge at 2
        assert find_divergence_point(chain_a, chain_b) == 2


class TestInclusionProof:
    """Tests for inclusion proofs"""

    def test_generate_proof(self):
        """Generate inclusion proof"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(5)]
        chain = chain_intents(intents)

        proof = generate_inclusion_proof(chain, 2)

        assert proof["target_sequence"] == 2
        assert proof["root_hash"] == compute_root_hash(chain)
        assert len(proof["proof_path"]) == 3  # From target to end

    def test_verify_valid_proof(self):
        """Verify valid inclusion proof"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(5)]
        chain = chain_intents(intents)

        proof = generate_inclusion_proof(chain, 2)

        assert verify_inclusion_proof(proof)

    def test_verify_invalid_proof(self):
        """Invalid proof fails verification"""
        intents = [Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}") for i in range(5)]
        chain = chain_intents(intents)

        proof = generate_inclusion_proof(chain, 2)
        proof["root_hash"] = "x" * 64  # Tamper with root

        assert not verify_inclusion_proof(proof)


class TestMerkleChainClass:
    """Tests for MerkleChain high-level interface"""

    def test_create_empty_chain(self):
        """Create empty chain"""
        chain = MerkleChain()
        assert chain.length == 0
        assert chain.root_hash == GENESIS_HASH

    def test_append_to_chain(self):
        """Append intents to chain"""
        chain = MerkleChain()

        intent1 = Intent(intent_name="First", intent_reasoning="First reason")
        chain.append(intent1)

        assert chain.length == 1
        assert chain.root_hash != GENESIS_HASH

    def test_verify_chain(self):
        """Verify chain through interface"""
        chain = MerkleChain()

        for i in range(5):
            chain.append(Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}"))

        result = chain.verify()
        assert result.valid

    def test_chain_serialization(self):
        """Chain can be serialized and deserialized"""
        chain = MerkleChain()
        for i in range(3):
            chain.append(Intent(intent_name=f"Intent {i}", intent_reasoning=f"R{i}"))

        # Serialize
        data = chain.to_list()

        # Deserialize
        restored = MerkleChain.from_list(data)

        assert restored.length == chain.length
        assert restored.root_hash == chain.root_hash


class TestChainedIntentSerialization:
    """Tests for ChainedIntent serialization"""

    def test_to_dict(self):
        """ChainedIntent can be converted to dict"""
        intent = Intent(intent_name="Test", intent_reasoning="Reason")
        chain = chain_intents([intent])

        data = chain[0].to_dict()

        assert "intent_id" in data
        assert "intent_hash" in data
        assert "prev_hash" in data
        assert "chain_hash" in data
        assert "sequence" in data

    def test_from_dict(self):
        """ChainedIntent can be restored from dict"""
        intent = Intent(intent_name="Test", intent_reasoning="Reason")
        chain = chain_intents([intent])

        data = chain[0].to_dict()
        restored = ChainedIntent.from_dict(data)

        assert restored.intent_hash == chain[0].intent_hash
        assert restored.chain_hash == chain[0].chain_hash


# ============================================================================
# Crypto Tests (only run if cryptography is available)
# ============================================================================

@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestKeyPairGeneration:
    """Tests for key pair generation"""

    def test_generate_key_pair(self):
        """Generate Ed25519 key pair"""
        key_pair = generate_key_pair()

        assert key_pair.private_key is not None
        assert key_pair.public_key is not None
        assert len(key_pair.key_id) == 8
        assert key_pair.created_at

    def test_key_pairs_are_unique(self):
        """Each generation produces unique keys"""
        kp1 = generate_key_pair()
        kp2 = generate_key_pair()

        assert kp1.key_id != kp2.key_id


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestKeySerializaton:
    """Tests for key serialization"""

    def test_serialize_private_key(self):
        """Private key can be serialized"""
        key_pair = generate_key_pair()

        pem = serialize_private_key(key_pair.private_key)

        assert b"BEGIN PRIVATE KEY" in pem

    def test_serialize_public_key(self):
        """Public key can be serialized"""
        key_pair = generate_key_pair()

        pem = serialize_public_key(key_pair.public_key)

        assert b"BEGIN PUBLIC KEY" in pem

    def test_load_private_key(self):
        """Private key can be loaded from PEM"""
        key_pair = generate_key_pair()
        pem = serialize_private_key(key_pair.private_key)

        loaded = load_private_key(pem)

        assert loaded is not None

    def test_load_public_key(self):
        """Public key can be loaded from PEM"""
        key_pair = generate_key_pair()
        pem = serialize_public_key(key_pair.public_key)

        loaded = load_public_key(pem)

        assert loaded is not None

    def test_password_protected_key(self):
        """Private key can be password protected"""
        key_pair = generate_key_pair()
        password = "test_password_123"

        pem = serialize_private_key(key_pair.private_key, password)

        # Loading without password should fail
        with pytest.raises(Exception):
            load_private_key(pem)

        # Loading with password should work
        loaded = load_private_key(pem, password)
        assert loaded is not None


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestSigning:
    """Tests for data signing"""

    def test_sign_data(self):
        """Sign data with private key"""
        key_pair = generate_key_pair()
        data = {"message": "Hello, World!", "value": 42}

        signature = sign_data(data, key_pair.private_key, key_pair.key_id)

        assert signature.signature is not None
        assert signature.key_id == key_pair.key_id
        assert signature.algorithm == "Ed25519"

    def test_verify_valid_signature(self):
        """Valid signature passes verification"""
        key_pair = generate_key_pair()
        data = {"message": "Hello, World!"}

        signature = sign_data(data, key_pair.private_key, key_pair.key_id)

        assert verify_signature(data, signature, key_pair.public_key)

    def test_verify_tampered_data(self):
        """Tampered data fails verification"""
        key_pair = generate_key_pair()
        data = {"message": "Hello, World!"}

        signature = sign_data(data, key_pair.private_key, key_pair.key_id)

        # Tamper with data
        data["message"] = "Tampered!"

        with pytest.raises(SignatureError):
            verify_signature(data, signature, key_pair.public_key)

    def test_verify_wrong_key(self):
        """Wrong public key fails verification"""
        kp1 = generate_key_pair()
        kp2 = generate_key_pair()
        data = {"message": "Hello"}

        signature = sign_data(data, kp1.private_key, kp1.key_id)

        with pytest.raises(SignatureError):
            verify_signature(data, signature, kp2.public_key)


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestSignatureSerialization:
    """Tests for signature serialization"""

    def test_signature_to_dict(self):
        """Signature can be converted to dict"""
        key_pair = generate_key_pair()
        data = {"test": "data"}
        signature = sign_data(data, key_pair.private_key, key_pair.key_id)

        sig_dict = signature.to_dict()

        assert "signature" in sig_dict
        assert "key_id" in sig_dict
        assert "algorithm" in sig_dict

    def test_signature_from_dict(self):
        """Signature can be restored from dict"""
        key_pair = generate_key_pair()
        data = {"test": "data"}
        signature = sign_data(data, key_pair.private_key, key_pair.key_id)

        sig_dict = signature.to_dict()
        restored = Signature.from_dict(sig_dict)

        assert restored.signature == signature.signature
        assert restored.key_id == signature.key_id


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestKeyManager:
    """Tests for KeyManager"""

    def test_generate_and_load_key(self):
        """Generate and load key through manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(Path(tmpdir))

            key_pair = km.generate_key(name="test")
            loaded = km.load_key(name="test")

            assert loaded.key_id == key_pair.key_id

    def test_list_keys(self):
        """List available keys"""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(Path(tmpdir))

            km.generate_key(name="key1")
            km.generate_key(name="key2")

            keys = km.list_keys()

            assert "key1" in keys
            assert "key2" in keys

    def test_default_key(self):
        """First key becomes default"""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(Path(tmpdir))

            km.generate_key(name="first")
            km.generate_key(name="second", set_default=False)

            assert km.get_default_key_name() == "first"

    def test_export_public_key(self):
        """Export public key as PEM"""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(Path(tmpdir))
            km.generate_key(name="test")

            pem = km.export_public_key("test")

            assert "BEGIN PUBLIC KEY" in pem

    def test_key_not_found(self):
        """KeyNotFoundError for missing key"""
        with tempfile.TemporaryDirectory() as tmpdir:
            km = KeyManager(Path(tmpdir))

            with pytest.raises(KeyNotFoundError):
                km.load_key("nonexistent")


# ============================================================================
# Storage Integration Tests
# ============================================================================

class TestStorageChainIntegration:
    """Tests for storage chain integration"""

    def test_add_chained_intent(self):
        """Add chained intent through storage"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            chained = storage.add_chained_intent(
                name="Test intent",
                reasoning="This is a test",
            )

            assert chained.sequence == 0
            assert chained.prev_hash == GENESIS_HASH

    def test_load_chained_intents(self):
        """Load chained intents from storage"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            storage.add_chained_intent(name="First", reasoning="R1")
            storage.add_chained_intent(name="Second", reasoning="R2")

            loaded = storage.load_chained_intents()

            assert len(loaded) == 2
            assert loaded[0].sequence == 0
            assert loaded[1].sequence == 1
            assert loaded[1].prev_hash == loaded[0].chain_hash

    def test_verify_chain_through_storage(self):
        """Verify chain through storage"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            storage.add_chained_intent(name="First", reasoning="R1")
            storage.add_chained_intent(name="Second", reasoning="R2")

            result = storage.verify_chain()

            assert result.valid

    def test_get_root_hash(self):
        """Get root hash through storage"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            storage.add_chained_intent(name="Test", reasoning="R1")

            root = storage.get_root_hash()

            assert root != GENESIS_HASH
            assert len(root) == 64

    def test_migrate_to_chain(self):
        """Migrate legacy intents to chain format"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            # Add old-style intents
            storage.add_intent(name="Old1", reasoning="R1")
            storage.add_intent(name="Old2", reasoning="R2")

            # Migrate
            count = storage.migrate_to_chain()

            assert count == 2

            # Verify chain format
            chained = storage.load_chained_intents()
            assert len(chained) == 2
            assert chained[0].sequence == 0


@pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
class TestStorageSigningIntegration:
    """Tests for storage signing integration"""

    def test_signed_commit(self):
        """Create signed commit through storage"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            # Generate key
            km = storage.get_key_manager()
            km.generate_key()

            # Create signed intent
            chained = storage.add_chained_intent(
                name="Signed intent",
                reasoning="This is signed",
                sign=True,
            )

            assert chained.signature is not None
            assert "key_id" in chained.signature

    def test_verify_signed_intent(self):
        """Verify signed intent"""
        from intentlog.storage import IntentLogStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = IntentLogStorage(Path(tmpdir))
            storage.init_project("test")

            # Generate key
            km = storage.get_key_manager()
            km.generate_key()

            # Create signed intent
            chained = storage.add_chained_intent(
                name="Signed intent",
                reasoning="This is signed",
                sign=True,
            )

            # Verify
            assert storage.verify_intent_signature(chained)
