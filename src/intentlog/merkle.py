"""
Merkle Tree Module for IntentLog

Provides hash-chain linking and verification for tamper-evident
intent history per MP-02 specification.

Features:
- Chain linking with prev_hash
- Root hash computation
- Chain integrity verification
- Branch divergence handling
"""

import json
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .core import Intent


# Genesis hash for first intent in chain
GENESIS_HASH = "0" * 64


@dataclass
class ChainedIntent:
    """Intent with chain linking metadata"""
    intent: Intent
    intent_hash: str       # Hash of this intent's content
    prev_hash: str         # Hash of previous intent (or GENESIS_HASH)
    chain_hash: str        # Combined hash: SHA256(intent_hash + prev_hash)
    sequence: int          # Position in chain (0-indexed)
    signature: Optional[Dict[str, Any]] = None  # Optional Ed25519 signature

    def to_dict(self) -> Dict[str, Any]:
        """Export chained intent as dictionary"""
        data = {
            "intent_id": self.intent.intent_id,
            "intent_name": self.intent.intent_name,
            "intent_reasoning": self.intent.intent_reasoning,
            "timestamp": (
                self.intent.timestamp.isoformat()
                if isinstance(self.intent.timestamp, datetime)
                else self.intent.timestamp
            ),
            "parent_intent_id": self.intent.parent_intent_id,
            "metadata": self.intent.metadata,
            "intent_hash": self.intent_hash,
            "prev_hash": self.prev_hash,
            "chain_hash": self.chain_hash,
            "sequence": self.sequence,
        }
        if self.signature:
            data["signature"] = self.signature
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChainedIntent":
        """Create ChainedIntent from dictionary"""
        intent = Intent(
            intent_id=data["intent_id"],
            intent_name=data["intent_name"],
            intent_reasoning=data["intent_reasoning"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            parent_intent_id=data.get("parent_intent_id"),
        )
        return cls(
            intent=intent,
            intent_hash=data["intent_hash"],
            prev_hash=data["prev_hash"],
            chain_hash=data["chain_hash"],
            sequence=data["sequence"],
            signature=data.get("signature"),
        )


@dataclass
class ChainVerificationResult:
    """Result of chain verification"""
    valid: bool
    entries_checked: int
    root_hash: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    broken_at: Optional[int] = None  # Sequence number where chain breaks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "entries_checked": self.entries_checked,
            "root_hash": self.root_hash,
            "errors": self.errors,
            "warnings": self.warnings,
            "broken_at": self.broken_at,
        }


def compute_intent_content_hash(intent: Intent) -> str:
    """
    Compute SHA-256 hash of an intent's content.

    Uses canonical JSON (sorted keys, compact) for deterministic hashing.
    This hash covers the intent's semantic content but NOT the chain links.

    Args:
        intent: Intent to hash

    Returns:
        Full 64-character SHA-256 hex string
    """
    canonical = {
        "intent_id": intent.intent_id,
        "intent_name": intent.intent_name,
        "intent_reasoning": intent.intent_reasoning,
        "timestamp": (
            intent.timestamp.isoformat()
            if isinstance(intent.timestamp, datetime)
            else intent.timestamp
        ),
        "parent_intent_id": intent.parent_intent_id,
        "metadata": intent.metadata,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def compute_chain_hash(intent_hash: str, prev_hash: str) -> str:
    """
    Compute chain hash by combining intent hash with previous hash.

    This creates the tamper-evident link between intents.

    Args:
        intent_hash: Hash of current intent's content
        prev_hash: Hash of previous intent in chain

    Returns:
        Combined chain hash
    """
    combined = f"{intent_hash}{prev_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()


def chain_intents(
    intents: List[Intent],
    starting_hash: str = GENESIS_HASH,
    starting_sequence: int = 0,
) -> List[ChainedIntent]:
    """
    Create a chain of linked intents with prev_hash.

    Args:
        intents: List of intents to chain (in order)
        starting_hash: Previous hash to link from (for appending)
        starting_sequence: Starting sequence number

    Returns:
        List of ChainedIntent with chain linking
    """
    chained = []
    prev_hash = starting_hash

    for i, intent in enumerate(intents):
        intent_hash = compute_intent_content_hash(intent)
        chain_hash = compute_chain_hash(intent_hash, prev_hash)

        chained_intent = ChainedIntent(
            intent=intent,
            intent_hash=intent_hash,
            prev_hash=prev_hash,
            chain_hash=chain_hash,
            sequence=starting_sequence + i,
        )
        chained.append(chained_intent)

        # Next intent links to this one
        prev_hash = chain_hash

    return chained


def append_to_chain(
    existing_chain: List[ChainedIntent],
    new_intent: Intent,
) -> ChainedIntent:
    """
    Append a single intent to an existing chain.

    Args:
        existing_chain: Existing chain of intents
        new_intent: New intent to append

    Returns:
        New ChainedIntent linked to the chain
    """
    if existing_chain:
        prev_hash = existing_chain[-1].chain_hash
        sequence = existing_chain[-1].sequence + 1
    else:
        prev_hash = GENESIS_HASH
        sequence = 0

    intent_hash = compute_intent_content_hash(new_intent)
    chain_hash = compute_chain_hash(intent_hash, prev_hash)

    return ChainedIntent(
        intent=new_intent,
        intent_hash=intent_hash,
        prev_hash=prev_hash,
        chain_hash=chain_hash,
        sequence=sequence,
    )


def verify_chain(chained_intents: List[ChainedIntent]) -> ChainVerificationResult:
    """
    Verify the integrity of an intent chain.

    Checks:
    1. Each intent's content hash matches stored hash
    2. Chain hashes are correctly computed
    3. prev_hash links are valid
    4. Sequence numbers are continuous

    Args:
        chained_intents: List of ChainedIntent to verify

    Returns:
        ChainVerificationResult with detailed status
    """
    if not chained_intents:
        return ChainVerificationResult(
            valid=True,
            entries_checked=0,
            root_hash=GENESIS_HASH,
        )

    errors = []
    warnings = []
    broken_at = None
    expected_prev_hash = GENESIS_HASH

    for i, chained in enumerate(chained_intents):
        # Check sequence
        if chained.sequence != i:
            warnings.append(
                f"Sequence gap at position {i}: expected {i}, got {chained.sequence}"
            )

        # Recompute content hash
        computed_hash = compute_intent_content_hash(chained.intent)
        if computed_hash != chained.intent_hash:
            errors.append(
                f"Intent hash mismatch at sequence {chained.sequence}: "
                f"content may have been modified"
            )
            if broken_at is None:
                broken_at = chained.sequence

        # Verify prev_hash link
        if chained.prev_hash != expected_prev_hash:
            errors.append(
                f"Chain break at sequence {chained.sequence}: "
                f"prev_hash doesn't match previous chain_hash"
            )
            if broken_at is None:
                broken_at = chained.sequence

        # Verify chain hash
        computed_chain_hash = compute_chain_hash(chained.intent_hash, chained.prev_hash)
        if computed_chain_hash != chained.chain_hash:
            errors.append(
                f"Chain hash mismatch at sequence {chained.sequence}: "
                f"chain may have been tampered"
            )
            if broken_at is None:
                broken_at = chained.sequence

        # Update expected for next iteration
        expected_prev_hash = chained.chain_hash

    # Root hash is the last chain hash
    root_hash = chained_intents[-1].chain_hash if chained_intents else GENESIS_HASH

    return ChainVerificationResult(
        valid=len(errors) == 0,
        entries_checked=len(chained_intents),
        root_hash=root_hash,
        errors=errors,
        warnings=warnings,
        broken_at=broken_at,
    )


def compute_root_hash(chained_intents: List[ChainedIntent]) -> str:
    """
    Compute the root hash of a chain.

    The root hash is the chain_hash of the last intent,
    which depends on all previous intents.

    Args:
        chained_intents: List of ChainedIntent

    Returns:
        Root hash string
    """
    if not chained_intents:
        return GENESIS_HASH
    return chained_intents[-1].chain_hash


def find_divergence_point(
    chain_a: List[ChainedIntent],
    chain_b: List[ChainedIntent],
) -> Optional[int]:
    """
    Find where two chains diverge.

    Used for branch reconciliation.

    Args:
        chain_a: First chain
        chain_b: Second chain

    Returns:
        Sequence number where chains diverge, or None if identical
    """
    min_len = min(len(chain_a), len(chain_b))

    for i in range(min_len):
        if chain_a[i].chain_hash != chain_b[i].chain_hash:
            return i

    # Chains match up to min_len
    if len(chain_a) != len(chain_b):
        return min_len  # Divergence at the length difference

    return None  # Chains are identical


def generate_inclusion_proof(
    chained_intents: List[ChainedIntent],
    target_sequence: int,
) -> Dict[str, Any]:
    """
    Generate a proof that an intent is included in the chain.

    The proof contains the chain of hashes needed to verify
    inclusion without the full chain.

    Args:
        chained_intents: Full chain of intents
        target_sequence: Sequence number of intent to prove

    Returns:
        Proof dictionary with hash path
    """
    if target_sequence >= len(chained_intents):
        raise ValueError(f"Sequence {target_sequence} not in chain")

    target = chained_intents[target_sequence]

    # Collect hash path from target to root
    hash_path = []
    for i in range(target_sequence, len(chained_intents)):
        chained = chained_intents[i]
        hash_path.append({
            "sequence": chained.sequence,
            "intent_hash": chained.intent_hash,
            "prev_hash": chained.prev_hash,
            "chain_hash": chained.chain_hash,
        })

    return {
        "target_sequence": target_sequence,
        "target_intent_id": target.intent.intent_id,
        "target_intent_hash": target.intent_hash,
        "root_hash": compute_root_hash(chained_intents),
        "proof_path": hash_path,
        "chain_length": len(chained_intents),
    }


def verify_inclusion_proof(proof: Dict[str, Any]) -> bool:
    """
    Verify an inclusion proof.

    Args:
        proof: Proof dictionary from generate_inclusion_proof

    Returns:
        True if proof is valid
    """
    path = proof["proof_path"]
    if not path:
        return False

    # Verify each step in the path
    for i, step in enumerate(path):
        computed = compute_chain_hash(step["intent_hash"], step["prev_hash"])
        if computed != step["chain_hash"]:
            return False

        # Verify chain continuity
        if i < len(path) - 1:
            next_step = path[i + 1]
            if step["chain_hash"] != next_step["prev_hash"]:
                return False

    # Verify root matches
    return path[-1]["chain_hash"] == proof["root_hash"]


class MerkleChain:
    """
    High-level interface for managing intent chains.

    Wraps chain operations with convenient methods for
    adding, verifying, and exporting chains.
    """

    def __init__(self, chained_intents: Optional[List[ChainedIntent]] = None):
        """
        Initialize chain.

        Args:
            chained_intents: Existing chain to load
        """
        self._chain: List[ChainedIntent] = chained_intents or []

    @property
    def chain(self) -> List[ChainedIntent]:
        """Get the current chain"""
        return self._chain

    @property
    def root_hash(self) -> str:
        """Get the current root hash"""
        return compute_root_hash(self._chain)

    @property
    def length(self) -> int:
        """Get chain length"""
        return len(self._chain)

    def append(self, intent: Intent, signature: Optional[Dict[str, Any]] = None) -> ChainedIntent:
        """
        Append a new intent to the chain.

        Args:
            intent: Intent to append
            signature: Optional signature to attach

        Returns:
            The new ChainedIntent
        """
        chained = append_to_chain(self._chain, intent)
        if signature:
            chained.signature = signature
        self._chain.append(chained)
        return chained

    def verify(self) -> ChainVerificationResult:
        """
        Verify the chain integrity.

        Returns:
            ChainVerificationResult
        """
        return verify_chain(self._chain)

    def get_proof(self, sequence: int) -> Dict[str, Any]:
        """
        Generate inclusion proof for an intent.

        Args:
            sequence: Sequence number

        Returns:
            Proof dictionary
        """
        return generate_inclusion_proof(self._chain, sequence)

    def to_list(self) -> List[Dict[str, Any]]:
        """Export chain as list of dictionaries"""
        return [c.to_dict() for c in self._chain]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "MerkleChain":
        """Create chain from list of dictionaries"""
        chained = [ChainedIntent.from_dict(d) for d in data]
        return cls(chained)

    @classmethod
    def from_intents(cls, intents: List[Intent]) -> "MerkleChain":
        """Create new chain from plain intents"""
        chained = chain_intents(intents)
        return cls(chained)
