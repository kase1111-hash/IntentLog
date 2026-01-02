"""
Ledger & Anchoring System for MP-02 Protocol

Per MP-02 Section 9, receipts are anchored by:
1. Hashing receipt contents
2. Appending hashes to a ledger

The ledger MUST be:
- Append-only
- Time-ordered
- Publicly verifiable

Per Section 10, third parties MUST be able to:
- Recompute receipt hashes
- Inspect validation metadata
- Confirm ledger inclusion
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
import hashlib
import json
import os
import uuid

from .receipt import Receipt
from ..filelock import file_lock


class LedgerError(Exception):
    """Error during ledger operations"""
    pass


@dataclass
class LedgerEntry:
    """
    A single entry in the ledger.

    Format: timestamp|receipt_hash|prev_hash|sequence
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    receipt_hash: str = ""
    receipt_id: str = ""
    prev_hash: str = ""               # Hash of previous entry (chain)
    sequence: int = 0                 # Sequence number in ledger
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute hash of this entry for chaining"""
        data = f"{self.timestamp.isoformat()}|{self.receipt_hash}|{self.prev_hash}|{self.sequence}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_line(self) -> str:
        """Serialize to ledger line format"""
        return json.dumps({
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "receipt_hash": self.receipt_hash,
            "receipt_id": self.receipt_id,
            "prev_hash": self.prev_hash,
            "sequence": self.sequence,
            "entry_hash": self.compute_hash(),
            "metadata": self.metadata,
        })

    @classmethod
    def from_line(cls, line: str) -> "LedgerEntry":
        """Deserialize from ledger line"""
        data = json.loads(line.strip())
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())[:8]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            receipt_hash=data.get("receipt_hash", ""),
            receipt_id=data.get("receipt_id", ""),
            prev_hash=data.get("prev_hash", ""),
            sequence=data.get("sequence", 0),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "receipt_hash": self.receipt_hash,
            "receipt_id": self.receipt_id,
            "prev_hash": self.prev_hash,
            "sequence": self.sequence,
            "entry_hash": self.compute_hash(),
            "metadata": self.metadata,
        }


@dataclass
class InclusionProof:
    """
    Proof that a receipt is included in the ledger.

    Allows third-party verification without full ledger.
    """
    receipt_id: str
    receipt_hash: str
    ledger_entry: LedgerEntry
    chain_verification: List[str]     # Hashes forming the chain
    verified: bool = False
    verification_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
            "ledger_entry": self.ledger_entry.to_dict(),
            "chain_verification": self.chain_verification,
            "verified": self.verified,
            "verification_time": self.verification_time.isoformat(),
        }


class Ledger:
    """
    Append-only ledger for anchoring receipts.

    Per Section 9:
    - Append-only: entries can only be added, never modified
    - Time-ordered: entries are stored in chronological order
    - Publicly verifiable: anyone can verify the chain

    Per Section 11 (Failure Modes), the ledger records:
    - Gaps in observation
    - Conflicting validations
    - Suspected manipulation
    - Incomplete segments
    """

    DEFAULT_FILENAME = "ledger.log"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB before rotation

    def __init__(self, ledger_dir: Optional[Path] = None):
        """
        Initialize ledger.

        Args:
            ledger_dir: Directory to store ledger files.
                       Defaults to .intentlog/ledger/
        """
        self.ledger_dir = ledger_dir or Path(".intentlog/ledger")
        self._ensure_dir()
        self._sequence = 0
        self._last_hash = ""
        self._initialize()

    def _ensure_dir(self) -> None:
        """Ensure ledger directory exists"""
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def _get_ledger_path(self) -> Path:
        """Get path to current ledger file"""
        return self.ledger_dir / self.DEFAULT_FILENAME

    def _get_rotated_path(self, index: int) -> Path:
        """Get path to rotated ledger file"""
        return self.ledger_dir / f"ledger.{index}.log"

    def _initialize(self) -> None:
        """Initialize ledger state from existing entries"""
        ledger_path = self._get_ledger_path()
        if not ledger_path.exists():
            return

        # Read last entry to get sequence and hash
        last_entry = None
        with open(ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        last_entry = LedgerEntry.from_line(line)
                    except Exception:
                        continue

        if last_entry:
            self._sequence = last_entry.sequence
            self._last_hash = last_entry.compute_hash()

    def append(self, receipt: Receipt) -> LedgerEntry:
        """
        Append a receipt to the ledger.

        This is the core operation for anchoring receipts per Section 9.
        """
        self._sequence += 1

        entry = LedgerEntry(
            timestamp=datetime.now(),
            receipt_hash=receipt.receipt_hash,
            receipt_id=receipt.receipt_id,
            prev_hash=self._last_hash,
            sequence=self._sequence,
        )

        # Write to ledger with file locking
        ledger_path = self._get_ledger_path()
        self._check_rotation()

        with file_lock(ledger_path, exclusive=True):
            with open(ledger_path, "a") as f:
                f.write(entry.to_line() + "\n")

        self._last_hash = entry.compute_hash()
        return entry

    def _check_rotation(self) -> None:
        """Check if ledger file needs rotation"""
        ledger_path = self._get_ledger_path()
        if not ledger_path.exists():
            return

        if ledger_path.stat().st_size >= self.MAX_FILE_SIZE:
            self._rotate()

    def _rotate(self) -> None:
        """Rotate ledger file"""
        ledger_path = self._get_ledger_path()

        # Find next rotation index
        index = 1
        while self._get_rotated_path(index).exists():
            index += 1

        # Move current ledger to rotated file
        ledger_path.rename(self._get_rotated_path(index))

    def read_entries(self, limit: int = 0, offset: int = 0) -> List[LedgerEntry]:
        """
        Read entries from the ledger.

        Args:
            limit: Maximum entries to return (0 = all)
            offset: Number of entries to skip
        """
        entries = []
        ledger_path = self._get_ledger_path()

        if not ledger_path.exists():
            return entries

        with open(ledger_path, "r") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if limit and len(entries) >= limit:
                    break
                if line.strip():
                    try:
                        entries.append(LedgerEntry.from_line(line))
                    except Exception:
                        continue

        return entries

    def iterate_entries(self) -> Iterator[LedgerEntry]:
        """Iterate over all entries in the ledger"""
        ledger_path = self._get_ledger_path()

        if not ledger_path.exists():
            return

        with open(ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        yield LedgerEntry.from_line(line)
                    except Exception:
                        continue

    def find_entry(self, receipt_id: str) -> Optional[LedgerEntry]:
        """Find entry by receipt ID"""
        for entry in self.iterate_entries():
            if entry.receipt_id == receipt_id:
                return entry
        return None

    def verify_chain(self) -> Dict[str, Any]:
        """
        Verify integrity of the entire ledger chain.

        Per Section 10, third parties must be able to verify the chain.
        """
        report = {
            "verified": True,
            "entries_checked": 0,
            "errors": [],
            "warnings": [],
            "first_entry": None,
            "last_entry": None,
        }

        prev_hash = ""
        prev_sequence = 0

        for entry in self.iterate_entries():
            report["entries_checked"] += 1

            if report["first_entry"] is None:
                report["first_entry"] = entry.to_dict()

            # Check chain linkage
            if prev_hash and entry.prev_hash != prev_hash:
                report["verified"] = False
                report["errors"].append(
                    f"Chain break at sequence {entry.sequence}: "
                    f"expected prev_hash {prev_hash[:8]}..., got {entry.prev_hash[:8]}..."
                )

            # Check sequence ordering
            if prev_sequence and entry.sequence != prev_sequence + 1:
                report["warnings"].append(
                    f"Sequence gap: {prev_sequence} -> {entry.sequence}"
                )

            prev_hash = entry.compute_hash()
            prev_sequence = entry.sequence
            report["last_entry"] = entry.to_dict()

        return report

    def generate_inclusion_proof(self, receipt_id: str) -> Optional[InclusionProof]:
        """
        Generate proof that a receipt is included in the ledger.

        This allows third-party verification without the full ledger.
        """
        entry = self.find_entry(receipt_id)
        if not entry:
            return None

        # Build chain verification (hashes of entries before and after)
        chain = []
        found = False

        for e in self.iterate_entries():
            chain.append(e.compute_hash())
            if e.entry_id == entry.entry_id:
                found = True
            if found and len(chain) > entry.sequence + 5:
                break  # Include a few entries after for context

        # Start from a few entries before
        start = max(0, entry.sequence - 5)
        chain = chain[start:]

        proof = InclusionProof(
            receipt_id=receipt_id,
            receipt_hash=entry.receipt_hash,
            ledger_entry=entry,
            chain_verification=chain,
            verified=True,
        )

        return proof

    def verify_inclusion(self, proof: InclusionProof) -> bool:
        """Verify an inclusion proof"""
        # Find the entry in current ledger
        entry = self.find_entry(proof.receipt_id)
        if not entry:
            return False

        # Verify the entry hash matches
        if entry.compute_hash() != proof.ledger_entry.compute_hash():
            return False

        # Verify receipt hash matches
        if entry.receipt_hash != proof.receipt_hash:
            return False

        return True

    def record_failure(
        self,
        failure_type: str,
        description: str,
        related_receipt_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LedgerEntry:
        """
        Record a failure mode in the ledger.

        Per Section 11, failures reduce confidence but don't invalidate receipts.

        Failure types:
        - "observation_gap": Gap in observation
        - "validation_conflict": Conflicting validations
        - "suspected_manipulation": Suspected tampering
        - "incomplete_segment": Incomplete segment
        """
        self._sequence += 1

        entry = LedgerEntry(
            timestamp=datetime.now(),
            receipt_hash="",  # No receipt for failures
            receipt_id=related_receipt_id,
            prev_hash=self._last_hash,
            sequence=self._sequence,
            metadata={
                "type": "failure",
                "failure_type": failure_type,
                "description": description,
                **(metadata or {}),
            },
        )

        # Write to ledger
        ledger_path = self._get_ledger_path()
        with file_lock(ledger_path, exclusive=True):
            with open(ledger_path, "a") as f:
                f.write(entry.to_line() + "\n")

        self._last_hash = entry.compute_hash()
        return entry

    def export(self, output_path: Path) -> int:
        """
        Export ledger for external verification.

        Returns number of entries exported.
        """
        count = 0
        with open(output_path, "w") as f:
            for entry in self.iterate_entries():
                f.write(entry.to_line() + "\n")
                count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics"""
        ledger_path = self._get_ledger_path()

        stats = {
            "ledger_path": str(ledger_path),
            "exists": ledger_path.exists(),
            "entry_count": 0,
            "file_size": 0,
            "first_entry_time": None,
            "last_entry_time": None,
            "failure_count": 0,
        }

        if not ledger_path.exists():
            return stats

        stats["file_size"] = ledger_path.stat().st_size

        for entry in self.iterate_entries():
            stats["entry_count"] += 1
            if stats["first_entry_time"] is None:
                stats["first_entry_time"] = entry.timestamp.isoformat()
            stats["last_entry_time"] = entry.timestamp.isoformat()
            if entry.metadata.get("type") == "failure":
                stats["failure_count"] += 1

        return stats


class AnchoringService:
    """
    Service for anchoring receipts to external systems.

    Supports multiple anchoring backends for additional trust.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self._checkpoint_interval = 100  # Entries between checkpoints

    def anchor_receipt(self, receipt: Receipt) -> LedgerEntry:
        """Anchor a receipt to the ledger"""
        return self.ledger.append(receipt)

    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current ledger state.

        Checkpoints can be anchored to external systems for
        additional verification.
        """
        stats = self.ledger.get_stats()
        verification = self.ledger.verify_chain()

        checkpoint = {
            "checkpoint_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "entry_count": stats["entry_count"],
            "last_entry_hash": self.ledger._last_hash,
            "chain_verified": verification["verified"],
            "ledger_stats": stats,
        }

        # Compute checkpoint hash
        canonical = json.dumps(checkpoint, sort_keys=True)
        checkpoint["checkpoint_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

        # Save checkpoint
        checkpoint_path = self.ledger.ledger_dir / f"checkpoint_{checkpoint['checkpoint_id'][:8]}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        return checkpoint

    def verify_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Verify a checkpoint against current ledger"""
        # Recompute checkpoint hash
        verify_data = {k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}
        canonical = json.dumps(verify_data, sort_keys=True)
        computed_hash = hashlib.sha256(canonical.encode()).hexdigest()

        if computed_hash != checkpoint.get("checkpoint_hash"):
            return False

        # Verify chain is still valid
        verification = self.ledger.verify_chain()
        return verification["verified"]
