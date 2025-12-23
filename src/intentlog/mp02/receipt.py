"""
Receipt Construction for MP-02 Protocol

Per MP-02 Section 8, each Effort Receipt MUST include:
- Receipt ID
- Time bounds
- Hashes of referenced signals
- Deterministic effort summary
- Validation metadata
- Observer and Validator identifiers

Receipts MAY reference:
- Prior receipts
- External artifacts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import hashlib
import json
import uuid

from .signal import Signal
from .segmentation import EffortSegment
from .validator import ValidationResult, ValidationMetadata


class ReceiptError(Exception):
    """Error during receipt construction or validation"""
    pass


@dataclass
class ExternalArtifact:
    """Reference to an external artifact (file, commit, etc.)"""
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    artifact_type: str = ""           # "file", "git_commit", "url", etc.
    reference: str = ""               # Path, hash, URL, etc.
    content_hash: str = ""            # Hash of content for integrity
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "reference": self.reference,
            "content_hash": self.content_hash,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExternalArtifact":
        return cls(
            artifact_id=data.get("artifact_id", str(uuid.uuid4())[:8]),
            artifact_type=data.get("artifact_type", ""),
            reference=data.get("reference", ""),
            content_hash=data.get("content_hash", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Receipt:
    """
    Cryptographic record attesting that a specific effort segment occurred.

    A Receipt is the fundamental unit of effort verification in MP-02.
    It records evidence, not conclusions about value (Section 2).
    """
    # Required fields (Section 8)
    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    signal_hashes: List[str] = field(default_factory=list)
    summary: str = ""                 # Deterministic effort summary
    validation_metadata: Optional[ValidationMetadata] = None
    observer_id: str = ""
    validator_id: str = ""

    # Optional references (Section 8)
    prior_receipt_ids: List[str] = field(default_factory=list)
    artifacts: List[ExternalArtifact] = field(default_factory=list)

    # Additional metadata
    segment_id: str = ""              # Reference to source segment
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"              # Receipt format version
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validation scores (preserved from validation)
    coherence_score: Optional[float] = None
    progression_score: Optional[float] = None
    consistency_score: Optional[float] = None
    confidence: float = 0.5

    # Computed hash
    _receipt_hash: str = field(default="", repr=False)

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of receipt contents.

        Per Section 9, receipts are anchored by hashing contents.
        """
        data = {
            "receipt_id": self.receipt_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "signal_hashes": sorted(self.signal_hashes),
            "summary": self.summary,
            "observer_id": self.observer_id,
            "validator_id": self.validator_id,
            "prior_receipt_ids": sorted(self.prior_receipt_ids),
            "segment_id": self.segment_id,
            "version": self.version,
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def receipt_hash(self) -> str:
        """Get receipt hash, computing if needed"""
        if not self._receipt_hash:
            self._receipt_hash = self.compute_hash()
        return self._receipt_hash

    def verify_hash(self) -> bool:
        """Verify that stored hash matches computed hash"""
        if not self._receipt_hash:
            return True
        return self.compute_hash() == self._receipt_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary"""
        return {
            "receipt_id": self.receipt_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "signal_hashes": self.signal_hashes,
            "summary": self.summary,
            "validation_metadata": self.validation_metadata.to_dict() if self.validation_metadata else None,
            "observer_id": self.observer_id,
            "validator_id": self.validator_id,
            "prior_receipt_ids": self.prior_receipt_ids,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "segment_id": self.segment_id,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
            "coherence_score": self.coherence_score,
            "progression_score": self.progression_score,
            "consistency_score": self.consistency_score,
            "confidence": self.confidence,
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Receipt":
        """Deserialize receipt from dictionary"""
        receipt = cls(
            receipt_id=data.get("receipt_id", str(uuid.uuid4())),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            signal_hashes=data.get("signal_hashes", []),
            summary=data.get("summary", ""),
            validation_metadata=ValidationMetadata.from_dict(data["validation_metadata"]) if data.get("validation_metadata") else None,
            observer_id=data.get("observer_id", ""),
            validator_id=data.get("validator_id", ""),
            prior_receipt_ids=data.get("prior_receipt_ids", []),
            artifacts=[ExternalArtifact.from_dict(a) for a in data.get("artifacts", [])],
            segment_id=data.get("segment_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
            coherence_score=data.get("coherence_score"),
            progression_score=data.get("progression_score"),
            consistency_score=data.get("consistency_score"),
            confidence=data.get("confidence", 0.5),
        )
        receipt._receipt_hash = data.get("receipt_hash", "")
        return receipt

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Receipt":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


class ReceiptBuilder:
    """
    Builder for constructing effort receipts.

    Ensures all required fields are present and properly formatted
    per Section 8 requirements.
    """

    def __init__(self):
        self._segment: Optional[EffortSegment] = None
        self._validation: Optional[ValidationResult] = None
        self._prior_receipts: List[str] = []
        self._artifacts: List[ExternalArtifact] = []
        self._metadata: Dict[str, Any] = {}

    def from_segment(self, segment: EffortSegment) -> "ReceiptBuilder":
        """Set the source segment for the receipt"""
        self._segment = segment
        return self

    def with_validation(self, validation: ValidationResult) -> "ReceiptBuilder":
        """Add validation result to receipt"""
        self._validation = validation
        return self

    def reference_prior(self, receipt_id: str) -> "ReceiptBuilder":
        """Reference a prior receipt (for chaining)"""
        if receipt_id not in self._prior_receipts:
            self._prior_receipts.append(receipt_id)
        return self

    def attach_artifact(
        self,
        artifact_type: str,
        reference: str,
        content_hash: str = "",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReceiptBuilder":
        """Attach an external artifact"""
        artifact = ExternalArtifact(
            artifact_type=artifact_type,
            reference=reference,
            content_hash=content_hash,
            description=description,
            metadata=metadata or {},
        )
        self._artifacts.append(artifact)
        return self

    def with_metadata(self, key: str, value: Any) -> "ReceiptBuilder":
        """Add metadata to receipt"""
        self._metadata[key] = value
        return self

    def build(self) -> Receipt:
        """
        Build the receipt.

        Raises ReceiptError if required fields are missing.
        """
        if not self._segment:
            raise ReceiptError("Segment is required to build receipt")

        if not self._segment.signals:
            raise ReceiptError("Segment must contain at least one signal")

        # Get signal hashes
        signal_hashes = [s.content_hash for s in self._segment.signals]

        # Get time bounds from segment
        start_time = self._segment.start_time
        end_time = self._segment.end_time

        if not start_time:
            start_time = min(s.timestamp for s in self._segment.signals)
        if not end_time:
            end_time = max(s.timestamp for s in self._segment.signals)

        # Generate summary
        if self._validation and self._validation.summary:
            summary = self._validation.summary
        else:
            summary = self._segment.get_summary()

        # Get observer ID from signals
        observer_id = ""
        for signal in self._segment.signals:
            if signal.source and signal.source.observer_id:
                observer_id = signal.source.observer_id
                break

        # Get validator ID and metadata
        validator_id = ""
        validation_metadata = None
        if self._validation and self._validation.metadata:
            validator_id = self._validation.metadata.validator_id
            validation_metadata = self._validation.metadata

        # Build receipt
        receipt = Receipt(
            start_time=start_time,
            end_time=end_time,
            signal_hashes=signal_hashes,
            summary=summary,
            validation_metadata=validation_metadata,
            observer_id=observer_id,
            validator_id=validator_id,
            prior_receipt_ids=self._prior_receipts.copy(),
            artifacts=self._artifacts.copy(),
            segment_id=self._segment.segment_id,
            metadata=self._metadata.copy(),
        )

        # Add validation scores if available
        if self._validation:
            receipt.coherence_score = self._validation.coherence_score
            receipt.progression_score = self._validation.progression_score
            receipt.consistency_score = self._validation.consistency_score
            receipt.confidence = self._validation.confidence

        return receipt

    def reset(self) -> "ReceiptBuilder":
        """Reset builder state for reuse"""
        self._segment = None
        self._validation = None
        self._prior_receipts = []
        self._artifacts = []
        self._metadata = {}
        return self


def verify_receipt(receipt: Receipt, signals: Optional[List[Signal]] = None) -> Dict[str, Any]:
    """
    Verify a receipt's integrity.

    Per Section 10, third parties MUST be able to:
    - Recompute receipt hashes
    - Inspect validation metadata
    - Confirm ledger inclusion (done separately)

    Returns a verification report.
    """
    report = {
        "receipt_id": receipt.receipt_id,
        "verified": True,
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    # Check 1: Verify receipt hash
    computed_hash = receipt.compute_hash()
    if receipt._receipt_hash and receipt._receipt_hash != computed_hash:
        report["verified"] = False
        report["errors"].append("Receipt hash mismatch - content may have been altered")
    else:
        report["checks"].append("Receipt hash verified")

    # Check 2: Verify time bounds
    if receipt.start_time and receipt.end_time:
        if receipt.start_time > receipt.end_time:
            report["verified"] = False
            report["errors"].append("Invalid time bounds: start_time > end_time")
        else:
            report["checks"].append("Time bounds valid")
    else:
        report["warnings"].append("Time bounds not fully specified")

    # Check 3: Verify signal hashes if signals provided
    if signals:
        expected_hashes = set(receipt.signal_hashes)
        actual_hashes = set(s.content_hash for s in signals)

        if expected_hashes != actual_hashes:
            missing = expected_hashes - actual_hashes
            extra = actual_hashes - expected_hashes
            if missing:
                report["errors"].append(f"Missing signals: {len(missing)} hashes not found")
            if extra:
                report["warnings"].append(f"Extra signals: {len(extra)} hashes not in receipt")
            report["verified"] = False
        else:
            report["checks"].append(f"All {len(signals)} signal hashes verified")

    # Check 4: Verify validation metadata present
    if receipt.validation_metadata:
        if receipt.validation_metadata.model_name:
            report["checks"].append(f"Validator disclosed: {receipt.validation_metadata.model_name}")
        else:
            report["warnings"].append("Validator model not disclosed")
    else:
        report["warnings"].append("No validation metadata present")

    # Check 5: Verify observer ID present
    if receipt.observer_id:
        report["checks"].append(f"Observer ID: {receipt.observer_id}")
    else:
        report["warnings"].append("No observer ID specified")

    return report
