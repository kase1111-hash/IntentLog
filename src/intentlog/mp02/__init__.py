"""
MP-02 Protocol Components

Implements the Proof-of-Effort Receipt Protocol as defined in mp-02-spec.md.

Components:
- Observer: Captures raw signals with timestamps
- Segmentation: Groups signals into bounded segments
- Validator: LLM-assisted coherence and progression analysis
- Receipt: Cryptographic records attesting effort occurred
- Ledger: Append-only log for receipt anchoring
"""

from .signal import Signal, SignalType, SignalSource
from .observer import Observer, TextObserver, CommandObserver
from .segmentation import SegmentationEngine, EffortSegment, SegmentationRule
from .validator import Validator, ValidationResult, ValidationMetadata
from .receipt import Receipt, ReceiptBuilder, ReceiptError
from .ledger import Ledger, LedgerEntry, LedgerError, InclusionProof

__all__ = [
    # Signal types
    "Signal",
    "SignalType",
    "SignalSource",
    # Observer
    "Observer",
    "TextObserver",
    "CommandObserver",
    # Segmentation
    "SegmentationEngine",
    "EffortSegment",
    "SegmentationRule",
    # Validator
    "Validator",
    "ValidationResult",
    "ValidationMetadata",
    # Receipt
    "Receipt",
    "ReceiptBuilder",
    "ReceiptError",
    # Ledger
    "Ledger",
    "LedgerEntry",
    "LedgerError",
    "InclusionProof",
]
