"""
Tests for MP-02 Protocol Components
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from intentlog.mp02.signal import Signal, SignalType, SignalSource
from intentlog.mp02.observer import Observer, TextObserver, CommandObserver, AnnotationObserver, ObserverConfig
from intentlog.mp02.segmentation import SegmentationEngine, SegmentationRule, SegmentationMethod, EffortSegment
from intentlog.mp02.validator import Validator, ValidationResult, ValidationMetadata
from intentlog.mp02.receipt import Receipt, ReceiptBuilder, ReceiptError, ExternalArtifact, verify_receipt
from intentlog.mp02.ledger import Ledger, LedgerEntry, LedgerError, InclusionProof, AnchoringService


class TestSignal:
    """Tests for Signal class"""

    def test_signal_creation(self):
        """Test creating a signal"""
        signal = Signal(
            signal_type=SignalType.TEXT_EDIT,
            content="Hello world",
        )
        assert signal.signal_type == SignalType.TEXT_EDIT
        assert signal.content == "Hello world"
        assert signal.signal_id is not None

    def test_signal_hash(self):
        """Test signal content hashing"""
        signal = Signal(content="Test content")
        assert signal.content_hash is not None
        assert len(signal.content_hash) == 16

    def test_signal_deterministic_hash(self):
        """Test that same content produces same hash"""
        signal1 = Signal(content="Same content")
        signal2 = Signal(content="Same content")
        assert signal1.content_hash == signal2.content_hash

    def test_signal_different_hash(self):
        """Test that different content produces different hash"""
        signal1 = Signal(content="Content 1")
        signal2 = Signal(content="Content 2")
        assert signal1.content_hash != signal2.content_hash

    def test_signal_integrity_verification(self):
        """Test signal integrity verification"""
        signal = Signal(content="Test content")
        assert signal.verify_integrity()

    def test_signal_serialization(self):
        """Test signal to_dict and from_dict"""
        signal = Signal(
            signal_type=SignalType.COMMAND,
            content="git commit",
            metadata={"exit_code": 0},
        )
        data = signal.to_dict()
        restored = Signal.from_dict(data)

        assert restored.signal_type == signal.signal_type
        assert restored.content == signal.content
        assert restored.metadata == signal.metadata

    def test_signal_json(self):
        """Test signal JSON serialization"""
        signal = Signal(content="Test")
        json_str = signal.to_json()
        restored = Signal.from_json(json_str)
        assert restored.content == signal.content


class TestSignalSource:
    """Tests for SignalSource class"""

    def test_source_creation(self):
        """Test creating a signal source"""
        source = SignalSource(
            observer_id="test123",
            observer_type="text",
            capture_modality="file_watcher",
        )
        assert source.observer_id == "test123"
        assert source.observer_type == "text"

    def test_source_serialization(self):
        """Test source serialization"""
        source = SignalSource(
            observer_id="obs1",
            observer_type="command",
            capture_modality="history",
            location="/home/user/.bash_history",
        )
        data = source.to_dict()
        restored = SignalSource.from_dict(data)

        assert restored.observer_id == source.observer_id
        assert restored.location == source.location


class TestObserver:
    """Tests for Observer classes"""

    def test_annotation_observer(self):
        """Test annotation observer"""
        observer = AnnotationObserver()
        observer.start()

        signal = observer.annotate("Test note", category="note")

        assert signal.signal_type == SignalType.ANNOTATION
        assert signal.content == "Test note"
        assert signal.metadata["category"] == "note"

        signals = observer.get_signals()
        assert len(signals) == 1

        observer.stop()

    def test_observer_config(self):
        """Test observer configuration"""
        config = ObserverConfig(
            buffer_size=500,
            auto_flush_interval=60.0,
            capture_modality="keyboard",
        )
        assert config.buffer_size == 500
        assert config.capture_modality == "keyboard"

    def test_observer_stats(self):
        """Test observer statistics"""
        observer = AnnotationObserver()
        observer.start()

        observer.annotate("Note 1")
        observer.annotate("Note 2")

        stats = observer.get_stats()
        assert stats["signals_buffered"] == 2
        assert stats["signals_captured"] == 2
        assert stats["is_running"] is True

        observer.stop()

    def test_command_observer_log(self):
        """Test explicit command logging"""
        observer = CommandObserver()
        observer.start()

        signal = observer.log_command("git status", working_dir="/home/test", exit_code=0)

        assert signal.signal_type == SignalType.COMMAND
        assert signal.content == "git status"
        assert signal.metadata["exit_code"] == 0

        observer.stop()


class TestSegmentation:
    """Tests for SegmentationEngine"""

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for testing"""
        base_time = datetime.now()
        return [
            Signal(
                signal_type=SignalType.TEXT_EDIT,
                timestamp=base_time + timedelta(minutes=i),
                content=f"Edit {i}",
            )
            for i in range(10)
        ]

    def test_segmentation_rule(self):
        """Test segmentation rule creation"""
        rule = SegmentationRule(
            method=SegmentationMethod.TIME_WINDOW,
            time_window_minutes=15,
        )
        assert rule.method == SegmentationMethod.TIME_WINDOW
        assert rule.time_window_minutes == 15

    def test_rule_disclosure(self):
        """Test rule disclosure generation"""
        rule = SegmentationRule(
            method=SegmentationMethod.TIME_WINDOW,
            time_window_minutes=30,
        )
        disclosure = rule.get_disclosure()
        assert "30-minute" in disclosure

    def test_time_window_segmentation(self, sample_signals):
        """Test time window segmentation"""
        rule = SegmentationRule(
            method=SegmentationMethod.TIME_WINDOW,
            time_window_minutes=5,
        )
        engine = SegmentationEngine(rule)
        segments = engine.segment(sample_signals)

        assert len(segments) >= 1
        for segment in segments:
            assert segment.signal_count >= 1

    def test_activity_boundary_segmentation(self):
        """Test activity boundary segmentation"""
        base_time = datetime.now()
        signals = [
            Signal(timestamp=base_time, content="1"),
            Signal(timestamp=base_time + timedelta(minutes=1), content="2"),
            Signal(timestamp=base_time + timedelta(minutes=10), content="3"),  # Gap
            Signal(timestamp=base_time + timedelta(minutes=11), content="4"),
        ]

        rule = SegmentationRule(
            method=SegmentationMethod.ACTIVITY_BOUNDARY,
            gap_threshold_minutes=5,
        )
        engine = SegmentationEngine(rule)
        segments = engine.segment(signals)

        assert len(segments) == 2

    def test_marker_segmentation(self):
        """Test explicit marker segmentation"""
        signals = [
            Signal(signal_type=SignalType.TEXT_EDIT, content="Edit 1"),
            Signal(signal_type=SignalType.TEXT_EDIT, content="Edit 2"),
            Signal(
                signal_type=SignalType.ANNOTATION,
                content="Milestone",
                metadata={"category": "milestone"},
            ),
            Signal(signal_type=SignalType.TEXT_EDIT, content="Edit 3"),
        ]

        rule = SegmentationRule(
            method=SegmentationMethod.EXPLICIT_MARKER,
            marker_categories=["milestone"],
        )
        engine = SegmentationEngine(rule)
        segments = engine.segment(signals)

        assert len(segments) == 2

    def test_segment_hash(self, sample_signals):
        """Test segment hash computation"""
        segment = EffortSegment(signals=sample_signals[:5])
        hash1 = segment.compute_hash()

        assert len(hash1) == 64  # SHA-256 hex
        assert hash1 == segment.compute_hash()  # Deterministic

    def test_segment_summary(self, sample_signals):
        """Test segment summary generation"""
        segment = EffortSegment(signals=sample_signals[:5])
        summary = segment.get_summary()

        assert "5 signals" in summary


class TestValidator:
    """Tests for Validator class"""

    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment for testing"""
        signals = [
            Signal(
                signal_type=SignalType.TEXT_EDIT,
                timestamp=datetime.now() + timedelta(minutes=i),
                content=f"Writing code for feature {i}",
            )
            for i in range(5)
        ]
        return EffortSegment(signals=signals)

    def test_validation_metadata(self):
        """Test validation metadata creation"""
        metadata = ValidationMetadata(
            model_name="gpt-4",
            model_version="0613",
            provider="openai",
        )
        assert metadata.model_name == "gpt-4"

        data = metadata.to_dict()
        restored = ValidationMetadata.from_dict(data)
        assert restored.model_name == metadata.model_name

    def test_rule_based_validation(self, sample_segment):
        """Test rule-based validation (no LLM)"""
        validator = Validator()
        result = validator.validate(sample_segment)

        assert isinstance(result, ValidationResult)
        assert result.segment_id == sample_segment.segment_id
        assert result.summary is not None
        assert 0 <= result.confidence <= 1

    def test_validation_detects_gaps(self):
        """Test that validation detects time gaps"""
        base_time = datetime.now()
        signals = [
            Signal(timestamp=base_time, content="1"),
            Signal(timestamp=base_time + timedelta(minutes=10), content="2"),  # Gap
        ]
        segment = EffortSegment(signals=signals)

        validator = Validator()
        result = validator.validate(segment)

        assert result.has_gaps is True

    def test_validation_detects_duplication(self):
        """Test that validation detects content duplication"""
        signals = [
            Signal(content="Same content"),
            Signal(content="Same content"),
            Signal(content="Same content"),
        ]
        segment = EffortSegment(signals=signals)

        validator = Validator()
        result = validator.validate(segment)

        assert result.possible_duplication is True

    def test_validation_result_hash(self, sample_segment):
        """Test validation result hash"""
        validator = Validator()
        result = validator.validate(sample_segment)

        assert result.result_hash is not None
        assert len(result.result_hash) == 64


class TestReceipt:
    """Tests for Receipt class"""

    @pytest.fixture
    def sample_receipt(self):
        """Create a sample receipt"""
        return Receipt(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            signal_hashes=["abc123", "def456"],
            summary="Test effort segment",
            observer_id="obs1",
            validator_id="val1",
        )

    def test_receipt_creation(self, sample_receipt):
        """Test receipt creation"""
        assert sample_receipt.receipt_id is not None
        assert len(sample_receipt.signal_hashes) == 2

    def test_receipt_hash(self, sample_receipt):
        """Test receipt hash computation"""
        hash1 = sample_receipt.receipt_hash
        assert len(hash1) == 64
        assert hash1 == sample_receipt.compute_hash()

    def test_receipt_verification(self, sample_receipt):
        """Test receipt hash verification"""
        assert sample_receipt.verify_hash()

    def test_receipt_serialization(self, sample_receipt):
        """Test receipt serialization"""
        data = sample_receipt.to_dict()
        restored = Receipt.from_dict(data)

        assert restored.receipt_id == sample_receipt.receipt_id
        assert restored.signal_hashes == sample_receipt.signal_hashes
        assert restored.summary == sample_receipt.summary

    def test_receipt_json(self, sample_receipt):
        """Test receipt JSON serialization"""
        json_str = sample_receipt.to_json()
        restored = Receipt.from_json(json_str)
        assert restored.receipt_id == sample_receipt.receipt_id

    def test_verify_receipt(self, sample_receipt):
        """Test receipt verification function"""
        report = verify_receipt(sample_receipt)

        assert "verified" in report
        assert "checks" in report
        assert report["receipt_id"] == sample_receipt.receipt_id


class TestReceiptBuilder:
    """Tests for ReceiptBuilder"""

    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment"""
        signals = [
            Signal(content=f"Signal {i}", source=SignalSource(
                observer_id="obs1",
                observer_type="text",
                capture_modality="test",
            ))
            for i in range(3)
        ]
        return EffortSegment(signals=signals)

    def test_builder_basic(self, sample_segment):
        """Test basic receipt building"""
        builder = ReceiptBuilder()
        receipt = builder.from_segment(sample_segment).build()

        assert receipt.segment_id == sample_segment.segment_id
        assert len(receipt.signal_hashes) == 3

    def test_builder_with_validation(self, sample_segment):
        """Test receipt building with validation"""
        validator = Validator()
        validation = validator.validate(sample_segment)

        builder = ReceiptBuilder()
        receipt = (builder
            .from_segment(sample_segment)
            .with_validation(validation)
            .build())

        assert receipt.confidence > 0
        assert receipt.summary == validation.summary

    def test_builder_with_artifacts(self, sample_segment):
        """Test receipt building with artifacts"""
        builder = ReceiptBuilder()
        receipt = (builder
            .from_segment(sample_segment)
            .attach_artifact("file", "/path/to/file", content_hash="abc123")
            .build())

        assert len(receipt.artifacts) == 1
        assert receipt.artifacts[0].artifact_type == "file"

    def test_builder_with_prior_receipt(self, sample_segment):
        """Test receipt chaining"""
        builder = ReceiptBuilder()
        receipt = (builder
            .from_segment(sample_segment)
            .reference_prior("prev-receipt-id")
            .build())

        assert "prev-receipt-id" in receipt.prior_receipt_ids

    def test_builder_requires_segment(self):
        """Test that builder requires segment"""
        builder = ReceiptBuilder()
        with pytest.raises(ReceiptError):
            builder.build()


class TestLedger:
    """Tests for Ledger class"""

    @pytest.fixture
    def temp_ledger(self, tmp_path):
        """Create a temporary ledger"""
        return Ledger(tmp_path / "ledger")

    @pytest.fixture
    def sample_receipt(self):
        """Create a sample receipt"""
        return Receipt(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            signal_hashes=["abc", "def"],
            summary="Test",
        )

    def test_ledger_creation(self, temp_ledger):
        """Test ledger creation"""
        stats = temp_ledger.get_stats()
        assert stats["entry_count"] == 0

    def test_ledger_append(self, temp_ledger, sample_receipt):
        """Test appending to ledger"""
        entry = temp_ledger.append(sample_receipt)

        assert entry.receipt_hash == sample_receipt.receipt_hash
        assert entry.sequence == 1

    def test_ledger_chain(self, temp_ledger, sample_receipt):
        """Test ledger chain integrity"""
        entry1 = temp_ledger.append(sample_receipt)
        entry2 = temp_ledger.append(sample_receipt)

        assert entry2.prev_hash == entry1.compute_hash()
        assert entry2.sequence == 2

    def test_ledger_read_entries(self, temp_ledger, sample_receipt):
        """Test reading ledger entries"""
        temp_ledger.append(sample_receipt)
        temp_ledger.append(sample_receipt)

        entries = temp_ledger.read_entries()
        assert len(entries) == 2

    def test_ledger_find_entry(self, temp_ledger, sample_receipt):
        """Test finding entry by receipt ID"""
        temp_ledger.append(sample_receipt)

        entry = temp_ledger.find_entry(sample_receipt.receipt_id)
        assert entry is not None
        assert entry.receipt_id == sample_receipt.receipt_id

    def test_ledger_verify_chain(self, temp_ledger, sample_receipt):
        """Test chain verification"""
        temp_ledger.append(sample_receipt)
        temp_ledger.append(sample_receipt)

        report = temp_ledger.verify_chain()
        assert report["verified"] is True
        assert report["entries_checked"] == 2

    def test_ledger_inclusion_proof(self, temp_ledger, sample_receipt):
        """Test inclusion proof generation"""
        temp_ledger.append(sample_receipt)

        proof = temp_ledger.generate_inclusion_proof(sample_receipt.receipt_id)
        assert proof is not None
        assert proof.receipt_id == sample_receipt.receipt_id
        assert proof.verified is True

    def test_ledger_record_failure(self, temp_ledger):
        """Test recording failures"""
        entry = temp_ledger.record_failure(
            failure_type="observation_gap",
            description="5 minute gap in observation",
        )

        assert entry.metadata["type"] == "failure"
        assert entry.metadata["failure_type"] == "observation_gap"

    def test_ledger_export(self, temp_ledger, sample_receipt, tmp_path):
        """Test ledger export"""
        temp_ledger.append(sample_receipt)
        temp_ledger.append(sample_receipt)

        output_path = tmp_path / "export.log"
        count = temp_ledger.export(output_path)

        assert count == 2
        assert output_path.exists()


class TestAnchoringService:
    """Tests for AnchoringService"""

    @pytest.fixture
    def temp_service(self, tmp_path):
        """Create temporary anchoring service"""
        ledger = Ledger(tmp_path / "ledger")
        return AnchoringService(ledger)

    @pytest.fixture
    def sample_receipt(self):
        """Create sample receipt"""
        return Receipt(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            signal_hashes=["abc"],
            summary="Test",
        )

    def test_anchor_receipt(self, temp_service, sample_receipt):
        """Test anchoring a receipt"""
        entry = temp_service.anchor_receipt(sample_receipt)
        assert entry.receipt_hash == sample_receipt.receipt_hash

    def test_create_checkpoint(self, temp_service, sample_receipt):
        """Test checkpoint creation"""
        temp_service.anchor_receipt(sample_receipt)
        checkpoint = temp_service.create_checkpoint()

        assert "checkpoint_id" in checkpoint
        assert "checkpoint_hash" in checkpoint
        assert checkpoint["entry_count"] == 1
        assert checkpoint["chain_verified"] is True

    def test_verify_checkpoint(self, temp_service, sample_receipt):
        """Test checkpoint verification"""
        temp_service.anchor_receipt(sample_receipt)
        checkpoint = temp_service.create_checkpoint()

        assert temp_service.verify_checkpoint(checkpoint) is True


class TestLedgerEntry:
    """Tests for LedgerEntry"""

    def test_entry_creation(self):
        """Test entry creation"""
        entry = LedgerEntry(
            receipt_hash="abc123",
            receipt_id="receipt1",
            sequence=1,
        )
        assert entry.receipt_hash == "abc123"
        assert entry.sequence == 1

    def test_entry_hash(self):
        """Test entry hash computation"""
        entry = LedgerEntry(
            receipt_hash="abc123",
            receipt_id="receipt1",
            sequence=1,
        )
        hash1 = entry.compute_hash()
        assert len(hash1) == 64

    def test_entry_serialization(self):
        """Test entry line serialization"""
        entry = LedgerEntry(
            receipt_hash="abc123",
            receipt_id="receipt1",
            sequence=1,
        )
        line = entry.to_line()
        restored = LedgerEntry.from_line(line)

        assert restored.receipt_hash == entry.receipt_hash
        assert restored.sequence == entry.sequence


class TestExternalArtifact:
    """Tests for ExternalArtifact"""

    def test_artifact_creation(self):
        """Test artifact creation"""
        artifact = ExternalArtifact(
            artifact_type="git_commit",
            reference="abc123def",
            description="Initial commit",
        )
        assert artifact.artifact_type == "git_commit"
        assert artifact.reference == "abc123def"

    def test_artifact_serialization(self):
        """Test artifact serialization"""
        artifact = ExternalArtifact(
            artifact_type="file",
            reference="/path/to/file",
            content_hash="hash123",
        )
        data = artifact.to_dict()
        restored = ExternalArtifact.from_dict(data)

        assert restored.artifact_type == artifact.artifact_type
        assert restored.content_hash == artifact.content_hash
