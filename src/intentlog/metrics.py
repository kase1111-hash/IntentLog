"""
Metrics Module for IntentLog

Implements metrics from Doctrine-of-intent.md:
- Intent Density (Di) scoring
- Information Density metrics
- Auditability and fraud resistance ratings

These metrics help assess the quality and verifiability of intent logs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import statistics

from .core import Intent


class DensityLevel(Enum):
    """Intent density levels"""
    SPARSE = "sparse"           # < 0.2: Minimal logging
    LOW = "low"                 # 0.2-0.4: Infrequent logging
    MODERATE = "moderate"       # 0.4-0.6: Regular logging
    HIGH = "high"               # 0.6-0.8: Frequent logging
    CONTINUOUS = "continuous"   # > 0.8: Near-continuous logging


@dataclass
class IntentDensity:
    """
    Intent Density (Di) Score

    Measures the resolution and continuity of intent records.
    Score ranges from 0.0 (no records) to 1.0 (continuous recording).

    Components:
    - Resolution: Average decisions logged per time unit
    - Continuity: Inverse of gap frequency
    - Coverage: Proportion of active time with records
    """
    score: float = 0.0                    # Overall density score (0.0-1.0)
    level: DensityLevel = DensityLevel.SPARSE
    resolution_score: float = 0.0         # Decisions per time unit
    continuity_score: float = 0.0         # Gap penalty inverse
    coverage_score: float = 0.0           # Time coverage
    intents_per_hour: float = 0.0         # Raw measure
    avg_gap_minutes: float = 0.0          # Average time between intents
    max_gap_minutes: float = 0.0          # Longest gap
    total_hours: float = 0.0              # Total span of records
    active_hours: float = 0.0             # Hours with activity
    sample_size: int = 0                  # Number of intents analyzed

    # Property aliases for convenient access
    @property
    def Di(self) -> float:
        """Alias for score"""
        return self.score

    @property
    def resolution(self) -> float:
        """Alias for resolution_score"""
        return self.resolution_score

    @property
    def continuity(self) -> float:
        """Alias for continuity_score"""
        return self.continuity_score

    @property
    def coverage(self) -> float:
        """Alias for coverage_score"""
        return self.coverage_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level.value,
            "components": {
                "resolution": round(self.resolution_score, 4),
                "continuity": round(self.continuity_score, 4),
                "coverage": round(self.coverage_score, 4),
            },
            "metrics": {
                "intents_per_hour": round(self.intents_per_hour, 2),
                "avg_gap_minutes": round(self.avg_gap_minutes, 2),
                "max_gap_minutes": round(self.max_gap_minutes, 2),
                "total_hours": round(self.total_hours, 2),
                "active_hours": round(self.active_hours, 2),
            },
        }


@dataclass
class InformationDensity:
    """
    Information Density Metrics

    Measures the information richness of intent records.

    Components:
    - Content depth: Average reasoning length and complexity
    - Metadata richness: Presence of contextual data
    - Linkage: Parent-child relationships
    """
    content_depth_score: float = 0.0      # 0.0-1.0
    metadata_richness_score: float = 0.0  # 0.0-1.0
    linkage_score: float = 0.0            # 0.0-1.0
    overall_score: float = 0.0            # Weighted average

    avg_reasoning_words: float = 0.0
    avg_reasoning_chars: float = 0.0
    avg_metadata_keys: float = 0.0
    linked_ratio: float = 0.0             # Ratio with parent_id
    unique_terms: int = 0
    total_terms: int = 0
    entropy: float = 0.0
    compression_ratio: float = 0.0

    # Property aliases for convenient access
    @property
    def avg_words(self) -> float:
        """Alias for avg_reasoning_words"""
        return self.avg_reasoning_words

    @property
    def avg_chars(self) -> float:
        """Alias for avg_reasoning_chars"""
        return self.avg_reasoning_chars

    @property
    def unique_terms_ratio(self) -> float:
        """Ratio of unique terms to total terms"""
        if self.total_terms == 0:
            return 0.0
        return self.unique_terms / self.total_terms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 4),
            "components": {
                "content_depth": round(self.content_depth_score, 4),
                "metadata_richness": round(self.metadata_richness_score, 4),
                "linkage": round(self.linkage_score, 4),
            },
            "metrics": {
                "avg_reasoning_words": round(self.avg_reasoning_words, 2),
                "avg_metadata_keys": round(self.avg_metadata_keys, 2),
                "linked_ratio": round(self.linked_ratio, 4),
            },
        }


@dataclass
class AuditabilityScore:
    """
    Auditability Score

    Measures how well the intent log supports external audit.

    Factors:
    - Completeness: All required fields present
    - Traceability: Can follow decision chains
    - Verifiability: Has hashes, signatures, timestamps
    """
    score: float = 0.0                    # 0.0-1.0
    completeness: float = 0.0
    traceability: float = 0.0
    verifiability: float = 0.0
    issues: List[str] = field(default_factory=list)

    @property
    def rating(self) -> str:
        """Get rating based on score"""
        if self.score >= 0.8:
            return "excellent"
        elif self.score >= 0.6:
            return "good"
        elif self.score >= 0.4:
            return "fair"
        else:
            return "poor"

    @property
    def components(self) -> Dict[str, float]:
        """Return components as dict"""
        return {
            "completeness": self.completeness,
            "traceability": self.traceability,
            "verifiability": self.verifiability,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "components": {
                "completeness": round(self.completeness, 4),
                "traceability": round(self.traceability, 4),
                "verifiability": round(self.verifiability, 4),
            },
            "issues": self.issues,
        }


@dataclass
class FraudResistance:
    """
    Fraud Resistance Rating

    Measures resistance to tampering and fabrication.

    Factors:
    - Temporal authenticity: Timestamps appear genuine
    - Content authenticity: Reasoning appears non-synthetic
    - Structural integrity: Hash chains intact
    """
    score: float = 0.0                    # 0.0-1.0
    temporal_score: float = 0.0
    content_score: float = 0.0
    integrity_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)

    @property
    def rating(self) -> str:
        """Get rating based on score"""
        if self.score >= 0.8:
            return "excellent"
        elif self.score >= 0.6:
            return "good"
        elif self.score >= 0.4:
            return "fair"
        else:
            return "poor"

    @property
    def factors(self) -> Dict[str, float]:
        """Return factors as dict"""
        return {
            "temporal_authenticity": self.temporal_score,
            "content_authenticity": self.content_score,
            "structural_integrity": self.integrity_score,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "components": {
                "temporal_authenticity": round(self.temporal_score, 4),
                "content_authenticity": round(self.content_score, 4),
                "structural_integrity": round(self.integrity_score, 4),
            },
            "risk_factors": self.risk_factors,
        }


class IntentMetrics:
    """
    Calculate metrics for intent logs.

    Based on Doctrine-of-intent.md framework for
    provenance verification and value attribution.
    """

    # Thresholds for density scoring
    RESOLUTION_OPTIMAL_PER_HOUR = 4.0     # Target intents per hour
    CONTINUITY_GAP_THRESHOLD = 30.0       # Minutes before gap penalty
    COVERAGE_ACTIVE_RATIO = 0.5           # Target active time ratio

    # Thresholds for content scoring
    MIN_REASONING_WORDS = 10              # Minimum expected words
    OPTIMAL_REASONING_WORDS = 50          # Optimal reasoning length
    OPTIMAL_METADATA_KEYS = 3             # Optimal metadata fields

    def __init__(self, intents: List[Intent]):
        """Initialize with list of intents"""
        self.intents = sorted(intents, key=lambda i: i.timestamp)

    def compute_intent_density(self) -> IntentDensity:
        """
        Compute Intent Density (Di) score.

        Di = (Resolution + Continuity + Coverage) / 3

        Where:
        - Resolution: min(1.0, intents_per_hour / OPTIMAL)
        - Continuity: 1.0 - (gap_penalty)
        - Coverage: active_hours / total_hours
        """
        if len(self.intents) < 2:
            return IntentDensity(
                score=0.0 if not self.intents else 0.2,
                level=DensityLevel.SPARSE,
            )

        # Calculate time span
        first_time = self.intents[0].timestamp
        last_time = self.intents[-1].timestamp
        total_seconds = (last_time - first_time).total_seconds()
        total_hours = total_seconds / 3600 if total_seconds > 0 else 0.001

        # Calculate gaps
        gaps = []
        for i in range(1, len(self.intents)):
            gap = (self.intents[i].timestamp - self.intents[i-1].timestamp).total_seconds() / 60
            gaps.append(gap)

        avg_gap = statistics.mean(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0

        # Resolution score
        intents_per_hour = len(self.intents) / total_hours
        resolution = min(1.0, intents_per_hour / self.RESOLUTION_OPTIMAL_PER_HOUR)

        # Continuity score (penalize large gaps)
        large_gaps = sum(1 for g in gaps if g > self.CONTINUITY_GAP_THRESHOLD)
        gap_penalty = large_gaps / len(gaps) if gaps else 0
        continuity = max(0.0, 1.0 - gap_penalty)

        # Coverage score (approximate active hours)
        active_chunks = 0
        current_chunk_start = self.intents[0].timestamp

        for i in range(1, len(self.intents)):
            gap = (self.intents[i].timestamp - self.intents[i-1].timestamp).total_seconds() / 60
            if gap > self.CONTINUITY_GAP_THRESHOLD:
                # End of active chunk
                chunk_hours = (self.intents[i-1].timestamp - current_chunk_start).total_seconds() / 3600
                active_chunks += max(0.1, chunk_hours)  # Minimum chunk size
                current_chunk_start = self.intents[i].timestamp

        # Add final chunk
        chunk_hours = (last_time - current_chunk_start).total_seconds() / 3600
        active_chunks += max(0.1, chunk_hours)

        active_hours = min(active_chunks, total_hours)
        coverage = active_hours / total_hours if total_hours > 0 else 0

        # Overall score
        score = (resolution + continuity + coverage) / 3

        # Determine level
        if score >= 0.8:
            level = DensityLevel.CONTINUOUS
        elif score >= 0.6:
            level = DensityLevel.HIGH
        elif score >= 0.4:
            level = DensityLevel.MODERATE
        elif score >= 0.2:
            level = DensityLevel.LOW
        else:
            level = DensityLevel.SPARSE

        return IntentDensity(
            score=score,
            level=level,
            resolution_score=resolution,
            continuity_score=continuity,
            coverage_score=coverage,
            intents_per_hour=intents_per_hour,
            avg_gap_minutes=avg_gap,
            max_gap_minutes=max_gap,
            total_hours=total_hours,
            active_hours=active_hours,
            sample_size=len(self.intents),
        )

    def compute_information_density(self) -> InformationDensity:
        """
        Compute Information Density metrics.

        Measures the richness of information in intent records.
        """
        if not self.intents:
            return InformationDensity()

        # Content depth: based on reasoning length
        word_counts = [len(i.intent_reasoning.split()) for i in self.intents]
        avg_words = statistics.mean(word_counts)

        if avg_words >= self.OPTIMAL_REASONING_WORDS:
            content_depth = 1.0
        elif avg_words >= self.MIN_REASONING_WORDS:
            content_depth = (avg_words - self.MIN_REASONING_WORDS) / (
                self.OPTIMAL_REASONING_WORDS - self.MIN_REASONING_WORDS
            )
        else:
            content_depth = avg_words / self.MIN_REASONING_WORDS * 0.5

        # Metadata richness
        metadata_counts = [len(i.metadata) for i in self.intents]
        avg_metadata = statistics.mean(metadata_counts)
        metadata_richness = min(1.0, avg_metadata / self.OPTIMAL_METADATA_KEYS)

        # Linkage: ratio of intents with parent_intent_id
        linked_count = sum(1 for i in self.intents if i.parent_intent_id)
        linked_ratio = linked_count / len(self.intents)
        linkage_score = linked_ratio  # Simple ratio

        # Calculate additional metrics
        char_counts = [len(i.intent_reasoning) for i in self.intents]
        avg_chars = statistics.mean(char_counts) if char_counts else 0

        # Unique terms analysis
        all_words = []
        for intent in self.intents:
            all_words.extend(intent.intent_reasoning.lower().split())
        total_terms = len(all_words)
        unique_terms = len(set(all_words))

        # Simple entropy approximation
        if total_terms > 0:
            entropy = unique_terms / total_terms * 3.5  # Approximate bits
        else:
            entropy = 0.0

        # Compression ratio estimate (unique/total)
        compression_ratio = unique_terms / total_terms if total_terms > 0 else 0.0

        # Overall (weighted average)
        overall = (
            content_depth * 0.4 +
            metadata_richness * 0.3 +
            linkage_score * 0.3
        )

        return InformationDensity(
            content_depth_score=content_depth,
            metadata_richness_score=metadata_richness,
            linkage_score=linkage_score,
            overall_score=overall,
            avg_reasoning_words=avg_words,
            avg_reasoning_chars=avg_chars,
            avg_metadata_keys=avg_metadata,
            linked_ratio=linked_ratio,
            unique_terms=unique_terms,
            total_terms=total_terms,
            entropy=entropy,
            compression_ratio=compression_ratio,
        )

    def compute_auditability(self) -> AuditabilityScore:
        """
        Compute Auditability Score.

        Measures how well the log supports external audit.
        """
        if not self.intents:
            return AuditabilityScore()

        issues = []

        # Completeness: required fields present
        complete_count = 0
        for intent in self.intents:
            is_complete = (
                intent.intent_id and
                intent.intent_name and
                intent.intent_reasoning and
                intent.timestamp
            )
            if is_complete:
                complete_count += 1
            else:
                if not issues or "incomplete" not in issues[-1]:
                    issues.append("Some intents have incomplete required fields")

        completeness = complete_count / len(self.intents)

        # Traceability: can follow decision chains
        traceable_count = 0
        orphan_children = 0
        parent_ids = {i.intent_id for i in self.intents}

        for intent in self.intents:
            if intent.parent_intent_id:
                if intent.parent_intent_id in parent_ids:
                    traceable_count += 1
                else:
                    orphan_children += 1

        if orphan_children > 0:
            issues.append(f"{orphan_children} intents reference non-existent parents")

        # Traceability score
        linked_intents = sum(1 for i in self.intents if i.parent_intent_id)
        if linked_intents > 0:
            traceability = traceable_count / linked_intents
        else:
            traceability = 0.5  # Neutral if no chains

        # Verifiability: has hashes, timestamps
        verifiable_count = 0
        for intent in self.intents:
            has_timestamp = intent.timestamp is not None
            has_hash = intent.metadata.get("hash") is not None
            # Also check for attached files with hashes
            has_file_hashes = bool(intent.metadata.get("attached_files"))

            if has_timestamp and (has_hash or has_file_hashes):
                verifiable_count += 1
            elif has_timestamp:
                verifiable_count += 0.5  # Partial credit

        verifiability = verifiable_count / len(self.intents)

        if verifiability < 0.5:
            issues.append("Low verifiability: missing hashes on most intents")

        # Overall score
        score = (completeness * 0.4 + traceability * 0.3 + verifiability * 0.3)

        return AuditabilityScore(
            score=score,
            completeness=completeness,
            traceability=traceability,
            verifiability=verifiability,
            issues=issues,
        )

    def compute_fraud_resistance(self) -> FraudResistance:
        """
        Compute Fraud Resistance Rating.

        Assesses resistance to tampering and fabrication.
        """
        if not self.intents:
            return FraudResistance()

        risk_factors = []

        # Temporal authenticity
        # Check for suspicious patterns:
        # - All same timestamp
        # - Perfect intervals (synthetic)
        # - Out of order timestamps

        timestamps = [i.timestamp for i in self.intents]
        unique_timestamps = len(set(timestamps))

        if unique_timestamps == 1 and len(self.intents) > 1:
            risk_factors.append("All intents have identical timestamps")
            temporal_score = 0.1
        elif unique_timestamps < len(self.intents) * 0.5:
            risk_factors.append("Many intents share timestamps")
            temporal_score = 0.4
        else:
            # Check for perfectly regular intervals (suspicious)
            if len(self.intents) > 2:
                gaps = []
                for i in range(1, len(self.intents)):
                    gap = (self.intents[i].timestamp - self.intents[i-1].timestamp).total_seconds()
                    gaps.append(gap)

                if gaps and len(set(gaps)) == 1:
                    risk_factors.append("Perfectly regular intervals (possible synthetic data)")
                    temporal_score = 0.5
                else:
                    temporal_score = 0.9
            else:
                temporal_score = 0.8

        # Content authenticity
        # Check for:
        # - Very short/empty reasoning
        # - Duplicate content
        # - Unusual patterns

        reasoning_texts = [i.intent_reasoning for i in self.intents]
        unique_reasoning = len(set(reasoning_texts))

        if unique_reasoning < len(self.intents) * 0.7:
            risk_factors.append("High content duplication")
            content_score = 0.4
        else:
            # Check for very short reasoning
            short_count = sum(1 for r in reasoning_texts if len(r.split()) < 5)
            if short_count > len(self.intents) * 0.5:
                risk_factors.append("Many intents have very short reasoning")
                content_score = 0.5
            else:
                content_score = 0.9

        # Structural integrity
        # Check for:
        # - Hash presence
        # - Chain integrity
        # - Metadata consistency

        hash_present = sum(1 for i in self.intents if i.metadata.get("hash"))
        if hash_present < len(self.intents) * 0.5:
            risk_factors.append("Many intents lack cryptographic hashes")
            integrity_score = 0.5
        else:
            integrity_score = 0.9

        # Overall score
        score = (temporal_score * 0.35 + content_score * 0.35 + integrity_score * 0.3)

        return FraudResistance(
            score=score,
            temporal_score=temporal_score,
            content_score=content_score,
            integrity_score=integrity_score,
            risk_factors=risk_factors,
        )

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a single report"""
        return {
            "intent_density": self.compute_intent_density().to_dict(),
            "information_density": self.compute_information_density().to_dict(),
            "auditability": self.compute_auditability().to_dict(),
            "fraud_resistance": self.compute_fraud_resistance().to_dict(),
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all metrics"""
        density = self.compute_intent_density()
        info = self.compute_information_density()
        audit = self.compute_auditability()
        fraud = self.compute_fraud_resistance()

        # Overall health score
        health = (
            density.score * 0.3 +
            info.overall_score * 0.2 +
            audit.score * 0.25 +
            fraud.score * 0.25
        )

        if health >= 0.8:
            status = "excellent"
        elif health >= 0.6:
            status = "good"
        elif health >= 0.4:
            status = "fair"
        else:
            status = "needs_improvement"

        return {
            "health_score": round(health, 4),
            "status": status,
            "intent_count": len(self.intents),
            "density_level": density.level.value,
        }
