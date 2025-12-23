"""
Validator Framework for MP-02 Protocol

Per MP-02 Section 7, Validators:
- MAY assess linguistic coherence, conceptual progression, internal consistency
- MAY detect indicators of synthesis vs duplication
- MUST produce deterministic summaries
- MUST disclose model identity and version
- MUST preserve dissent and uncertainty
- MUST NOT declare effort as valuable
- MUST NOT assert originality or ownership
- MUST NOT collapse ambiguous signals into certainty
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import hashlib
import json
import uuid

from .signal import Signal, SignalType
from .segmentation import EffortSegment


@dataclass
class ValidationMetadata:
    """
    Metadata about the validation process.

    Required per Section 7: model identity and version must be disclosed.
    """
    validator_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""              # e.g., "gpt-4", "claude-3"
    model_version: str = ""           # Specific version string
    provider: str = ""                # e.g., "openai", "anthropic"
    timestamp: datetime = field(default_factory=datetime.now)
    prompt_hash: str = ""             # Hash of prompt used
    parameters: Dict[str, Any] = field(default_factory=dict)  # Model parameters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "prompt_hash": self.prompt_hash,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationMetadata":
        return cls(
            validator_id=data.get("validator_id", str(uuid.uuid4())[:8]),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            provider=data.get("provider", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            prompt_hash=data.get("prompt_hash", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class ValidationResult:
    """
    Result of validating an effort segment.

    Per Section 7:
    - Summaries must be deterministic
    - Uncertainty must be preserved, not collapsed
    - No value judgments (just evidence of effort)
    """
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str = ""
    metadata: Optional[ValidationMetadata] = None

    # Assessment scores (0.0 to 1.0, with None meaning not assessed)
    coherence_score: Optional[float] = None    # Linguistic coherence
    progression_score: Optional[float] = None  # Conceptual progression
    consistency_score: Optional[float] = None  # Internal consistency

    # Deterministic summary (required)
    summary: str = ""

    # Uncertainty indicators
    confidence: float = 0.5           # Overall confidence (0.0 to 1.0)
    uncertainty_notes: List[str] = field(default_factory=list)
    dissenting_observations: List[str] = field(default_factory=list)

    # Flags (not value judgments, just observations)
    appears_continuous: bool = True   # Activity appears continuous
    appears_progressive: bool = True  # Shows conceptual progression
    has_gaps: bool = False            # Contains observation gaps
    possible_duplication: bool = False  # May contain duplicated content

    # Raw observations
    observations: List[str] = field(default_factory=list)

    # Hash of the validation for integrity
    _result_hash: str = field(default="", repr=False)

    def compute_hash(self) -> str:
        """Compute deterministic hash of validation result"""
        data = {
            "result_id": self.result_id,
            "segment_id": self.segment_id,
            "summary": self.summary,
            "coherence_score": self.coherence_score,
            "progression_score": self.progression_score,
            "consistency_score": self.consistency_score,
            "confidence": self.confidence,
            "observations": sorted(self.observations),
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def result_hash(self) -> str:
        if not self._result_hash:
            self._result_hash = self.compute_hash()
        return self._result_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "segment_id": self.segment_id,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "coherence_score": self.coherence_score,
            "progression_score": self.progression_score,
            "consistency_score": self.consistency_score,
            "summary": self.summary,
            "confidence": self.confidence,
            "uncertainty_notes": self.uncertainty_notes,
            "dissenting_observations": self.dissenting_observations,
            "appears_continuous": self.appears_continuous,
            "appears_progressive": self.appears_progressive,
            "has_gaps": self.has_gaps,
            "possible_duplication": self.possible_duplication,
            "observations": self.observations,
            "result_hash": self.result_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        result = cls(
            result_id=data.get("result_id", str(uuid.uuid4())),
            segment_id=data.get("segment_id", ""),
            metadata=ValidationMetadata.from_dict(data["metadata"]) if data.get("metadata") else None,
            coherence_score=data.get("coherence_score"),
            progression_score=data.get("progression_score"),
            consistency_score=data.get("consistency_score"),
            summary=data.get("summary", ""),
            confidence=data.get("confidence", 0.5),
            uncertainty_notes=data.get("uncertainty_notes", []),
            dissenting_observations=data.get("dissenting_observations", []),
            appears_continuous=data.get("appears_continuous", True),
            appears_progressive=data.get("appears_progressive", True),
            has_gaps=data.get("has_gaps", False),
            possible_duplication=data.get("possible_duplication", False),
            observations=data.get("observations", []),
        )
        result._result_hash = data.get("result_hash", "")
        return result


class Validator:
    """
    LLM-assisted validator for effort segments.

    Analyzes effort segments for coherence and progression without
    making value judgments.
    """

    # Default prompts for validation
    COHERENCE_PROMPT = """Analyze the following signals for linguistic coherence.
Consider:
- Are the signals related to a coherent activity?
- Do they form a logical sequence?
- Is there clear thematic consistency?

Respond with:
1. A coherence score from 0.0 to 1.0
2. Brief observations (2-3 sentences)
3. Any uncertainty or dissent

Signals:
{signals}

Format your response as:
SCORE: [0.0-1.0]
OBSERVATIONS: [observations]
UNCERTAINTY: [any uncertainty notes]"""

    PROGRESSION_PROMPT = """Analyze the following signals for conceptual progression.
Consider:
- Does the activity show development over time?
- Are there indicators of iterative refinement?
- Does later work build on earlier work?

Do NOT judge the value or quality of the work.

Signals:
{signals}

Format your response as:
SCORE: [0.0-1.0]
OBSERVATIONS: [observations]
UNCERTAINTY: [any uncertainty notes]"""

    SUMMARY_PROMPT = """Generate a deterministic summary of the following effort signals.
The summary should:
- Describe WHAT was done (not judge its value)
- Note the time span and signal types
- Be factual and reproducible
- Preserve any ambiguity

Signals:
{signals}

Time span: {start_time} to {end_time}
Signal count: {signal_count}
Signal types: {signal_types}

Provide a 2-4 sentence summary:"""

    def __init__(self, llm_provider=None):
        """
        Initialize validator.

        Args:
            llm_provider: Optional LLM provider for AI-assisted validation.
                         If None, uses rule-based validation only.
        """
        self.llm_provider = llm_provider
        self._validator_id = str(uuid.uuid4())[:8]

    def _get_metadata(self) -> ValidationMetadata:
        """Create metadata for this validation"""
        if self.llm_provider:
            return ValidationMetadata(
                validator_id=self._validator_id,
                model_name=getattr(self.llm_provider, 'name', 'unknown'),
                model_version=getattr(self.llm_provider, 'default_model', ''),
                provider=getattr(self.llm_provider, 'name', 'unknown'),
            )
        return ValidationMetadata(
            validator_id=self._validator_id,
            model_name="rule-based",
            model_version="1.0",
            provider="local",
        )

    def validate(self, segment: EffortSegment) -> ValidationResult:
        """
        Validate an effort segment.

        If LLM provider is available, uses AI-assisted validation.
        Otherwise falls back to rule-based validation.
        """
        if self.llm_provider:
            return self._validate_with_llm(segment)
        return self._validate_rule_based(segment)

    def _validate_rule_based(self, segment: EffortSegment) -> ValidationResult:
        """Rule-based validation without LLM"""
        signals = segment.signals
        observations = []
        uncertainty_notes = []

        # Check for gaps
        has_gaps = False
        if len(signals) > 1:
            for i in range(1, len(signals)):
                gap = (signals[i].timestamp - signals[i-1].timestamp).total_seconds()
                if gap > 300:  # 5 minute gap
                    has_gaps = True
                    observations.append(f"Gap of {int(gap/60)} minutes detected")

        # Check signal type consistency
        signal_types = set(s.signal_type for s in signals)
        if len(signal_types) == 1:
            observations.append(f"Consistent signal type: {list(signal_types)[0].value}")
        else:
            observations.append(f"Mixed signal types: {', '.join(t.value for t in signal_types)}")

        # Check for potential duplication
        content_hashes = [s.content_hash for s in signals]
        unique_hashes = set(content_hashes)
        duplication_ratio = 1 - (len(unique_hashes) / len(content_hashes)) if content_hashes else 0
        possible_duplication = duplication_ratio > 0.3

        if possible_duplication:
            uncertainty_notes.append(f"High content similarity detected ({int(duplication_ratio*100)}%)")

        # Compute basic scores
        coherence = 0.5 if len(signal_types) <= 2 else 0.3
        progression = 0.5 if not has_gaps else 0.3
        consistency = 1.0 - duplication_ratio

        # Generate summary
        duration = segment.duration
        duration_str = f"{int(duration.total_seconds()/60)} minutes" if duration else "unknown duration"
        type_counts = {}
        for s in signals:
            t = s.signal_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        summary = (
            f"Effort segment spanning {duration_str} with {len(signals)} signals. "
            f"Signal types: {', '.join(f'{v} {k}' for k, v in type_counts.items())}. "
        )
        if has_gaps:
            summary += "Contains observation gaps. "
        if possible_duplication:
            summary += "Some content repetition detected."

        return ValidationResult(
            segment_id=segment.segment_id,
            metadata=self._get_metadata(),
            coherence_score=coherence,
            progression_score=progression,
            consistency_score=consistency,
            summary=summary,
            confidence=0.5,  # Rule-based has lower confidence
            uncertainty_notes=uncertainty_notes,
            appears_continuous=not has_gaps,
            appears_progressive=True,  # Can't determine without LLM
            has_gaps=has_gaps,
            possible_duplication=possible_duplication,
            observations=observations,
        )

    def _validate_with_llm(self, segment: EffortSegment) -> ValidationResult:
        """LLM-assisted validation"""
        signals = segment.signals
        metadata = self._get_metadata()

        # Prepare signal text for prompts
        signal_text = self._format_signals(signals)
        signal_types = set(s.signal_type.value for s in signals)

        # Get coherence assessment
        coherence_score, coherence_obs, coherence_unc = self._assess_coherence(signal_text)

        # Get progression assessment
        progression_score, progression_obs, progression_unc = self._assess_progression(signal_text)

        # Generate summary
        summary = self._generate_summary(segment, signal_text)

        # Combine observations
        observations = coherence_obs + progression_obs
        uncertainty_notes = coherence_unc + progression_unc

        # Check for gaps (rule-based)
        has_gaps = False
        if len(signals) > 1:
            for i in range(1, len(signals)):
                gap = (signals[i].timestamp - signals[i-1].timestamp).total_seconds()
                if gap > 300:
                    has_gaps = True
                    break

        # Check for duplication (rule-based)
        content_hashes = [s.content_hash for s in signals]
        unique_hashes = set(content_hashes)
        duplication_ratio = 1 - (len(unique_hashes) / len(content_hashes)) if content_hashes else 0
        possible_duplication = duplication_ratio > 0.3

        # Calculate consistency from duplication
        consistency_score = 1.0 - duplication_ratio

        # Overall confidence
        avg_score = (
            (coherence_score or 0.5) +
            (progression_score or 0.5) +
            (consistency_score or 0.5)
        ) / 3
        confidence = avg_score * 0.8 + 0.1  # Scale to 0.1-0.9

        return ValidationResult(
            segment_id=segment.segment_id,
            metadata=metadata,
            coherence_score=coherence_score,
            progression_score=progression_score,
            consistency_score=consistency_score,
            summary=summary,
            confidence=confidence,
            uncertainty_notes=uncertainty_notes,
            appears_continuous=not has_gaps,
            appears_progressive=progression_score is not None and progression_score > 0.4,
            has_gaps=has_gaps,
            possible_duplication=possible_duplication,
            observations=observations,
        )

    def _format_signals(self, signals: List[Signal], max_signals: int = 20) -> str:
        """Format signals for LLM prompt"""
        # Limit to prevent token overflow
        selected = signals[:max_signals] if len(signals) > max_signals else signals

        lines = []
        for i, s in enumerate(selected):
            timestamp = s.timestamp.strftime("%H:%M:%S")
            content_preview = s.content[:200] + "..." if len(s.content) > 200 else s.content
            lines.append(f"[{i+1}] {timestamp} ({s.signal_type.value}): {content_preview}")

        if len(signals) > max_signals:
            lines.append(f"... and {len(signals) - max_signals} more signals")

        return "\n".join(lines)

    def _assess_coherence(self, signal_text: str) -> tuple:
        """Assess coherence using LLM"""
        try:
            prompt = self.COHERENCE_PROMPT.format(signals=signal_text)
            response = self.llm_provider.complete(prompt)
            return self._parse_assessment(response.content)
        except Exception as e:
            return None, [f"Coherence assessment failed: {e}"], ["Unable to assess coherence"]

    def _assess_progression(self, signal_text: str) -> tuple:
        """Assess progression using LLM"""
        try:
            prompt = self.PROGRESSION_PROMPT.format(signals=signal_text)
            response = self.llm_provider.complete(prompt)
            return self._parse_assessment(response.content)
        except Exception as e:
            return None, [f"Progression assessment failed: {e}"], ["Unable to assess progression"]

    def _generate_summary(self, segment: EffortSegment, signal_text: str) -> str:
        """Generate deterministic summary using LLM"""
        try:
            signal_types = set(s.signal_type.value for s in segment.signals)
            prompt = self.SUMMARY_PROMPT.format(
                signals=signal_text,
                start_time=segment.start_time.isoformat() if segment.start_time else "unknown",
                end_time=segment.end_time.isoformat() if segment.end_time else "unknown",
                signal_count=len(segment.signals),
                signal_types=", ".join(signal_types),
            )
            response = self.llm_provider.complete(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Summary generation failed: {e}"

    def _parse_assessment(self, response: str) -> tuple:
        """Parse LLM assessment response"""
        score = None
        observations = []
        uncertainty = []

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("OBSERVATIONS:"):
                obs = line.replace("OBSERVATIONS:", "").strip()
                if obs:
                    observations.append(obs)
            elif line.startswith("UNCERTAINTY:"):
                unc = line.replace("UNCERTAINTY:", "").strip()
                if unc and unc.lower() not in ["none", "n/a"]:
                    uncertainty.append(unc)

        return score, observations, uncertainty
