"""
LLM-Based Intent Classification

Provides semantic classification of intents using LLM understanding,
replacing simple keyword-based classification with nuanced analysis.

This module integrates with the Memory Vault classification system
to provide intelligent routing of intents based on their content.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from ..core import Intent
from ..llm.provider import LLMProvider, LLMConfig


class IntentCategory(Enum):
    """Semantic categories for intent classification"""
    TRANSIENT = "transient"              # Low-value, ephemeral
    LEARNED_HEURISTIC = "learned"        # Learned patterns and insights
    FAILED_PATH_LESSON = "failure"       # Lessons from failures
    LONG_TERM_GOAL = "strategic"         # Strategic goals and principles
    RECOVERY_SEED = "critical"           # Critical recovery information
    ARCHITECTURE = "architecture"        # Architectural decisions
    SECURITY = "security"               # Security-related decisions
    COMPLIANCE = "compliance"           # Regulatory/compliance decisions
    PERFORMANCE = "performance"         # Performance optimizations
    USER_EXPERIENCE = "ux"              # UX/UI decisions


@dataclass
class ClassificationResult:
    """Result of LLM-based intent classification"""
    category: IntentCategory
    confidence: float                   # 0.0 to 1.0
    classification_level: int           # Memory Vault level (0-5)
    reasoning: str                      # Why this classification
    secondary_categories: List[IntentCategory] = field(default_factory=list)
    sensitivity_score: float = 0.0      # 0.0 to 1.0, higher = more sensitive
    retention_priority: str = "normal"  # low, normal, high, critical
    suggested_tags: List[str] = field(default_factory=list)
    model: str = ""
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "classification_level": self.classification_level,
            "reasoning": self.reasoning,
            "secondary_categories": [c.value for c in self.secondary_categories],
            "sensitivity_score": self.sensitivity_score,
            "retention_priority": self.retention_priority,
            "suggested_tags": self.suggested_tags,
            "model": self.model,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationResult":
        """Create from dictionary"""
        return cls(
            category=IntentCategory(data["category"]),
            confidence=data["confidence"],
            classification_level=data["classification_level"],
            reasoning=data["reasoning"],
            secondary_categories=[IntentCategory(c) for c in data.get("secondary_categories", [])],
            sensitivity_score=data.get("sensitivity_score", 0.0),
            retention_priority=data.get("retention_priority", "normal"),
            suggested_tags=data.get("suggested_tags", []),
            model=data.get("model", ""),
            cached=data.get("cached", False),
        )


# Classification prompt template
CLASSIFICATION_SYSTEM_PROMPT = """You are an expert at analyzing intent statements and classifying them for long-term storage and retrieval.

Your task is to classify an intent based on its name, reasoning, and context. Consider:
1. The strategic importance of the decision
2. Whether it contains sensitive or critical information
3. Its long-term relevance for the project
4. Whether it represents a lesson learned or a failure analysis
5. Its connection to architecture, security, compliance, or performance

Classification levels (Memory Vault compatible):
- Level 0 (TRANSIENT): Low-value, ephemeral intents that don't need long-term storage
- Level 1 (LEARNED_HEURISTIC): Learned patterns, insights, best practices
- Level 2 (FAILED_PATH_LESSON): Lessons from failures, what didn't work
- Level 3 (LONG_TERM_GOAL): Strategic goals, principles, mission-critical decisions
- Level 5 (RECOVERY_SEED): Critical recovery info, credentials, keys, seeds

Categories for semantic understanding:
- transient: Routine operations, temporary decisions
- learned: Insights and patterns discovered through experience
- failure: Analysis of what went wrong, lessons learned
- strategic: Long-term goals, vision, principles
- critical: Highly sensitive recovery or security information
- architecture: System design, technical decisions
- security: Security-related decisions and policies
- compliance: Regulatory or compliance requirements
- performance: Performance optimizations and trade-offs
- ux: User experience and interface decisions"""

CLASSIFICATION_PROMPT = """Classify this intent:

INTENT NAME: {intent_name}
REASONING: {intent_reasoning}
{context}

Analyze the intent and provide classification in this exact format:

CATEGORY: [one of: transient, learned, failure, strategic, critical, architecture, security, compliance, performance, ux]
CLASSIFICATION_LEVEL: [0, 1, 2, 3, or 5]
CONFIDENCE: [0.0 to 1.0]
SENSITIVITY: [0.0 to 1.0]
RETENTION: [low, normal, high, critical]
SECONDARY_CATEGORIES: [comma-separated list or "none"]
SUGGESTED_TAGS: [comma-separated list or "none"]

REASONING:
[Explain why you chose this classification in 1-2 sentences]"""


class LLMIntentClassifier:
    """
    LLM-powered intent classification engine.

    Uses semantic understanding to classify intents more accurately
    than keyword-based approaches, while remaining compatible with
    the Memory Vault classification system.
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache_classifications: bool = True,
        fallback_to_keywords: bool = True,
    ):
        """
        Initialize the LLM classifier.

        Args:
            provider: LLM provider for classification
            cache_classifications: Whether to cache results
            fallback_to_keywords: Fall back to keyword matching if LLM fails
        """
        self.provider = provider
        self.cache_classifications = cache_classifications
        self.fallback_to_keywords = fallback_to_keywords
        self._cache: Dict[str, ClassificationResult] = {}

    def _get_cache_key(self, intent_name: str, intent_reasoning: str) -> str:
        """Generate cache key for classification"""
        import hashlib
        content = f"{intent_name}:{intent_reasoning}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _parse_classification_response(self, content: str) -> Tuple[
        str, int, float, float, str, List[str], List[str], str
    ]:
        """
        Parse LLM classification response.

        Returns:
            Tuple of (category, level, confidence, sensitivity, retention,
                     secondary_categories, tags, reasoning)
        """
        category = "transient"
        level = 0
        confidence = 0.5
        sensitivity = 0.0
        retention = "normal"
        secondary = []
        tags = []
        reasoning = ""

        lines = content.strip().split("\n")
        in_reasoning = False
        reasoning_lines = []

        for line in lines:
            line = line.strip()

            if in_reasoning:
                reasoning_lines.append(line)
                continue

            if line.startswith("CATEGORY:"):
                cat = line.replace("CATEGORY:", "").strip().lower()
                if cat in [c.value for c in IntentCategory]:
                    category = cat

            elif line.startswith("CLASSIFICATION_LEVEL:"):
                try:
                    lvl = int(line.replace("CLASSIFICATION_LEVEL:", "").strip())
                    if lvl in [0, 1, 2, 3, 5]:
                        level = lvl
                except ValueError:
                    pass

            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                    confidence = max(0.0, min(1.0, conf))
                except ValueError:
                    pass

            elif line.startswith("SENSITIVITY:"):
                try:
                    sens = float(line.replace("SENSITIVITY:", "").strip())
                    sensitivity = max(0.0, min(1.0, sens))
                except ValueError:
                    pass

            elif line.startswith("RETENTION:"):
                ret = line.replace("RETENTION:", "").strip().lower()
                if ret in ["low", "normal", "high", "critical"]:
                    retention = ret

            elif line.startswith("SECONDARY_CATEGORIES:"):
                cats = line.replace("SECONDARY_CATEGORIES:", "").strip()
                if cats.lower() != "none":
                    for c in cats.split(","):
                        c = c.strip().lower()
                        if c in [cat.value for cat in IntentCategory]:
                            secondary.append(c)

            elif line.startswith("SUGGESTED_TAGS:"):
                tag_str = line.replace("SUGGESTED_TAGS:", "").strip()
                if tag_str.lower() != "none":
                    tags = [t.strip() for t in tag_str.split(",") if t.strip()]

            elif line.startswith("REASONING:"):
                in_reasoning = True

        reasoning = " ".join(reasoning_lines).strip()

        return category, level, confidence, sensitivity, retention, secondary, tags, reasoning

    def _keyword_fallback(
        self,
        intent_name: str,
        intent_reasoning: str,
    ) -> ClassificationResult:
        """
        Keyword-based fallback classification.

        Compatible with the existing Memory Vault integration.
        """
        name_lower = intent_name.lower()
        reasoning_lower = intent_reasoning.lower()
        combined = f"{name_lower} {reasoning_lower}"

        # Critical keywords (Level 5)
        critical_keywords = ["seed", "key", "password", "credential", "recovery", "secret", "token"]
        if any(kw in combined for kw in critical_keywords):
            return ClassificationResult(
                category=IntentCategory.RECOVERY_SEED,
                confidence=0.7,
                classification_level=5,
                reasoning="Contains critical security keywords (keyword fallback)",
                sensitivity_score=1.0,
                retention_priority="critical",
                suggested_tags=["security", "critical"],
            )

        # Strategic keywords (Level 3)
        strategic_keywords = ["goal", "principle", "strategy", "mission", "vision", "roadmap"]
        if any(kw in combined for kw in strategic_keywords):
            return ClassificationResult(
                category=IntentCategory.LONG_TERM_GOAL,
                confidence=0.6,
                classification_level=3,
                reasoning="Contains strategic planning keywords (keyword fallback)",
                retention_priority="high",
                suggested_tags=["strategic"],
            )

        # Failure/lesson keywords (Level 2)
        failure_keywords = ["failed", "lesson", "mistake", "error", "wrong", "fix", "bug"]
        if any(kw in combined for kw in failure_keywords):
            return ClassificationResult(
                category=IntentCategory.FAILED_PATH_LESSON,
                confidence=0.6,
                classification_level=2,
                reasoning="Contains failure/lesson keywords (keyword fallback)",
                retention_priority="high",
                suggested_tags=["lesson-learned"],
            )

        # Learning keywords (Level 1)
        learning_keywords = ["learned", "heuristic", "pattern", "insight", "discovered"]
        if any(kw in combined for kw in learning_keywords):
            return ClassificationResult(
                category=IntentCategory.LEARNED_HEURISTIC,
                confidence=0.6,
                classification_level=1,
                reasoning="Contains learning pattern keywords (keyword fallback)",
                retention_priority="normal",
                suggested_tags=["heuristic"],
            )

        # Architecture keywords
        arch_keywords = ["architecture", "design", "structure", "component", "module", "system"]
        if any(kw in combined for kw in arch_keywords):
            return ClassificationResult(
                category=IntentCategory.ARCHITECTURE,
                confidence=0.5,
                classification_level=2,
                reasoning="Contains architecture keywords (keyword fallback)",
                retention_priority="high",
                suggested_tags=["architecture"],
            )

        # Security keywords (but not critical)
        security_keywords = ["security", "auth", "permission", "access", "encrypt"]
        if any(kw in combined for kw in security_keywords):
            return ClassificationResult(
                category=IntentCategory.SECURITY,
                confidence=0.5,
                classification_level=2,
                reasoning="Contains security keywords (keyword fallback)",
                sensitivity_score=0.6,
                retention_priority="high",
                suggested_tags=["security"],
            )

        # Default to transient
        return ClassificationResult(
            category=IntentCategory.TRANSIENT,
            confidence=0.5,
            classification_level=0,
            reasoning="No significant keywords detected (keyword fallback)",
            retention_priority="low",
        )

    def classify(
        self,
        intent_name: str,
        intent_reasoning: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        Classify an intent using LLM understanding.

        Args:
            intent_name: Name of the intent
            intent_reasoning: Reasoning content
            context: Additional context for classification
            metadata: Additional metadata

        Returns:
            ClassificationResult with semantic classification
        """
        # Check cache
        if self.cache_classifications:
            cache_key = self._get_cache_key(intent_name, intent_reasoning)
            if cache_key in self._cache:
                result = self._cache[cache_key]
                result.cached = True
                return result

        # Build context string
        context_str = ""
        if context:
            context_str = f"\nCONTEXT: {context}"
        if metadata:
            context_str += f"\nMETADATA: {metadata}"

        # Build prompt
        prompt = CLASSIFICATION_PROMPT.format(
            intent_name=intent_name,
            intent_reasoning=intent_reasoning,
            context=context_str,
        )

        try:
            # Call LLM
            response = self.provider.complete(prompt, system=CLASSIFICATION_SYSTEM_PROMPT)

            # Parse response
            (category, level, confidence, sensitivity,
             retention, secondary, tags, reasoning) = self._parse_classification_response(
                response.content
            )

            # Build result
            result = ClassificationResult(
                category=IntentCategory(category),
                confidence=confidence,
                classification_level=level,
                reasoning=reasoning or "LLM classification",
                secondary_categories=[IntentCategory(c) for c in secondary],
                sensitivity_score=sensitivity,
                retention_priority=retention,
                suggested_tags=tags,
                model=response.model,
                cached=False,
            )

            # Cache result
            if self.cache_classifications:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            # Fall back to keyword classification if configured
            if self.fallback_to_keywords:
                result = self._keyword_fallback(intent_name, intent_reasoning)
                result.reasoning = f"Fallback due to LLM error: {str(e)[:50]}"
                return result
            raise

    def classify_intent(self, intent: Intent, context: Optional[str] = None) -> ClassificationResult:
        """
        Classify an Intent object.

        Args:
            intent: The intent to classify
            context: Additional context

        Returns:
            ClassificationResult
        """
        return self.classify(
            intent_name=intent.intent_name,
            intent_reasoning=intent.intent_reasoning,
            context=context,
            metadata=intent.metadata,
        )

    def batch_classify(
        self,
        intents: List[Intent],
        context: Optional[str] = None,
    ) -> List[ClassificationResult]:
        """
        Classify multiple intents.

        Args:
            intents: List of intents to classify
            context: Shared context for all

        Returns:
            List of ClassificationResult objects
        """
        return [self.classify_intent(intent, context) for intent in intents]

    def get_memory_vault_level(self, result: ClassificationResult) -> int:
        """
        Get Memory Vault compatible classification level.

        Args:
            result: Classification result

        Returns:
            Integer level (0-5)
        """
        return result.classification_level

    def should_persist(self, result: ClassificationResult) -> bool:
        """
        Determine if intent should be persisted based on classification.

        Args:
            result: Classification result

        Returns:
            True if should be persisted
        """
        # Persist anything above transient level
        return result.classification_level > 0

    def should_use_vault(self, result: ClassificationResult) -> bool:
        """
        Determine if intent should use Memory Vault.

        Args:
            result: Classification result

        Returns:
            True if should use Memory Vault
        """
        # Use vault for level 2 and above (matching existing behavior)
        return result.classification_level >= 2

    def clear_cache(self) -> None:
        """Clear the classification cache"""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "enabled": self.cache_classifications,
        }


# Convenience function for one-off classification
def classify_intent_with_llm(
    provider: LLMProvider,
    intent_name: str,
    intent_reasoning: str,
    context: Optional[str] = None,
) -> ClassificationResult:
    """
    Convenience function to classify an intent with LLM.

    Args:
        provider: LLM provider
        intent_name: Intent name
        intent_reasoning: Intent reasoning
        context: Optional context

    Returns:
        ClassificationResult
    """
    classifier = LLMIntentClassifier(provider, cache_classifications=False)
    return classifier.classify(intent_name, intent_reasoning, context)
