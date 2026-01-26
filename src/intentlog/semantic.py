"""
Semantic Features for IntentLog

Provides LLM-powered semantic diff, search, merge, and formalization capabilities.
"""

import math
import hashlib
import json
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .core import Intent
from .llm.provider import LLMProvider, LLMConfig, LLMResponse, EmbeddingResponse


class FormalizationType(Enum):
    """Types of formalized output that can be derived from prose intent"""
    CODE = "code"           # Executable code (Python, JS, etc.)
    RULES = "rules"         # Business rules or constraints
    HEURISTICS = "heuristics"  # Decision-making guidelines
    SCHEMA = "schema"       # Data structures (JSON Schema, TypeScript types)
    CONFIG = "config"       # Configuration files (YAML, TOML, JSON)
    SPEC = "spec"           # Formal specification or requirements
    TESTS = "tests"         # Test cases derived from intent


@dataclass
class ProvenanceRecord:
    """Track the provenance chain from prose to formalized output"""
    source_intent_ids: List[str]
    source_reasoning: str  # Combined reasoning that was formalized
    formalized_at: datetime
    model: str
    formalization_type: FormalizationType
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "source_intent_ids": self.source_intent_ids,
            "source_reasoning": self.source_reasoning,
            "formalized_at": self.formalized_at.isoformat(),
            "model": self.model,
            "formalization_type": self.formalization_type.value,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Create from dictionary.

        Raises:
            ValueError: If required fields are missing or have invalid format
        """
        required_fields = ["source_intent_ids", "source_reasoning", "formalized_at",
                          "model", "formalization_type"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        try:
            formalized_at = datetime.fromisoformat(data["formalized_at"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid formalized_at format: {e}") from e

        try:
            formalization_type = FormalizationType(data["formalization_type"])
        except ValueError as e:
            raise ValueError(f"Invalid formalization_type: {e}") from e

        return cls(
            source_intent_ids=data["source_intent_ids"],
            source_reasoning=data["source_reasoning"],
            formalized_at=formalized_at,
            model=data["model"],
            formalization_type=formalization_type,
            parameters=data.get("parameters", {}),
        )


@dataclass
class FormalizedOutput:
    """Result of formalizing prose intent into code, rules, or heuristics"""
    content: str                    # The formalized output
    formalization_type: FormalizationType
    language: Optional[str]         # Programming language (for CODE type)
    explanation: str                # Why this formalization was derived
    provenance: ProvenanceRecord    # Full provenance chain
    confidence: float = 0.8         # Model's confidence (0-1)
    warnings: List[str] = field(default_factory=list)  # Any caveats or warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content": self.content,
            "formalization_type": self.formalization_type.value,
            "language": self.language,
            "explanation": self.explanation,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormalizedOutput":
        """Create from dictionary"""
        return cls(
            content=data["content"],
            formalization_type=FormalizationType(data["formalization_type"]),
            language=data.get("language"),
            explanation=data["explanation"],
            provenance=ProvenanceRecord.from_dict(data["provenance"]),
            confidence=data.get("confidence", 0.8),
            warnings=data.get("warnings", []),
        )


@dataclass
class SemanticDiff:
    """Result of a semantic diff between intents"""
    summary: str
    changes: List[str]
    intent_a: Intent
    intent_b: Intent
    model: str
    cached: bool = False


@dataclass
class SemanticSearchResult:
    """Result from semantic search"""
    intent: Intent
    score: float  # Similarity score (0-1)
    rank: int


@dataclass
class MergeResolution:
    """Result of LLM-assisted merge resolution"""
    resolved_reasoning: str
    explanation: str
    intent_a: Intent
    intent_b: Intent
    model: str


# Prompt templates
DIFF_SYSTEM_PROMPT = """You are an expert at analyzing changes in human reasoning and intent.
Your task is to compare two intent statements and provide a clear, concise summary of how the reasoning evolved.
Focus on:
1. What changed in the approach or perspective
2. What was added or removed
3. The significance of the change for the project"""

DIFF_PROMPT_TEMPLATE = """Compare these two intent statements and describe the key changes:

EARLIER INTENT ({timestamp_a}):
Name: {name_a}
Reasoning: {reasoning_a}

LATER INTENT ({timestamp_b}):
Name: {name_b}
Reasoning: {reasoning_b}

Provide:
1. A one-sentence summary of the change
2. Key specific changes (as a bulleted list)

Keep your response concise and focused on the evolution of reasoning."""

MERGE_SYSTEM_PROMPT = """You are an expert at resolving conflicts in human reasoning.
Your task is to synthesize two divergent intent statements into a coherent resolution.
Preserve valuable insights from both sides while creating a clear, unified direction."""

MERGE_PROMPT_TEMPLATE = """These two intent statements represent divergent thinking that needs to be reconciled:

INTENT A (from branch '{branch_a}'):
Name: {name_a}
Reasoning: {reasoning_a}

INTENT B (from branch '{branch_b}'):
Name: {name_b}
Reasoning: {reasoning_b}

Provide:
1. A merged reasoning statement that synthesizes both perspectives
2. An explanation of how you reconciled the differences

Format your response as:
MERGED REASONING:
[Your merged statement]

EXPLANATION:
[Your explanation of the reconciliation]"""

# Formalization prompt templates
FORMALIZE_SYSTEM_PROMPT = """You are an expert at deriving formal representations from natural language intent.
Your task is to transform prose reasoning into precise, actionable formal outputs while preserving the original meaning.
Always:
1. Maintain full fidelity to the source intent
2. Note any ambiguities or assumptions made
3. Provide clear rationale for your formalization choices"""

FORMALIZE_CODE_TEMPLATE = """Transform this intent into executable {language} code.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Generate clean, idiomatic {language} code
2. Include comments explaining key decisions
3. Follow best practices for {language}
4. Make the code modular and testable

Format your response as:
CODE:
```{language}
[Your code here]
```

EXPLANATION:
[Explain how the code implements the intent]

CONFIDENCE: [0.0-1.0]
[Your confidence in this implementation]

WARNINGS:
[Any caveats, assumptions, or limitations]"""

FORMALIZE_RULES_TEMPLATE = """Extract formal business rules from this intent.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Express rules in clear, unambiguous language
2. Use consistent structure (IF/WHEN/THEN format)
3. Number each rule for reference
4. Mark required vs optional rules

Format your response as:
RULES:
[Your numbered rules here]

EXPLANATION:
[Explain how these rules capture the intent]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Any ambiguities or edge cases not covered]"""

FORMALIZE_HEURISTICS_TEMPLATE = """Derive decision-making heuristics from this intent.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Express heuristics as actionable guidelines
2. Order by importance/priority
3. Include conditions for when each applies
4. Note trade-offs or exceptions

Format your response as:
HEURISTICS:
[Your ordered heuristics here]

EXPLANATION:
[Explain the reasoning behind these heuristics]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Situations where these heuristics may not apply]"""

FORMALIZE_SCHEMA_TEMPLATE = """Generate a data schema from this intent.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Use {schema_format} format
2. Include all implied entities and relationships
3. Add appropriate constraints and validations
4. Document each field

Format your response as:
SCHEMA:
```{schema_format}
[Your schema here]
```

EXPLANATION:
[Explain the schema design decisions]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Any assumptions about data structure]"""

FORMALIZE_CONFIG_TEMPLATE = """Generate configuration from this intent.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Use {config_format} format
2. Include sensible defaults
3. Comment key configuration options
4. Group related settings logically

Format your response as:
CONFIG:
```{config_format}
[Your configuration here]
```

EXPLANATION:
[Explain the configuration choices]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Any environment-specific considerations]"""

FORMALIZE_SPEC_TEMPLATE = """Create a formal specification from this intent.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Use precise, unambiguous language
2. Define all terms and constraints
3. Include acceptance criteria
4. Note dependencies and assumptions

Format your response as:
SPECIFICATION:
[Your formal specification here]

EXPLANATION:
[Explain the specification choices]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Any areas needing clarification]"""

FORMALIZE_TESTS_TEMPLATE = """Generate test cases from this intent.

INTENT:
Name: {intent_name}
Reasoning: {reasoning}
{context}

Requirements:
1. Generate {language} test code
2. Cover happy path and edge cases
3. Include descriptive test names
4. Group related tests logically

Format your response as:
TESTS:
```{language}
[Your test code here]
```

EXPLANATION:
[Explain the test coverage strategy]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Any scenarios not covered]"""

FORMALIZE_CHAIN_TEMPLATE = """Formalize the combined intent from this chain of reasoning.

INTENT CHAIN (chronological order):
{intent_chain}

Requirements:
1. Synthesize the evolution of reasoning
2. Derive {output_type} from the cumulative intent
3. Note how earlier intents inform later decisions
4. Preserve the narrative context

{type_specific_requirements}

Format your response as:
FORMALIZED OUTPUT:
{output_format_placeholder}

PROVENANCE SUMMARY:
[How the chain of intents led to this formalization]

CONFIDENCE: [0.0-1.0]

WARNINGS:
[Any conflicts or tensions in the chain]"""


class SemanticEngine:
    """
    Engine for semantic operations on intents.

    Provides:
    - Semantic diff between intents
    - Semantic search using embeddings
    - Merge conflict resolution
    - Embedding caching
    - Deferred formalization (prose to code/rules/heuristics)
    """

    def __init__(
        self,
        provider: LLMProvider,
        embedding_provider: Optional[LLMProvider] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize semantic engine.

        Args:
            provider: LLM provider for completions (diff, merge)
            embedding_provider: Provider for embeddings (optional, defaults to provider)
            cache_dir: Directory for caching embeddings (optional)
        """
        self.provider = provider
        self.embedding_provider = embedding_provider or provider
        self.cache_dir = cache_dir
        self._embedding_cache: Dict[str, List[float]] = {}

        if cache_dir:
            self._load_embedding_cache()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _load_embedding_cache(self) -> None:
        """Load embedding cache from disk"""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "embeddings.json"
        if cache_file.is_file():
            try:
                with open(cache_file, "r") as f:
                    self._embedding_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._embedding_cache = {}

    def _save_embedding_cache(self) -> None:
        """Save embedding cache to disk"""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "embeddings.json"
        with open(cache_file, "w") as f:
            json.dump(self._embedding_cache, f)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available"""
        cache_key = self._get_cache_key(text)

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        response = self.embedding_provider.embed(text)
        embedding = response.embedding

        self._embedding_cache[cache_key] = embedding
        if self.cache_dir:
            self._save_embedding_cache()

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if len(a) != len(b):
            raise ValueError("Vectors must have same length")

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _intent_to_text(self, intent: Intent) -> str:
        """Convert intent to text for embedding"""
        return f"{intent.intent_name}: {intent.intent_reasoning}"

    def semantic_diff(
        self,
        intent_a: Intent,
        intent_b: Intent,
    ) -> SemanticDiff:
        """
        Generate semantic diff between two intents.

        Args:
            intent_a: Earlier intent
            intent_b: Later intent

        Returns:
            SemanticDiff with LLM-generated summary and changes
        """
        # Format timestamps
        ts_a = intent_a.timestamp
        ts_b = intent_b.timestamp
        if hasattr(ts_a, "strftime"):
            ts_a = ts_a.strftime("%Y-%m-%d %H:%M")
        if hasattr(ts_b, "strftime"):
            ts_b = ts_b.strftime("%Y-%m-%d %H:%M")

        prompt = DIFF_PROMPT_TEMPLATE.format(
            timestamp_a=ts_a,
            name_a=intent_a.intent_name,
            reasoning_a=intent_a.intent_reasoning,
            timestamp_b=ts_b,
            name_b=intent_b.intent_name,
            reasoning_b=intent_b.intent_reasoning,
        )

        response = self.provider.complete(prompt, system=DIFF_SYSTEM_PROMPT)

        # Parse response into summary and changes
        lines = response.content.strip().split("\n")
        summary = ""
        changes = []

        in_changes = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                in_changes = True
                changes.append(line.lstrip("-•* "))
            elif not in_changes and not summary:
                summary = line
            elif in_changes:
                # Continuation of changes
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    changes.append(line.lstrip("-•* "))

        if not summary:
            summary = response.content.split("\n")[0]

        return SemanticDiff(
            summary=summary,
            changes=changes,
            intent_a=intent_a,
            intent_b=intent_b,
            model=response.model,
        )

    def semantic_search(
        self,
        query: str,
        intents: List[Intent],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[SemanticSearchResult]:
        """
        Search intents using semantic similarity.

        Args:
            query: Natural language search query
            intents: List of intents to search
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of SemanticSearchResult sorted by relevance
        """
        if not intents:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Score all intents
        scored = []
        for intent in intents:
            intent_text = self._intent_to_text(intent)
            intent_embedding = self._get_embedding(intent_text)
            score = self._cosine_similarity(query_embedding, intent_embedding)

            if score >= threshold:
                scored.append((intent, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        results = []
        for rank, (intent, score) in enumerate(scored[:top_k], 1):
            results.append(
                SemanticSearchResult(
                    intent=intent,
                    score=score,
                    rank=rank,
                )
            )

        return results

    def resolve_merge(
        self,
        intent_a: Intent,
        intent_b: Intent,
        branch_a: str = "branch-a",
        branch_b: str = "branch-b",
    ) -> MergeResolution:
        """
        Resolve merge conflict between two intents using LLM.

        Args:
            intent_a: Intent from first branch
            intent_b: Intent from second branch
            branch_a: Name of first branch
            branch_b: Name of second branch

        Returns:
            MergeResolution with synthesized reasoning and explanation
        """
        prompt = MERGE_PROMPT_TEMPLATE.format(
            branch_a=branch_a,
            name_a=intent_a.intent_name,
            reasoning_a=intent_a.intent_reasoning,
            branch_b=branch_b,
            name_b=intent_b.intent_name,
            reasoning_b=intent_b.intent_reasoning,
        )

        response = self.provider.complete(prompt, system=MERGE_SYSTEM_PROMPT)

        # Parse response
        content = response.content
        merged_reasoning = ""
        explanation = ""

        if "MERGED REASONING:" in content:
            parts = content.split("MERGED REASONING:", 1)
            remainder = parts[1] if len(parts) > 1 else ""

            if "EXPLANATION:" in remainder:
                merged_parts = remainder.split("EXPLANATION:", 1)
                merged_reasoning = merged_parts[0].strip()
                explanation = merged_parts[1].strip() if len(merged_parts) > 1 else ""
            else:
                merged_reasoning = remainder.strip()
        else:
            # Fallback: use full response as reasoning
            merged_reasoning = content.strip()

        return MergeResolution(
            resolved_reasoning=merged_reasoning,
            explanation=explanation,
            intent_a=intent_a,
            intent_b=intent_b,
            model=response.model,
        )

    def diff_branches(
        self,
        intents_a: List[Intent],
        intents_b: List[Intent],
        branch_a: str = "main",
        branch_b: str = "branch",
    ) -> List[SemanticDiff]:
        """
        Generate diffs for intents that differ between branches.

        Compares intents by matching IDs or by semantic similarity
        when no exact match exists.

        Args:
            intents_a: Intents from first branch
            intents_b: Intents from second branch
            branch_a: Name of first branch
            branch_b: Name of second branch

        Returns:
            List of SemanticDiff for changed/new intents
        """
        diffs = []

        # Index intents by ID
        a_by_id = {i.intent_id: i for i in intents_a}
        b_by_id = {i.intent_id: i for i in intents_b}

        # Find intents only in B (new intents)
        new_in_b = [i for i in intents_b if i.intent_id not in a_by_id]

        # For new intents, find closest match in A for context
        for intent_b in new_in_b:
            if intents_a:
                # Find most similar intent in A
                best_match = None
                best_score = 0

                b_embedding = self._get_embedding(self._intent_to_text(intent_b))

                for intent_a in intents_a:
                    a_embedding = self._get_embedding(self._intent_to_text(intent_a))
                    score = self._cosine_similarity(a_embedding, b_embedding)
                    if score > best_score:
                        best_score = score
                        best_match = intent_a

                if best_match and best_score > 0.5:
                    diff = self.semantic_diff(best_match, intent_b)
                    diffs.append(diff)

        return diffs

    def _get_formalization_template(
        self,
        formalization_type: FormalizationType,
    ) -> str:
        """Get the appropriate prompt template for a formalization type"""
        templates = {
            FormalizationType.CODE: FORMALIZE_CODE_TEMPLATE,
            FormalizationType.RULES: FORMALIZE_RULES_TEMPLATE,
            FormalizationType.HEURISTICS: FORMALIZE_HEURISTICS_TEMPLATE,
            FormalizationType.SCHEMA: FORMALIZE_SCHEMA_TEMPLATE,
            FormalizationType.CONFIG: FORMALIZE_CONFIG_TEMPLATE,
            FormalizationType.SPEC: FORMALIZE_SPEC_TEMPLATE,
            FormalizationType.TESTS: FORMALIZE_TESTS_TEMPLATE,
        }
        return templates.get(formalization_type, FORMALIZE_CODE_TEMPLATE)

    def _parse_formalized_response(
        self,
        content: str,
        formalization_type: FormalizationType,
    ) -> Tuple[str, str, float, List[str]]:
        """
        Parse LLM response for formalization.

        Returns:
            Tuple of (formalized_content, explanation, confidence, warnings)
        """
        formalized = ""
        explanation = ""
        confidence = 0.8
        warnings = []

        # Define section markers based on type
        content_markers = {
            FormalizationType.CODE: "CODE:",
            FormalizationType.RULES: "RULES:",
            FormalizationType.HEURISTICS: "HEURISTICS:",
            FormalizationType.SCHEMA: "SCHEMA:",
            FormalizationType.CONFIG: "CONFIG:",
            FormalizationType.SPEC: "SPECIFICATION:",
            FormalizationType.TESTS: "TESTS:",
        }

        content_marker = content_markers.get(formalization_type, "CODE:")

        # Parse content section
        if content_marker in content:
            parts = content.split(content_marker, 1)
            remainder = parts[1] if len(parts) > 1 else ""

            # Check for code blocks
            if "```" in remainder:
                # Extract code between backticks
                code_parts = remainder.split("```")
                if len(code_parts) >= 2:
                    # Handle language specifier in code block
                    code_block = code_parts[1]
                    if "\n" in code_block:
                        # Skip language line if present
                        first_line, rest = code_block.split("\n", 1)
                        if not first_line.strip().startswith(("def ", "class ", "import ", "from ")):
                            code_block = rest
                    formalized = code_block.strip()
                    remainder = "```".join(code_parts[2:])
            else:
                # No code block, extract until next section
                for section in ["EXPLANATION:", "CONFIDENCE:", "WARNINGS:"]:
                    if section in remainder:
                        formalized, remainder = remainder.split(section, 1)
                        remainder = section + remainder
                        break
                else:
                    formalized = remainder
                    remainder = ""
                formalized = formalized.strip()
        else:
            # Fallback: use entire content
            formalized = content.strip()

        # Parse explanation
        if "EXPLANATION:" in content:
            parts = content.split("EXPLANATION:", 1)
            exp_text = parts[1] if len(parts) > 1 else ""
            for section in ["CONFIDENCE:", "WARNINGS:"]:
                if section in exp_text:
                    explanation, _ = exp_text.split(section, 1)
                    break
            else:
                explanation = exp_text
            explanation = explanation.strip()

        # Parse confidence
        if "CONFIDENCE:" in content:
            parts = content.split("CONFIDENCE:", 1)
            conf_text = parts[1] if len(parts) > 1 else ""
            # Extract number
            import re
            match = re.search(r"(\d+\.?\d*)", conf_text)
            if match:
                try:
                    confidence = float(match.group(1))
                    if confidence > 1.0:
                        confidence /= 100.0  # Handle percentage
                except ValueError:
                    confidence = 0.8

        # Parse warnings
        if "WARNINGS:" in content:
            parts = content.split("WARNINGS:", 1)
            warn_text = parts[1] if len(parts) > 1 else ""
            # Extract lines that look like warnings
            for line in warn_text.strip().split("\n"):
                line = line.strip()
                if line and line not in ["None", "N/A", "-"]:
                    warnings.append(line.lstrip("-•* "))

        return formalized, explanation, confidence, warnings

    def formalize(
        self,
        intent: Intent,
        formalization_type: FormalizationType = FormalizationType.CODE,
        language: Optional[str] = None,
        context: Optional[str] = None,
        additional_context: Optional[List[Intent]] = None,
    ) -> FormalizedOutput:
        """
        Derive formal code, rules, or heuristics from prose intent.

        This implements "deferred formalization" - the ability to keep intent
        in prose form and derive formal representations on demand, with full
        provenance tracking back to the source narrative.

        Args:
            intent: The intent to formalize
            formalization_type: Type of output (CODE, RULES, HEURISTICS, etc.)
            language: Programming language (for CODE/TESTS types)
            context: Additional context string to include
            additional_context: Related intents for context

        Returns:
            FormalizedOutput with code/rules and provenance
        """
        # Determine language for code types
        if formalization_type in (FormalizationType.CODE, FormalizationType.TESTS):
            language = language or "python"
        elif formalization_type == FormalizationType.SCHEMA:
            language = language or "json_schema"
        elif formalization_type == FormalizationType.CONFIG:
            language = language or "yaml"

        # Build context from additional intents
        context_str = ""
        if context:
            context_str = f"\nAdditional Context:\n{context}\n"
        if additional_context:
            context_parts = ["\nRelated Intents:"]
            for i, ctx_intent in enumerate(additional_context, 1):
                context_parts.append(
                    f"{i}. [{ctx_intent.intent_name}]: {ctx_intent.intent_reasoning}"
                )
            context_str += "\n".join(context_parts)

        # Get template and format prompt
        template = self._get_formalization_template(formalization_type)

        # Format based on type
        if formalization_type == FormalizationType.SCHEMA:
            prompt = template.format(
                intent_name=intent.intent_name,
                reasoning=intent.intent_reasoning,
                context=context_str,
                schema_format=language,
            )
        elif formalization_type == FormalizationType.CONFIG:
            prompt = template.format(
                intent_name=intent.intent_name,
                reasoning=intent.intent_reasoning,
                context=context_str,
                config_format=language,
            )
        else:
            prompt = template.format(
                intent_name=intent.intent_name,
                reasoning=intent.intent_reasoning,
                context=context_str,
                language=language or "python",
            )

        # Call LLM
        response = self.provider.complete(prompt, system=FORMALIZE_SYSTEM_PROMPT)

        # Parse response
        formalized, explanation, confidence, warnings = self._parse_formalized_response(
            response.content, formalization_type
        )

        # Build provenance record
        source_ids = [intent.intent_id]
        if additional_context:
            source_ids.extend([i.intent_id for i in additional_context])

        provenance = ProvenanceRecord(
            source_intent_ids=source_ids,
            source_reasoning=intent.intent_reasoning,
            formalized_at=datetime.now(),
            model=response.model,
            formalization_type=formalization_type,
            parameters={
                "language": language,
                "has_additional_context": bool(additional_context),
            },
        )

        return FormalizedOutput(
            content=formalized,
            formalization_type=formalization_type,
            language=language,
            explanation=explanation,
            provenance=provenance,
            confidence=confidence,
            warnings=warnings,
        )

    def formalize_chain(
        self,
        intents: List[Intent],
        formalization_type: FormalizationType = FormalizationType.CODE,
        language: Optional[str] = None,
    ) -> FormalizedOutput:
        """
        Formalize from a chain of intents, synthesizing their combined reasoning.

        This allows deriving formal output from an evolution of intent,
        capturing how decisions were refined over time.

        Args:
            intents: List of intents in chronological order
            formalization_type: Type of output to generate
            language: Programming language (for CODE/TESTS types)

        Returns:
            FormalizedOutput with synthesized formalization
        """
        if not intents:
            raise ValueError("At least one intent is required")

        # Determine language
        if formalization_type in (FormalizationType.CODE, FormalizationType.TESTS):
            language = language or "python"
        elif formalization_type == FormalizationType.SCHEMA:
            language = language or "json_schema"
        elif formalization_type == FormalizationType.CONFIG:
            language = language or "yaml"

        # Build intent chain representation
        chain_parts = []
        for i, intent in enumerate(intents, 1):
            ts = intent.timestamp
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%Y-%m-%d %H:%M")
            chain_parts.append(
                f"[{i}] {ts}\n"
                f"    Name: {intent.intent_name}\n"
                f"    Reasoning: {intent.intent_reasoning}"
            )
        intent_chain = "\n\n".join(chain_parts)

        # Determine output format based on type
        output_type = formalization_type.value.upper()
        type_requirements = ""
        output_format = "[Your formalized output here]"

        if formalization_type == FormalizationType.CODE:
            type_requirements = f"Generate clean, idiomatic {language} code with comments."
            output_format = f"```{language}\n[Your code here]\n```"
        elif formalization_type == FormalizationType.RULES:
            type_requirements = "Express as numbered IF/WHEN/THEN rules."
            output_format = "[Numbered rules]"
        elif formalization_type == FormalizationType.HEURISTICS:
            type_requirements = "Express as ordered actionable guidelines."
            output_format = "[Ordered heuristics]"
        elif formalization_type == FormalizationType.SCHEMA:
            type_requirements = f"Generate {language} schema with documentation."
            output_format = f"```{language}\n[Your schema here]\n```"
        elif formalization_type == FormalizationType.CONFIG:
            type_requirements = f"Generate {language} configuration with comments."
            output_format = f"```{language}\n[Your config here]\n```"
        elif formalization_type == FormalizationType.SPEC:
            type_requirements = "Create formal specification with acceptance criteria."
            output_format = "[Formal specification]"
        elif formalization_type == FormalizationType.TESTS:
            type_requirements = f"Generate {language} test code covering the requirements."
            output_format = f"```{language}\n[Your tests here]\n```"

        prompt = FORMALIZE_CHAIN_TEMPLATE.format(
            intent_chain=intent_chain,
            output_type=output_type,
            type_specific_requirements=type_requirements,
            output_format_placeholder=output_format,
        )

        # Call LLM
        response = self.provider.complete(prompt, system=FORMALIZE_SYSTEM_PROMPT)

        # Parse response (use "FORMALIZED OUTPUT:" as marker)
        content = response.content
        formalized = ""
        explanation = ""
        confidence = 0.8
        warnings = []

        if "FORMALIZED OUTPUT:" in content:
            parts = content.split("FORMALIZED OUTPUT:", 1)
            remainder = parts[1] if len(parts) > 1 else ""

            # Check for code blocks
            if "```" in remainder:
                code_parts = remainder.split("```")
                if len(code_parts) >= 2:
                    code_block = code_parts[1]
                    if "\n" in code_block:
                        first_line, rest = code_block.split("\n", 1)
                        if not first_line.strip().startswith(("def ", "class ", "import ")):
                            code_block = rest
                    formalized = code_block.strip()
            else:
                for section in ["PROVENANCE SUMMARY:", "CONFIDENCE:", "WARNINGS:"]:
                    if section in remainder:
                        formalized, _ = remainder.split(section, 1)
                        break
                else:
                    formalized = remainder
                formalized = formalized.strip()

        if "PROVENANCE SUMMARY:" in content:
            parts = content.split("PROVENANCE SUMMARY:", 1)
            exp_text = parts[1] if len(parts) > 1 else ""
            for section in ["CONFIDENCE:", "WARNINGS:"]:
                if section in exp_text:
                    explanation, _ = exp_text.split(section, 1)
                    break
            else:
                explanation = exp_text
            explanation = explanation.strip()

        if "CONFIDENCE:" in content:
            import re
            parts = content.split("CONFIDENCE:", 1)
            if len(parts) > 1:
                match = re.search(r"(\d+\.?\d*)", parts[1])
                if match:
                    try:
                        confidence = float(match.group(1))
                        if confidence > 1.0:
                            confidence /= 100.0
                    except ValueError:
                        pass

        if "WARNINGS:" in content:
            parts = content.split("WARNINGS:", 1)
            if len(parts) > 1:
                for line in parts[1].strip().split("\n"):
                    line = line.strip()
                    if line and line not in ["None", "N/A", "-"]:
                        warnings.append(line.lstrip("-•* "))

        # Combine reasoning from all intents
        combined_reasoning = " → ".join([i.intent_reasoning for i in intents])

        # Build provenance record
        provenance = ProvenanceRecord(
            source_intent_ids=[i.intent_id for i in intents],
            source_reasoning=combined_reasoning,
            formalized_at=datetime.now(),
            model=response.model,
            formalization_type=formalization_type,
            parameters={
                "language": language,
                "chain_length": len(intents),
            },
        )

        return FormalizedOutput(
            content=formalized,
            formalization_type=formalization_type,
            language=language,
            explanation=explanation,
            provenance=provenance,
            confidence=confidence,
            warnings=warnings,
        )

    def formalize_from_search(
        self,
        query: str,
        intents: List[Intent],
        formalization_type: FormalizationType = FormalizationType.CODE,
        language: Optional[str] = None,
        top_k: int = 3,
    ) -> FormalizedOutput:
        """
        Search for relevant intents and formalize the combined results.

        Useful when you want to derive code/rules from related intents
        without specifying exact intent IDs.

        Args:
            query: Natural language query to find relevant intents
            intents: All intents to search through
            formalization_type: Type of output to generate
            language: Programming language (for CODE/TESTS types)
            top_k: Number of top matching intents to use

        Returns:
            FormalizedOutput derived from matching intents
        """
        # Find relevant intents
        results = self.semantic_search(query, intents, top_k=top_k)

        if not results:
            raise ValueError(f"No intents found matching query: {query}")

        # Use the matched intents for formalization
        matched_intents = [r.intent for r in results]

        if len(matched_intents) == 1:
            return self.formalize(
                matched_intents[0],
                formalization_type=formalization_type,
                language=language,
            )
        else:
            return self.formalize_chain(
                matched_intents,
                formalization_type=formalization_type,
                language=language,
            )
