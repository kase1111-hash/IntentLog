"""
Semantic Features for IntentLog

Provides LLM-powered semantic diff, search, and merge capabilities.
"""

import math
import hashlib
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .core import Intent
from .llm.provider import LLMProvider, LLMConfig, LLMResponse, EmbeddingResponse


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


class SemanticEngine:
    """
    Engine for semantic operations on intents.

    Provides:
    - Semantic diff between intents
    - Semantic search using embeddings
    - Merge conflict resolution
    - Embedding caching
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
