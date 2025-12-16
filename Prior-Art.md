Design Note & Prior Art (December 2025)

IntentLog is presented as a conceptual design that explores version control for human reasoning—specifically, treating natural-language intent as a first-class, versioned artifact alongside (but not replacing) code or other formal outputs.

This repository documents several core ideas that, to our knowledge, are not jointly articulated in existing version control or collaboration tools:

Prose-first commits: Versioned, cryptographically signed natural-language explanations of why a change was made, with code or other artifacts treated as attachments rather than the primary unit of history.

Merges via explanation: Resolving divergent branches through an explicit narrative commit that records trade-offs and rationale, rather than selecting one syntactic outcome.

Semantic diffs of intent: Using language models to summarize and contrast changes in reasoning over time, while preserving the underlying human-authored prose as the authoritative record.

Deferred formalization: Allowing ambiguous or high-level intent to remain in natural language, with heuristics, rules, or code derived later—always traceable back to the original narrative source.

Precedent trails: Treating prior decisions as referential context (similar to case law), enabling visible chains of reasoning rather than static, disconnected decision records.

These ideas are intentionally described at the level of design primitives, not as a finalized implementation. The goal is to establish a clear reference point for a growing design space around language-native tooling, human-AI co-reasoning, and interpretable system evolution.

We expect—and welcome—independent implementations, alternative architectures, and critical extensions of these concepts. If future systems converge on similar patterns, this repository may serve as a useful historical marker for how this problem was framed and decomposed at this time.

Initial public release and timestamp: December 16, 2025
License: CC BY-SA 4.0 (reuse encouraged with attribution)
