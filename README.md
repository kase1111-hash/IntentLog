IntentLog: Version Control for Human Reasoning
What if Git tracked why, not just what?

Note: This is a conceptual design released for collaborative development. Reference implementation coming Q1 2026. Feedback and prototype contributions welcome.

Most version control systems excel at capturing syntactic changes—what lines were added, deleted, or modified—but they remain blind to the human reasoning behind those changes. Decisions scatter across chat threads, emails, meeting notes, and forgotten comments, leaving future contributors to reverse-engineer intent from cryptic diffs and commit messages. IntentLog introduces semantic versioning for reasoning itself: prose commits that preserve narrative context, LLM-powered explanations of changes, and traceable evolution of ideas. By making natural language the first-class artifact, it turns decision-making into an auditable, branchable, and interpretable history—bridging the gap between code (or any collaborative artifact) and the human discourse that shapes it.
The Problem Developers Already Have

Decision context gets lost: A refactor happens, but the rationale—"we chose this pattern to reduce latency under high concurrency"—vanishes into ephemeral Slack threads or PR comments that fade over time.
"Why did we do this?" lives in old Slack threads: Onboarding new team members becomes archaeological work; debugging legacy choices feels like guesswork.
ADRs exist but aren't connected to evolution: Architecture Decision Records are static documents, disconnected from the actual commits they justify, making it hard to see how decisions adapted as reality unfolded.

These issues aren't unique to code—they plague any collaborative knowledge work where intent matters more than raw output.
The Solution
IntentLog is Git, but for intent.

Commits are prose: Each commit is a short natural-language explanation of why something changed, cryptographically signed and timestamped. Code (or other files) can be attached, but the prose is the primary artifact.
Branches for exploration: Experiment with alternative directions ("what if we used event sourcing?") without polluting the main history.
Semantic diffs, not just syntactic: Instead of line-by-line changes, IntentLog uses LLMs to generate readable summaries: "This revision shifts from optimistic concurrency to pessimistic locking to prevent race conditions in multi-region deployments."
Merges via explanation: Conflicts aren't resolved by picking sides—they're reconciled with a new prose commit that narrates the trade-offs and final rationale.
Deferred formalization: Ambiguous intent stays in prose; LLMs can derive heuristics, rules, or even code on demand, with full provenance back to the source narrative.
Precedent trails: Every commit can reference prior ones, building a visible chain of reasoning—like case law for your project.

Under the hood: Merkle trees for integrity, optional decentralized storage, and pluggable LLM backends for semantic features.
Quick Start: Prototyping Ideas
Get started in minutes with a local prototype (open-source repo coming soon).
bash# Install (pip or brew placeholder)
pip install intentlog

# Initialize a project
ilog init my-project

# Make your first intent commit
ilog commit "We're starting with a monolithic repo because the team is small and we need fast iteration. We'll revisit splitting services once we hit scaling pain."

# Attach code/files if desired
git add .          # or any files
ilog commit "Adding user authentication module" --attach

# Branch for an experiment
ilog branch experimental-event-sourcing
ilog commit "Exploring event sourcing for better auditability—pros: immutable history; cons: steeper learning curve."

# See semantic history
ilog log            # Shows narrative timeline
ilog diff main..experimental-event-sourcing
# → "The experimental branch introduces event sourcing to preserve full history, 
#    trading simplicity for long-term audit benefits."

# Query past reasoning semantically
ilog search "why did we choose monolithic over microservices"
# → Returns relevant commits with context, even if exact words differ

# Merge with explanation
ilog merge main --message "After prototyping, we're sticking with relational model for now—event sourcing adds complexity without immediate payoff. Revisit in Q3."
Play with it locally—no blockchain, no network required for solo/team use.
Use Cases
For Development Teams

Architecture decision tracking: Every major refactor or tech choice gets a living, branchable rationale.
AI agent instruction evolution: Track how prompts and system instructions evolve as agents improve—critical for debugging emergent behavior.
Open source governance: Proposals, RFCs, and constitution changes live as mergeable prose histories.

Beyond Code

Research notebooks: Scientific reasoning preserved alongside data/experiments; failed hypotheses kept as branches for negative precedent.
Policy documents: Organizational policies evolve with visible debate trails—perfect for DAOs, co-ops, or compliance-heavy teams.
Creative applications: Screenwriting teams tracking plot motivations; design systems preserving "why this shade of blue?"; collaborative world-building for games or fiction.

Why This Matters Now

In a world where AI can produce code, text, and designs in seconds, the scarce resource is no longer the artifact—it's the intentional human signal. By making prose the commit, IntentLog elevates the actual effort—deliberation, trade-offs, and the "why"—to first-class status. Everything else can be derived, regenerated, or reinterpreted, but original human reasoning stays intact and visible.

- **AI systems need interpretable instruction histories**: As we build increasingly capable agents, understanding how their guiding instructions evolved becomes essential for safety, alignment, and debugging.
- **Distributed teams need shared context**: Remote and asynchronous work amplifies the cost of lost rationale—IntentLog makes reasoning a durable team asset.
- **We're entering an era of human-AI co-reasoning**: Tools that preserve narrative intent will separate robust collaborative systems from brittle ones.

Every commit is a breadcrumb of cognition. Every branch is an explored possibility. Every merge is a narrated reconciliation. As AI handles more of the "what," we need better tools for preserving and compounding the human "why."

IntentLog isn't about replacing Git—it's about augmenting it with the missing layer: legible, evolvable human reasoning. It's a preservation system for collective intelligence. Start small, commit your why, and watch your project's collective intelligence compound over time.

Open for collaboration. Prototype early 2026. Prior art timestamped December 16, 2025.
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
You are free to share and adapt, provided you give appropriate credit and share alike.
