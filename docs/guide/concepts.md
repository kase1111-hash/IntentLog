# Core Concepts

## The Doctrine of Intent

IntentLog is built on the Doctrine of Intent, which establishes principles for capturing and preserving human reasoning.

### Intent as First-Class Artifact

Traditional version control tracks *what* changed. IntentLog tracks *why* things changed.

```
Git:       "Modified auth.py: added JWT validation"
IntentLog: "Implementing JWT validation to prevent token replay attacks
            after security audit revealed session hijacking vulnerability"
```

### Key Metrics

#### Intent Density (Di)

The resolution at which reasoning is captured:

- **High Di**: Every decision documented with context
- **Low Di**: Sparse documentation, missing reasoning

Formula: `Di = R × C × Co`

Where:
- **R (Resolution)**: Granularity of intent capture
- **C (Continuity)**: Consistency across time
- **Co (Coverage)**: Percentage of decisions documented

#### Information Density

The richness of captured context:

- Word count and vocabulary diversity
- Metadata completeness
- Cross-references to external resources

#### Auditability

The ability to trace decisions:

- Cryptographic chain linking
- Timestamp integrity
- Signature verification

#### Fraud Resistance

Guarantees of authenticity:

- Merkle tree integrity
- Ed25519 signatures
- Append-only ledger

## Prose Commits

Unlike code commits, IntentLog commits are prose-first:

```bash
# Good: Explains the why
ilog commit "Adding rate limiting to prevent API abuse.
Customer support reported increased timeout complaints.
Chose token bucket algorithm for smooth traffic shaping.
Considered leaky bucket but needed burst handling."

# Bad: Just describes what
ilog commit "Added rate limiter"
```

## Chain Integrity

Each intent links to the previous via cryptographic hash:

```
Intent 1 ─────────────────┐
    │                     │
    └──hash──> Intent 2 ──┤
                   │      │
                   └──hash──> Intent 3
```

This creates a tamper-evident chain where any modification breaks verification.

## Branches

Like Git, IntentLog supports branches for parallel reasoning tracks:

```
main:     Design ──> Implement ──> Review
              │
feature:      └──> Experiment ──> Iterate ──> (merge)
```

## MP-02 Protocol

The MP-02 protocol extends IntentLog with effort receipts:

1. **Signals**: Raw observations (keystrokes, file changes)
2. **Segments**: Time-grouped effort periods
3. **Receipts**: Cryptographic proof of effort
4. **Ledger**: Immutable record of all receipts

See [MP-02 Protocol](mp02.md) for details.

## Privacy Model

IntentLog respects privacy via MP-02 Section 12:

- **Encryption**: Sensitive content can be encrypted
- **Access Control**: Granular permissions per intent
- **Revocation**: Future observation can be blocked
- **Immutability**: Past receipts remain unchanged

## LLM Integration

Optional LLM features enhance IntentLog:

- **Semantic Search**: Find intents by meaning, not keywords
- **Semantic Diff**: Understand conceptual changes between branches
- **Formalization**: Generate code/rules from prose intent

## Next Steps

- [CLI Reference](cli.md)
- [MP-02 Protocol](mp02.md)
- [API Reference](../api/index.md)
