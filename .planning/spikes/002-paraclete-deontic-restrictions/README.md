---
spike: 002
name: paraclete-deontic-restrictions
type: standard
validates: "Given 'harms' restrictions, when 'Socrates harms Pig', then Socrates is inferred as 'MoralAgent'. AND when a forbidden action is asserted, then sync_reasoner() raises OwlReadyInconsistentOntologyError."
verdict: PENDING
related: [001]
tags: [nlp, reasoning, paraclete, owl]
---

# Spike 002: Paraclete Deontic Restrictions via OWL

## What This Validates
This spike validates the v0.4 architectural keystone: that Paraclete's T1 axioms (safety gates) can be expressed as OWL `EquivalentTo` restrictions and `DisjointWith` axioms, allowing the HermiT reasoner to enforce safety structurally.

Specifically:
1. **Positive Inference**: Does `ObjectProperty` domain/range and `some` restrictions correctly infer new classes for agents (e.g., `MoralAgent`)?
2. **Negative Inference (Block)**: Does asserting a forbidden action (violating disjointness) correctly raise `OwlReadyInconsistentOntologyError`?

## Research
- **EquivalentTo**: Used to define `RestrictedAction` as equivalent to `Action & harms some MoralPatient`.
- **DisjointWith**: Used to mark `RestrictedAction` and `PermittedAction` as mutually exclusive.
- **HermiT**: Required to compute these inferences and consistency.

## How to Run
```bash
uv run python .planning/spikes/002-paraclete-deontic-restrictions/spike.py
```

## What to Expect
- **Inference**: Socrates is inferred to be a `MoralAgent` because he `harms` a `Pig` (which is a `MoralPatient`).
- **Inconsistency**: Asserting an action is both `RestrictedAction` and `PermittedAction` should cause the reasoner to fail with a consistency error.

## Investigation Trail
- [2026-05-03] Initial setup. Researching `equivalent_to` vs `is_a` for deontic gates.
- [2026-05-03] Collapsed 003 into 002 to test the full EBE (Evidence-Belief-Escalation) chain semantics in one go.

## Results
**VERDICT: VALIDATED ✓**

The spike successfully confirmed the v0.4 architectural keystone:
1. **Positive Inference**: HermiT correctly inferred that `Socrates` is a `MoralAgent` based on the `domain` restriction of the `harms` property when he was asserted to harm a `MoralPatient`.
2. **Negative Inference (Block)**: Defining a `RestrictedAction` using an `equivalent_to` restriction (`Action & harms some MoralPatient`) and marking it as `DisjointWith(PermittedAction)` created a functional safety gate.
3. **Consistency Error**: When an action was asserted that met the definition of a `RestrictedAction` but was also a `PermittedAction`, `sync_reasoner()` raised `OwlReadyInconsistentOntologyError`.

This proves that **T1 axioms can be enforced structurally by the reasoner** without manual Cypher checks.

### Surprises/Gotchas:
- **`AllDisjoint`**: Essential for the Paraclete gate. Without explicit disjointness, the reasoner would simply infer that the action is *both*, rather than flagging an inconsistency.
- **Inference Chain**: The inference `Socrates -> MoralAgent` happened automatically during the first `sync_reasoner()` call, validating that Layer 2 can handle relational claims effectively.
