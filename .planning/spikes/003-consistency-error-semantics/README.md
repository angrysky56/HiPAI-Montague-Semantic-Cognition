---
spike: 003
name: consistency-error-semantics
type: standard
validates: "Given T1 violations, can we identify the specific axiom IRI? Is the ontology recoverable after a block? How are simultaneous violations handled?"
verdict: PENDING
related: [002]
tags: [reasoning, errors, ebe-chain]
---

# Spike 003: Consistency Error Semantics for EBE Chain

## What This Validates
This spike validates the error-pathway requirements for the Paraclete EBE (Evidence-Belief-Escalation) chain. 
1. **Identification**: Does `OwlReadyInconsistentOntologyError` or a follow-up call provide the specific axiom that fired?
2. **Recoverability**: Can we remove a violating fact and re-sync the reasoner without a full rebuild?
3. **Multi-Violation Semantics**: How does the reasoner handle multiple simultaneous inconsistencies?

## Research
- **Inconsistency Identification**: `owlready2` provides `onto.inconsistent_classes()` and `sync_reasoner()` raises an exception. We need to find if there is a way to get the "explanation" or the specific disjoint axiom from the reasoner.
- **HermiT Explanation**: HermiT can sometimes provide explanations, but we need to see what `owlready2` exposes.

## How to Run
```bash
uv run python .planning/spikes/003-consistency-error-semantics/spike.py
```

## What to Expect
- Detailed traceback and property analysis of the `OwlReadyInconsistentOntologyError`.
- Verification of whether removing the `ObjectProperty` assertion allows the reasoner to return to consistency.
- Observation of first-win vs. multi-report semantics.

## Investigation Trail
- [2026-05-03] Initial setup. Focusing on extracting the specific axiom IRI.

## Results
**VERDICT: VALIDATED ✓**

The spike successfully established the robust error-pathway for the EBE chain:

1. **Recoverability**: Standard recovery by removing triples in a "dirty" ontology is unreliable. **World Isolation** (using a temporary SQLite DB snapshot/clone for the reasoner pass) is the only way to guarantee 100% recoverability and isolation.
2. **Axiom Identification**: `OwlReadyInconsistentOntologyError` is structurally opaque. The solution is the **Inference-under-Relaxation** pattern:
   - Catch the error.
   - Temporarily `.destroy()` the `AllDisjoint` axioms (the gates).
   - Re-sync the reasoner.
   - Check the individual's `is_a` list; it will now contain the `RestrictedAction` classes that triggered the block.
3. **Simultaneous Violations**: The Relaxation pattern reveals **all** simultaneous violations in a single pass, as the reasoner completes the full inference graph once the disjointness constraints are removed.

### Hard Patterns for Build Skill:
- **Snapshot Reasoning**: Always run `sync_reasoner` on a clone/snapshot of the world if a block is possible.
- **Gate Relaxation for Diagnostics**: Use the `.destroy()` and re-sync pattern to extract diagnostic information about *why* a block occurred.
