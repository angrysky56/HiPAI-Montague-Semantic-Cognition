# Implementation Plan: HiPAI-Montague Refactor (v0.5)

## Objective: Transition to Authoritative OWL reasoning with Neo4j projection.

---

## Phase A: Foundational Substrate (Day 1 Morning)
1. [x] **Environment**: Update `pyproject.toml` with `spacy` and `owlready2` dependencies.
2. [x] **NLP Init**: Download `en_core_web_md` and verify spaCy load.
3. [x] **OWL Init**: Setup base `world.db` (sqlite) and seed initial T1 axioms.

## Phase B: Layer 2 Refactor (Day 1 Afternoon)
4. [x] **Internal Translation**: Replace regex-based `claim_synthesis` with spaCy dependency parsing logic.
5. [x] **Authority Shift**: Refactor `world_model` to write to `owlready2` by default.
6. [x] **API Mirroring**: 
    - Maintain existing signatures (`add_belief`, `evaluate_hypothesis`) but wrap the new OWL backend.
7. [x] **Unit Tests**: Verify the 21-test regression suite passes with the new backend.

## Phase C: The EBE Pipeline (Day 2 Morning)
8. [x] **Snapshot Logic**: Implement `IsolationWorld` context manager for transaction-per-claim semantics.
9. [x] **Relaxation Logic**: Implement the diagnostic path for identifying `blocking_axiom` via disjoint-destroy pattern.
10. [x] **Error Mapping**: Map `OwlReadyInconsistentOntologyError` to the Paraclete `EBE_Chain` signal.

## Phase D: The Projection Layer (Day 2 Afternoon)
11. [x] **Projector**: Implement the OWL → FalkorDB (Neo4j-compatible) projection function.
12. [x] **Hierarchy Support**: Ensure `.ancestors()` walk is included for transitive type signal.
13. [x] **Idempotency**: Use `MERGE` statements and verify no duplicate edges.

## Phase E: Integration & Calibration (Day 3)
14. [x] **L1 Integration**: Update Council routing to use the new L2/L3 orthogonal boundaries.
15. [x] **UAT**: Run the 50-claim calibration suite.
16. [x] **Final Review**: Assert `closure_status` improvement (WEAK -> STRONG).

## Phase F: Advanced Cognition (Recursive Logic & Ambiguity)
17. [x] **Recursive Parsing**: Extract nested observations for attitude verbs (`ccomp` support).
18. [x] **Graph Recursion**: Link Entities to EpistemicNodes in the projection layer.
19. [x] **Entity Resolution**: Implement semantic-search disambiguation in the belief pipeline.
20. [x] **Confidence Propagation**: Propagate resolution quality from individuals to observations.

---

## Critical Invariants (Mandatory)
- **AllDisjoint**: Never omit disjointness axioms for Paraclete gates.
- **One-Way Sync**: Neo4j is ONLY updated after a successful `sync_reasoner()` pass.
- **Transaction Rollback**: Failures in `IsolationWorld` must not persist to the main `world.db`.

## Note on Compatibility
This plan follows the **Conservative Path**, preserving existing MCP signatures for mid-refactor regression testing. Total estimated duration: 3 days.
