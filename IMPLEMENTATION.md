# Implementation Plan: HiPAI-Montague Refactor (v0.5)

## Objective: Transition to Authoritative OWL reasoning with Neo4j projection.

---

## Phase A: Foundational Substrate (Day 1 Morning)
1. [ ] **Environment**: Update `pyproject.toml` with `spacy` and `owlready2` dependencies.
2. [ ] **NLP Init**: Download `en_core_web_md` and verify spaCy load in `src/nlp_engine.py`.
3. [ ] **OWL Init**: Setup base `world.db` (sqlite) and seed initial T1 axioms (Action, Entity, HARM, etc.).

## Phase B: Layer 2 Refactor (Day 1 Afternoon)
4. [ ] **Internal Translation**: Replace regex-based `claim_synthesis` with spaCy dependency parsing logic.
5. [ ] **Authority Shift**: Refactor `world_model` to write to `owlready2` by default.
6. [ ] **API Mirroring**: 
    - Maintain existing signatures (`add_belief`, `evaluate_hypothesis`) but wrap the new OWL backend.
    - Keep `register_agent_state` active for baseline closure_status monitoring.
7. [ ] **Unit Tests**: Verify the 21-test regression suite passes with the new backend.

## Phase C: The EBE Pipeline (Day 2 Morning)
8. [ ] **Snapshot Logic**: Implement `IsolationWorld` context manager for transaction-per-claim semantics.
9. [ ] **Relaxation Logic**: Implement the diagnostic path for identifying `blocking_axiom` via disjoint-destroy pattern.
10. [ ] **Error Mapping**: Map `OwlReadyInconsistentOntologyError` to the Paraclete `EBE_Chain` signal.

## Phase D: The Projection Layer (Day 2 Afternoon)
11. [ ] **Projector**: Implement the OWL → Neo4j projection function (~50 lines).
12. [ ] **Hierarchy Support**: Ensure `.ancestors()` walk is included for transitive type signal.
13. [ ] **Idempotency**: Use `MERGE` statements and add a test to verify no duplicate edges on repeated passes.

## Phase E: Integration & Calibration (Day 3)
14. [ ] **L1 Integration**: Update Council routing to use the new L2/L3 orthogonal boundaries.
15. [ ] **UAT**: Run the 50-claim calibration suite (council-generated vs solo).
16. [ ] **Final Review**: Assert `closure_status` improvement (WEAK -> STRONG).

---

## Critical Invariants (Mandatory)
- **AllDisjoint**: Never omit disjointness axioms for Paraclete gates.
- **One-Way Sync**: Neo4j is ONLY updated after a successful `sync_reasoner()` pass.
- **Transaction Rollback**: Failures in `IsolationWorld` must not persist to the main `world.db`.

## Note on Compatibility
This plan follows the **Conservative Path**, preserving existing MCP signatures for mid-refactor regression testing. Total estimated duration: 3 days.
