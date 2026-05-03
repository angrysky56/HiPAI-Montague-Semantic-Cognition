# Roadmap (v0.5-refactor)

## Phase 1: Foundational Substrate
**Goal:** Establish the spaCy and owlready2 environment and seed the authoritative ontology.

- [ ] PLAN-1: Update dependencies and verify spaCy/en_core_web_md load
- [ ] PLAN-2: Initialize authoritative world.db (SQLite) and seed T1 axioms
- [ ] PLAN-3: Implement base OntologyManager for class/property creation

## Phase 2: Layer 2 Refactor & API Mirroring
**Goal:** Transition the semantic engine to the new NLP/OWL pipeline while maintaining MCP compatibility.

- [ ] PLAN-1: Implement spaCy-based dependency parsing for claim extraction
- [ ] PLAN-2: Refactor world_model to use owlready2 as the primary store
- [ ] PLAN-3: Mirror legacy API signatures (add_belief, etc.) and verify regression tests

## Phase 3: The EBE Pipeline
**Goal:** Implement transaction-per-claim semantics and robust inconsistency diagnostics.

- [ ] PLAN-1: Implement IsolationWorld (snapshot reasoning) pattern
- [ ] PLAN-2: Implement Relaxed-Sync Diagnostics (gate relaxation) pattern
- [ ] PLAN-3: Integrate EBE signal with Paraclete Manager

## Phase 4: The Projection Layer
**Goal:** Establish Neo4j as a read-model for retrieval and embeddings.

- [ ] PLAN-1: Implement OWL -> Neo4j Projection engine
- [ ] PLAN-2: Implement transitive type closure walk (.ancestors)
- [ ] PLAN-3: Verify projection idempotency and Cypher query results

## Phase 5: Integration & Calibration
**Goal:** Finalize the end-to-end SEG Council loop and perform the calibration run.

- [ ] PLAN-1: Update Council routing for orthogonal L2/L3 regimes
- [ ] PLAN-2: Execute 50-claim Calibration Suite (council-generated)
- [ ] PLAN-3: Final UAT and closure_status verification
