# Roadmap (v0.6-advanced-cognition)

## Phase 1: Foundational Substrate (v0.5)
**Goal:** Establish the spaCy and owlready2 environment and seed the authoritative ontology.

- [x] PLAN-1: Update dependencies and verify spaCy/en_core_web_md load
- [x] PLAN-2: Initialize authoritative world.db (SQLite) and seed T1 axioms
- [x] PLAN-3: Implement base OntologyManager for class/property creation

## Phase 2: Layer 2 Refactor & API Mirroring (v0.5)
**Goal:** Transition the semantic engine to the new NLP/OWL pipeline while maintaining MCP compatibility.

- [x] PLAN-1: Implement spaCy-based dependency parsing for claim extraction
- [x] PLAN-2: Refactor world_model to use owlready2 as the primary store
- [x] PLAN-3: Mirror legacy API signatures (add_belief, etc.) and verify regression tests

## Phase 3: The EBE Pipeline (v0.5)
**Goal:** Implement transaction-per-claim semantics and robust inconsistency diagnostics.

- [x] PLAN-1: Implement IsolationWorld (snapshot reasoning) pattern
- [x] PLAN-2: Implement Relaxed-Sync Diagnostics (gate relaxation) pattern
- [x] PLAN-3: Integrate EBE signal with Paraclete Manager

## Phase 4: The Projection Layer (v0.5)
**Goal:** Establish Neo4j/FalkorDB as a read-model for retrieval and embeddings.

- [x] PLAN-1: Implement OWL -> FalkorDB Projection engine
- [x] PLAN-2: Implement transitive type closure walk (.ancestors)
- [x] PLAN-3: Verify projection idempotency and Cypher query results

## Phase 5: Integration & Calibration (v0.5)
**Goal:** Finalize the end-to-end SEG Council loop and perform the calibration run.

- [x] PLAN-1: Update Council routing for orthogonal L2/L3 regimes
- [x] PLAN-2: Execute 50-claim Calibration Suite (council-generated)
- [x] PLAN-3: Final UAT and closure_status verification

## Phase 6: Advanced Cognition (v0.6)
**Goal:** Implement recursive attitudes and hardened entity disambiguation.

- [x] PLAN-1: Recursive Claim Extraction (beliefs about beliefs)
- [x] PLAN-2: Graph-Driven Ambiguity Resolution (semantic similarity)
- [x] PLAN-3: Compound Name & Multi-Token Entity Handling
- [x] PLAN-4: Cognitive Depth Throttling (Recursion Depth Control)

## Phase 7: Executive Control (v0.7)
**Goal:** Implement higher-order reasoning over conflicting belief sets.

- [ ] PLAN-1: Multi-Perspective Synthesis (Braided Beliefs)
- [ ] PLAN-2: Automated Counter-Hypothesis Generation
- [ ] PLAN-3: Logical Conflict Resolution Strategies
