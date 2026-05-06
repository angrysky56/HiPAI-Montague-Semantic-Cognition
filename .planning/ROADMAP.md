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


### Phase 8: Synthesis Calibration & Polarity (v0.7)

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 7
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd-plan-phase 8 to break down)

### Phase 9: Formal Verification Foundation (Isabelle)

**Goal:** Integrate Isabelle 2025-2 and verify the Paraclete Protocol foundation.
**Requirements:** isabelle-2025-2
**Depends on:** Phase 1-6
**Plans:** 3 plans

Plans:
- [x] 09-01: Isabelle 2025-2 environment installation and session configuration
- [x] 09-02: Paraclete_Foundation.thy meta-theorems and Makefile integration
- [x] 09-03: MCP tool integration for machine-checked logic verification

### Phase 10: Framework Reframing & Belief Dynamics (v0.7)

**Goal:** Lift HiPAI from "the Paraclete implementation" to "a framework for declarative structural-ethics reasoning, with the Paraclete as one configuration." Includes the audit-driven correction of the keyword-coupled gate (Phase A) and the parametric formalization of subsumption-based gating + paraconsistent escalation (Phase B).

**Requirements:** isabelle-2025-2, sentence-transformers (or EmbeddingGemma), `Hierarchy_Soundness.thy`, `Belief_Dynamics.thy`

**Depends on:** Phase 9

**Background:** `docs/logic/RESOLUTION_AUDIT.md` exposed that the v0.5–v0.6 gate was a literal-string check on `Concept_Patient`, with the OWL hierarchy beneath it essentially empty. The Isabelle proof from Phase 9 was structurally sound but proved theorems about an abstract model the implementation didn't realize. Phase 10 closes that gap.

#### Phase A — Framework reframing (LANDED, 2026-05-06)

Plans:
- [x] 10A-01: Real protected hierarchy seeded under `Concept_Patient`
- [x] 10A-02: `_resolve_or_create_class` with embedding-anchored parent selection
- [x] 10A-03: `WorldModel._classify_class_term` (sentence-transformer cosine match)
- [x] 10A-04: `calibrate_belief` polarity-aware source counting (audit §5.1 fix)
- [x] 10A-05: Smoke test `tests/test_framework_reframing.py` covering 5 probe groups
- [x] 10A-06: Idempotent reseed on every init (free migration for existing DBs)

See `docs/logic/PHASE_A_NOTES.md` for the change set and validation procedure.

#### Phase B — MCP primitives + parametric Isabelle (LANDED, 2026-05-06)

Plans:
- [x] 10B-01: MCP tools: `declare_class_hierarchy`, `set_default_unclassified`, `list_protected_closure`
- [x] 10B-02: Move Paraclete-specific subhierarchy into a swappable config (`paraclete_config.py`); `seed_axioms()` retains only framework classes
- [x] 10B-03: `Hierarchy_Soundness.thy` — parametric theorem `∀ H. well_formed(H) ⟹ gate_correct_wrt(H)`. Phase 1's theorem becomes a corollary
- [x] 10B-04: `Belief_Dynamics.thy` — Belnap-4 paraconsistent state + source-count + conservative-default. Meta-theorem: escalation pipeline is monotone-toward-FINAL_BLOCK relative to the gate (per audit §4)
- [x] 10B-05: `evaluate_hypothesis` returns `contradicted` flag when both `p` and `¬p` hold (audit §5.4 fix)
- [x] 10B-06: EmbeddingGemma swap (308M, 100+ languages, MRL-truncatable). Mechanical change; retune MID/HIGH thresholds afterwards.
