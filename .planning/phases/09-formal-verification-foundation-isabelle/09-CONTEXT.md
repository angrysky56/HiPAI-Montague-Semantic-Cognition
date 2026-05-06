# Phase 09: Formal Verification Foundation (Isabelle) - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Source:** PRD Express Path (/home/ty/Documents/LLM-WIKI/raw/Isabelle Installation.md) and Design Note (docs/logic/ISABELLE_INTEGRATION.md)

<domain>
## Phase Boundary

This phase establishes the Tier 0 (T0) verification layer for HiPAI. It involves installing the Isabelle theorem prover, landing the `Paraclete_Foundation.thy` theory, and integrating a machine-checked consistency proof for the seed axioms into the project workflow.

**Deliverables:**
- Isabelle 2025-2 installed and available in the environment.
- `Paraclete_Foundation.thy` fully implemented and verified.
- Build automation (Makefile or script) to run `isabelle build -D docs/logic`.

</domain>

<decisions>
## Implementation Decisions

### Installation (from PRD)
- Use **Isabelle 2025-2** for Linux.
- Installation path: Local to the repository or standard user location (e.g., `~/Isabelle2025-2`).
- Command-line tool `isabelle` must be available to the agent.

### Verification Foundation (from Design Note)
- **Model:** Entities, agents, patients, and action triples `(s, r, o)` as datatypes.
- **Axioms:** `AllDisjoint(Action, Agent, Patient)` enforcement proofs.
- **Meta-theorems to prove:**
  - **Consistency:** Seed axioms have at least one model.
  - **Gate Soundness:** `BLOCKED ⟹ ∀ extension W. ¬ permitted_in W (s,r,o)`.
  - **Monotonicity:** Adding beliefs cannot move an action from BLOCKED to PERMITTED.

### the agent's Discretion
- Specific script implementation for triggering the build.
- Integration with local CI/pre-commit hooks.
- Exact directory structure for Isabelle session logs if needed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Logic & Design
- `docs/logic/ISABELLE_INTEGRATION.md` — Core design for Isabelle integration and phases.
- `docs/logic/Paraclete_Foundation.thy` — Starter theory file for Phase 1.

### Setup
- `/home/ty/Documents/LLM-WIKI/raw/Isabelle Installation.md` — Detailed installation guide for Isabelle 2025-2.

</canonical_refs>

<specifics>
## Specific Ideas
- The build command is `isabelle build -D docs/logic`.
- If `seed_axioms()` in `src/hipai/ontology_manager.py` changes, the proof must be re-run.

</specifics>

<deferred>
## Deferred Ideas
- Phase 2: Mechanizing the EBE Theorem (`EBE_Theorem.thy`).
- Phase 3: Montague-in-Pure (typed lambda calculus object logic).

</deferred>

---

*Phase: 09-formal-verification-foundation-isabelle*
*Context gathered: 2026-05-06 via PRD Express Path*
