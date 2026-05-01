# HiPAI Montague Semantic Cognition

## What This Is

A semantic cognition engine implementing Montague semantics, backed by a world model in FalkorDB. It synthesizes natural language observations into formal logical representations, checks constraints, and clusters concepts to build a dynamic knowledge graph.

## Core Value

Robust, formal semantic parsing and logical reasoning over a dynamic, graph-based world model.

## Requirements

### Validated

- ✓ Semantic engine with Montague semantics (e, t, <e,t>) and lambda composition — existing
- ✓ Persistence layer mapping semantic entities to a FalkorDB graph — existing
- ✓ Synthesis engine for belief addition, hypothesis evaluation, and concept synthesis — existing
- ✓ Pydantic-based data ontology — existing

### Active

- [ ] Evolve and advance semantic parsing of natural language to formal observations.
- [ ] Harden parsing to handle edge cases and linguistic ambiguity gracefully.

### Out of Scope

- Performance optimization of KMeans and embeddings — deferred until scale issues manifest.

## Context

- **Current Architecture**: See `.planning/codebase/ARCHITECTURE.md`.
- **Known Issues**: Parsing can be brittle; edge cases in natural language currently break the parser or produce ambiguous observations.

## Constraints

- **Compatibility**: Must remain compatible with the existing FalkorDB schema and Pydantic models.
- **Reliability**: Semantic parsing outputs must be valid logical structures for the world model.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on Parsing Robustness | Parsing is the most brittle area currently, breaking down on edge cases. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-01 after initialization*
