---
spike: 004
name: neo4j-projection
type: integration
validates: "Given successful reasoning, can we project inferred class memberships and provenance to Neo4j as a read-model? Is the projection idempotent and queryable via Cypher?"
verdict: PENDING
related: [002, 003]
tags: [integration, neo4j, projection]
---

# Spike 004: Neo4j Projection as a Read-Model

## What This Validates
This spike validates the "Projection" pattern where Neo4j acts as a read-model for retrieval and embeddings, while the OWL ontology (via `owlready2`) remains the source of truth for reasoning and safety.

Specifically:
1. **Membership Projection**: Inferred class memberships (e.g., `Socrates -> MoralAgent`) are projected to Neo4j.
2. **Idempotency**: The projection can be run multiple times without duplicating data.
3. **Queryability**: Inferred types are available to Cypher-side retrieval logic.
4. **Provenance (Bonus)**: Can we project the "justification" for an inference?

## Research
- **Sync Direction**: One-way (OWL -> Neo4j).
- **Transaction Boundary**: Snapshot Reasoning pattern (from 003) ensures only consistent states are projected.
- **Justifications**: `owlready2` + Pellet supports `explain()`, but HermiT's support is limited. We will probe for provenance early in the spike.

## How to Run
```bash
uv run python .planning/spikes/004-neo4j-projection/spike.py
```

## What to Expect
- Neo4j will reflect inferences made by the reasoner.
- Cypher queries for inferred classes will return the correct results.
- A determination on the feasibility of provenance projection.

## Investigation Trail
- [2026-05-03] Initial setup. Re-scoped from "Neo4j as world model" to "Neo4j as projection".

## Results
**VERDICT: VALIDATED ✓**

The spike successfully validated the "Projection" pattern:
1. **Membership Projection**: By walking `cls.ancestors()` for each `is_a` class of an individual, we can project the full transitive closure of inferred types to Neo4j. 
2. **Relational Typing**: `Socrates` was correctly projected as a `MoralAgent` (inferred from domain restrictions) alongside his asserted and inherited types (`Man`, `Mortal`).
3. **Idempotency**: The implementation uses `MERGE` statements, ensuring that the projection step is stable across multiple reasoner passes.
4. **Provenance (The "Justification" Gap)**: A sanity check confirmed that `owlready2` + `HermiT` does not expose justifications (explanations) natively in a form that is easy to project. Provenance projection is deferred to future work (potential Spike 005 or direct implementation in the EBE chain logic).

### Hard Patterns for Build Skill:
- **Transitive Projection**: Use `for ancestor in cls.ancestors():` to ensure the read-model (Neo4j) has the full type signal.
- **Downstream Only**: The projection step must only run *after* a successful `sync_reasoner()` call to ensure only consistent ground truth is published.
