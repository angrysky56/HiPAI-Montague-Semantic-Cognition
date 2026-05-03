# HiPAI-Montague Design Specification v0.5

## Status: PROMOTED (Post-Spike Campaign)

This document outlines the v0.5 architecture for the HiPAI-Montague Semantic Cognition engine, incorporating validated findings from Spikes 001–004.

---

## 1. Architectural Core: The Four Pillars

### Pillar 1: Semantic Synthesis (L1 → L2)
- **Engine**: spaCy (en_core_web_md).
- **Function**: Replaces legacy regex-based pattern matching with structural dependency parsing.
- **Invariants**: 
  - Verbs are lemmatized to their base form.
  - Plurality and casing are normalized at the NLP layer.
  - Claims are mapped to OWL individuals and properties.

### Pillar 2: Authority via OWL (L2)
- **Engine**: `owlready2` + HermiT (Java Reasoner).
- **Function**: Acts as the authoritative world model for reasoning, consistency checking, and safety.
- **Safety Gates (Paraclete)**: 
  - T1 axioms are expressed as OWL `equivalent_to` restrictions.
  - **Mandatory Disjointness**: Every restricted class must be explicitly `DisjointWith(PermittedAction)`.
  - **Relational Typing**: Uses domain/range restrictions to automatically infer types (e.g., `Socrates harms X` -> `Socrates: MoralAgent`).

### Pillar 3: The EBE Pipeline (Error Semantics)
- **Engine**: Transaction-per-claim Reasoner pass.
- **Snapshot Reasoning Pattern**:
  - Every reasoning pass is performed on a World Isolation snapshot (cloned SQLite DB).
  - Ensures a "dirty" ontology after inconsistency does not pollute the main world.
- **Inference-under-Relaxation Pattern**:
  - To identify *why* a block occurred, the engine temporarily destroys the `AllDisjoint` gate axioms and re-syncs.
  - This reveals the full inference graph and identifies multiple simultaneous violations.

### Pillar 4: Neo4j Projection (Read-Model)
- **Engine**: One-way Downstream Projector.
- **Function**: Projects successful inferences to Neo4j for high-performance Cypher-based retrieval and embedding search.
- **Invariants**: 
  - Sync direction is strictly **OWL → Neo4j**.
  - Sync only occurs after a **successful** (consistent) reasoner pass.
  - **Transitive Projection**: Walking `.ancestors()` ensures the full type hierarchy is queryable in Neo4j.

---

## 2. Component Layout

| Layer | Responsibility | Substrate |
|---|---|---|
| **L1: Sensory** | Claim Extraction | spaCy |
| **L2: Semantic** | Reasoning / Safety | owlready2 + HermiT |
| **L3: Logical** | FOL / Categorical | mcp-logic |
| **L4: Cognitive** | Synthesis / Memory | project-synapse |
| **L5: Executive** | EBE Chain / Gatekeeping | Paraclete Manager |

---

## 3. Key Design Patterns

### Snapshot Reasoning
```python
# Create Isolation for the Transaction
scratch_world = World(filename=temp_db_snapshot)
sync_reasoner(scratch_world) # Rollback is implicitly scratch_world.close()
```

### Relaxed-Sync Diagnostics
```python
try:
    sync_reasoner(isolation_world)
except OwlReadyInconsistentOntologyError:
    # Break the gates
    gate_axiom.destroy()
    sync_reasoner(isolation_world)
    # Read the culprits from the inferred types
    diagnose(individual.is_a)
```

---
*Index at 116 pages (Wiki Mirror)*
