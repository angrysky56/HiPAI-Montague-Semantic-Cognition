# HiPAI-Montague Design Specification v0.5

## Status: PROMOTED (Post-Spike Campaign)

This document outlines the v0.5 architecture for the HiPAI-Montague Semantic Cognition engine, incorporating validated findings from Spikes 001–004.

---

## 1. Architectural Core: The Four Pillars

### Pillar 1: Semantic Synthesis (L1 → L2)
- **Engine**: spaCy (en_core_web_md).
- **Function**: Translates natural language into OWL individuals and properties using structural dependency parsing.
- **Recursive Extraction**: Supports clausal complements (`ccomp`) to extract nested observations for attitude verbs (e.g., "believes", "says").
- **Entity Resolution**: Implements graph-driven ambiguity resolution using semantic similarity search to link mentions to existing individuals.

### Pillar 2: Authority via OWL (L2)
- **Engine**: `owlready2` + HermiT (Java Reasoner).
- **Function**: Acts as the authoritative world model for reasoning, consistency checking, and safety.
- **Safety Gates (Paraclete)**: 
  - T1 axioms are expressed as OWL `equivalent_to` restrictions.
  - **Dynamic Axiom Injection**: Supports loading custom T1 axioms from the graph read-model into the OWL world for runtime constraint updates.
  - **Recursive Inheritance**: Safety gates traverse the class hierarchy properly using authoritative OWL reasoning.
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

### Pillar 4: Graph Projection (Read-Model)
- **Engine**: FalkorDB (Neo4j-compatible).
- **Namespace Bridge**: Implements the `REPRESENTS` bridge between `Concept` nodes (universals) and `Entity` nodes (individuals) to enable transitive reasoning across both namespaces.
- **Function**: Projects successful inferences to a vector-capable graph database for high-performance Cypher-based retrieval and semantic search.
- **Invariants**: 
  - Sync direction is strictly **OWL → Graph**.
  - Sync only occurs after a **successful** (consistent) reasoner pass.
  - **Recursive Projection**: Nested observations (attitudes) are projected as linked `EpistemicNode` structures.

### Pillar 5: Recursive Attitudes (Higher-Order Cognition)
- **Architecture**: Supports n-order beliefs (e.g., "Alice believes that Bob thinks that Charlie is happy").
- **Graph Mapping**: Attitude relations link `ContentNode:Entity` to `EpistemicNode:Observation` in the graph layer.
- **Ontology Mapping**: Nested individuals are recursively added to the authoritative OWL ontology.

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
# Create Isolation for the Transaction via HIPAIManager fork
isolated_mgr = main_mgr.fork(session_id="exploration_42")
isolated_mgr.add_belief("Exploratory claim...") 
# Rollback is implicit: isolated_mgr.clear_database() or simply not merging
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
