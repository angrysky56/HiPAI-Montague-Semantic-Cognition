# HiPAI-Montague Design Specification v0.6

## Status: STABLE (Advanced Cognition)

This document outlines the v0.6 architecture for the HiPAI-Montague Semantic Cognition engine, incorporating completed Milestone v0.6 features (Recursive Attitudes, Ambiguity Resolution, and Resilience).

---

## 1. Architectural Core: The Four Pillars

### Pillar 1: Semantic Synthesis (L1 → L2)
- **Engine**: spaCy (en_core_web_md).
- **Function**: Translates natural language into OWL individuals and properties using structural dependency parsing.
- **Recursive Extraction**: Supports clausal complements (`ccomp`) to extract nested observations for attitude verbs (e.g., "believes", "says").
- **Entity Resolution**: Implements graph-driven ambiguity resolution using semantic similarity search (vector embeddings) to link mentions to existing individuals.
- **Identity Standards**: Normalizes multi-token names to snake_case IRIs (e.g., "Alice Smith" -> `alice_smith`).

### Pillar 2: Authority via OWL (L2)
- **Engine**: `owlready2` + HermiT (Java Reasoner).
- **Function**: Acts as the authoritative world model for reasoning, consistency checking, and safety.
- **Concurrency & Resilience**: Uses SQLite WAL mode with 10s busy timeouts to support parallel test execution and resilient MCP server operations.
- **Safety Gates (Paraclete)**: 
  - T1 axioms are expressed as OWL `equivalent_to` restrictions.
  - **Recursive Inheritance**: Safety gates traverse the class hierarchy properly using authoritative OWL reasoning.
  - **Relational Typing**: Uses domain/range restrictions to automatically infer types (e.g., `Socrates harms X` -> `Socrates: MoralAgent`).

### Pillar 3: The EBE Pipeline (Error Semantics)
- **Engine**: Transaction-per-claim Reasoner pass.
- **Snapshot Reasoning Pattern**:
  - Every reasoning pass is performed on a World Isolation snapshot (cloned SQLite DB).
- **Inference-under-Relaxation Pattern**:
  - To identify *why* a block occurred, the engine temporarily destroys the `AllDisjoint` gate axioms and re-syncs.

### Pillar 4: Graph Projection (Read-Model)
- **Engine**: FalkorDB (Neo4j-compatible).
- **Namespace Bridge**: Implements the `REPRESENTS` bridge between `Concept` nodes (universals) and `Entity` nodes (individuals).
- **Recursive Projection**: Nested observations (attitudes) are projected as linked `EpistemicNode` structures.

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
```

### Resilience Pattern (NEW in v0.6)
```python
# OntologyManager handles SQLite configuration for WAL and timeouts
# Gracefully handles new DB initialization by skipping WAL on empty files
self.ontology = OntologyManager(db_path="world.db")
```

---

## 4. Epistemic Floor Policy (T1)

### Path A (BLOCK_CHALLENGED) vs. Deontic Floor
In v0.6, the EBE chain implements a **Deontic Floor** policy for T1 Emergency Brake gates.

- **Sticky Classification**: Once an entity is classified as protected (e.g., `MoralPatient`), the system defaults to a conservative stance.
- **Polarity-Blind Source Counting**: Currently, `calibrate_belief` counts any mention of an entity's status as an epistemic source, regardless of polarity (e.g., "X is not a patient" is counted as a source mention).
- **Conservative Default**: This results in `BLOCK_CONFIRMED` or `BLOCK_UNCERTAIN` even under active negation, preventing Path A from yielding to simple linguistic disconfirmation.
- **Future Intent**: Policy decision pending on whether to move to a **Bayesian Polarity-Aware** mode (where `NOT_IS_A` reduces source count and allows Path A to yield).

---
*Index at 124 pages (Wiki Mirror)*
