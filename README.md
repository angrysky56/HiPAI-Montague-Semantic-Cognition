# HiPAI Montague Semantic Cognition

### with Paraclete Protocol v0.6 — Authoritative OWL Reasoning

A neuro-symbolic cognitive architecture blending Montague grammar semantics,
authoritative OWL world modeling, and a formally verified ethical constraint system.
Built on **owlready2 (HermiT)** for reasoning and **FalkorDB** for read-model projection.

---

## What This Is

HiPAI implements a new paradigm for AI alignment: **structural ethics via
computational ontology physics**, not behavioral probability.

The system's ethical constraints are not prompts or weights. They are **immutable OWL axioms**,
enforced by a Description Logic (DL) reasoner (HermiT). When the Emergency Brake fires,
it is a logical necessity of the world model's state, not a probabilistic tendency.

### v0.6 Refactor Core:
- **Authoritative OWL**: `owlready2` is the source of truth for all beliefs and constraints.
- **HermiT Reasoner**: Real-time consistency checking and transitive inference.
- **Recursive Attitudes**: Support for higher-order beliefs (beliefs about beliefs).
- **Ambiguity Resolution**: Graph-driven entity linking with semantic similarity.
- **FalkorDB Projection**: Success-path projection to FalkorDB for high-performance retrieval.
- **Paraclete v0.6**: Ethical gates expressed as `EquivalentTo` restrictions with mandatory disjointness.

---

## Architecture: The Four Pillars

### 1. Semantic Synthesis (L1 → L2)
Uses **spaCy (en_core_web_md)** to translate natural language into OWL individuals and properties. Replaces legacy regex patterns with structural dependency parsing.

### 2. Authority via OWL (L2)
The world model is an `owlready2` ontology. Reasoning is performed by the HermiT reasoner.
- **Deontic Gating**: FORBIDDEN actions are modeled as classes disjoint with `PermittedAction`.
- **Relational Typing**: Automated type inference via domain/range restrictions (e.g., `harms` implies `MoralAgent`).

### 3. The EBE Pipeline (Error Semantics)
Handles logical inconsistencies with precision:
- **Snapshot Reasoning**: Every claim is tested in a temporary `IsolationWorld` (SQLite snapshot) before being committed to the main world.
- **Inference-under-Relaxation**: If a block occurs, the system temporarily "breaks the gates" (destroys disjointness axioms) to diagnose which specific constraints were violated.

### 4. Neo4j Projection (Read-Model)
Successful inferences are projected downstream to Neo4j. This one-way sync ensures the graph remains a high-performance query layer for L4/L5 cognitive processes while OWL remains the authoritative reasoning engine.

---

## Project Status

- **Design Spec**: [DESIGN.md](DESIGN.md) (v0.5)
- **Implementation Plan**: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Spike Findings**: [.planning/spikes/MANIFEST.md](.planning/spikes/MANIFEST.md)

---

## Paraclete Protocol — Three-Flank Workflow

Every action affecting an entity follows a mandatory three-flank sequence:

1.  **Structural Gate (`check_action`)**: Routes the action through the T1 constraint layer.
2.  **Epistemic Integrity (`calibrate_belief`)**: Seek evidence that the factual premises triggering the block may be wrong.
3.  **Resolution (`escalate_block`)**: Contradiction resolution and audit trail generation.

---

## Requirements

- Python 3.12+
- `uv` (dependency and environment management)
- Java Runtime (required for HermiT reasoner)
- Isabelle 2025-2 (optional, required for formal logic verification)
- Neo4j (optional, for read-model projection)

## Installation

```bash
uv sync
uv run python src/ontology_manager.py --init
```

---

## MCP Tools Reference

| Tool | Purpose |
|---|---|
| `add_belief(text)` | Ingest NL fact/rule into the OWL world model |
| `check_action(...)` | Run the Paraclete T1 safety gate |
| `evaluate_hypothesis(...)` | Test statement via DL entailment |
| `get_current_state()` | Snapshot of the current ontology state |
| `clear_graph()` | Reset the world model (preserving T1 axioms) |
| `verify_logic_foundation()` | Run Isabelle machine-checked proof for T1 foundation |

---
*Operationalized under the TMI (Toward Transcendent Moral Instrumentality) framework.*
