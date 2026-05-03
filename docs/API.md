# API Reference (HiPAI-Montague v0.5)

The HiPAI-Montague engine is exposed via a FastMCP server. Below are the primary tool categories.

## 1. Paraclete Safety Tools (L5)

### `check_action(subject, relation, object)`
- **Purpose**: Validates if an action triple violates any T1 deontological constraints.
- **Backend**: Runs `sync_reasoner()` on an `IsolationWorld` snapshot.
- **Returns**: `BLOCKED` (with culprit reasons) or `PERMITTED`.

### `calibrate_belief(object, axiom, relation)`
- **Purpose**: Epistemic integrity check. Searches for disconfirming evidence of the object's moral status.
- **Verdict**: `BLOCK_CONFIRMED`, `BLOCK_UNCERTAIN`, or `BLOCK_CHALLENGED`.

## 2. Semantic Cognition Tools (L2/L3)

### `add_belief(text)`
- **Purpose**: Processes natural language text into OWL individuals/properties via spaCy.
- **Invariants**: All claims are tested for consistency before being committed to the main world.

### `evaluate_hypothesis(hypothesis)`
- **Purpose**: Checks if a natural language hypothesis is entailed by the current world model.
- **Backend**: Uses the HermiT reasoner's subsumption and membership checks.

## 3. World Management (L4)

### `get_current_state()`
- **Purpose**: Returns a summary of all active individuals, classes, and properties in the ontology.

### `clear_graph()`
- **Purpose**: Resets the world model state. Preserves the foundational T1 axioms (Action, Entity, etc.).

---
*For raw access, use the internal `OntologyManager` Python class.*
