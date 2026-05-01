# Phase 2 Research: Logical Form Expansion and Montague Grammar Integration

## Objective
Answer: "What do I need to know to PLAN this phase well?"
We need to expand the parser's vocabulary to handle complex linguistic structures, specifically quantifiers ("some", "every", "no") and tense (past, present, future), integrating them smoothly into the Montague Grammar semantic framework and the FalkorDB world model.

## Current State Analysis

### 1. `src/hipai/models.py`
- `Observation`: Represents a parsed statement. Contains `individuals` and `relations`.
- `Relation`: Has `source_id`, `target_id`, and `relation_type`.
- `Individual`: Has `name` and `properties`.
- None of these models currently have a way to represent **tense** or **temporal validity**.

### 2. `src/hipai/synthesis.py` (`add_belief`)
- The parser is a rules-based regex/string-matching system.
- Currently handles one quantifier explicitly: **"All X are Y"** (maps to `universal_belief` type, setting a boolean property directly on a Concept node).
- Does not handle "Some X are Y" or "No X are Y".
- Assumes present tense for everything ("is", "are", "has", "causes").

## Research & Proposed Architecture

### 1. Quantification in the Graph Model
In Montague semantics, noun phrases act as generalized quantifiers (sets of properties). In our graph representation:

*   **Universal ("Every X is Y" / "All X are Y")**:
    *   *Current*: Handled by setting `prop_Y = true` on `Concept_X`.
    *   *Extension*: Needs to apply to relations as well (e.g., "All X love Y" -> create a graph rule or a `UNIVERSAL_RELATION` edge between Concept nodes).
*   **Existential ("Some X is Y" / "A few X are Y")**:
    *   *Approach*: We cannot easily represent this universally on the Concept. Instead, "Some X is Y" implies the existence of at least one entity.
    *   *Graph Mapping*: Create an anonymous `Entity` node (e.g., `id: anonymous_<uuid>`) that is an `INSTANCE_OF` `Concept_X` and possesses `prop_Y`.
*   **Negative Universal ("No X are Y")**:
    *   *Approach*: Represents a strict constraint.
    *   *Graph Mapping*: Add a property `prop_not_Y = true` to `Concept_X` (for properties) or a T1/T2 constraint in the world model (for relations, e.g., "No humans fly" -> `DeontologicalAxiom` or equivalent ontological axiom blocking `FLY` relation for `Human`).

### 2. Tense (Past, Present, Future)
Natural language often carries temporal context:
- "Socrates *was* a man"
- "Alice *will visit* Bob"

*   **Model Update**: Add a `tense` field to the `Relation` and `Observation` models.
    *   `tense: Literal["past", "present", "future"] = "present"`
*   **Graph Mapping**: When creating edges (relations) or setting properties, add a metadata property `tense`. For example: `(Alice)-[:VISIT {tense: "future"}]->(Bob)`.
*   **Parser Update**: Expand the verb matching logic to detect tense. Instead of just looking for "is", look for "was", "will be", "has", "had", "will have".
    *   *Example*: "X was Y" -> `tense="past"`.
    *   *Example*: "X will cause Y" -> `tense="future"`.

### 3. Integration with `add_belief`
The `add_belief` method currently checks patterns sequentially and collects possible parses. We must add:
- **Quantifier Patterns**:
    - `"Some X are Y"` -> Generate anonymous individual.
    - `"No X are Y"` -> Generate negative universal belief / constraint.
- **Tense Patterns**:
    - Modify the "X is a Y" block to check for "X was a Y" / "X will be a Y".
    - Modify the Relational verbs block to capture helping verbs ("will", "did", "had") or verb endings ("-ed", "-s") to infer tense, or at least handle the basic forms explicitly.

## Requirements for Planning
To successfully implement this phase, the plan must:
1. Update `models.py` to support `tense`.
2. Update `world_model.py` to properly store `tense` metadata on nodes/edges and handle anonymous existence entities.
3. Update `synthesis.py` (`add_belief`) to parse new quantifiers ("some", "no").
4. Update `synthesis.py` (`add_belief`) to detect past/future tenses in basic relational and property patterns.
5. Create comprehensive tests in `test_synthesis.py` or a new test file for the new parsing capabilities.
