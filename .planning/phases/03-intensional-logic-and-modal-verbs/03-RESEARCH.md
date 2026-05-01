# Phase 3 Research: Intensional Logic and Modal Verbs

## Overview
This phase extends the extensional world model into the intensional realm, allowing the HiPAI engine to distinguish between what IS, what MUST be, and what is BELIEVED to be.

## 1. Intensional Operators in Montague Grammar
In Montague semantics, intensionality is handled via type-shifting from $a$ to $\langle s, a \rangle$ (where $s$ is the type of indices or possible worlds). While we cannot represent infinite possible worlds in FalkorDB, we can implement **propositional attitudes** (Attitude Verbs) and **modality** as follows:

### Attitude Verbs (Believe, Know, Say)
- **Semantic Type**: $\langle \langle s, t \rangle, \langle e, t \rangle \rangle$ (A function from propositions to properties of individuals).
- **Factivity**: 
    - `Know` is factive: $Know(x, p) \vdash p$.
    - `Believe` is non-factive: $Believe(x, p) \not\vdash p$.
- **Graph Mapping**: Link an `Entity` node (the agent) to an `Observation` node (the proposition) via a labeled edge (`BELIEVES`, `KNOWS`).

### Modal Verbs (Must, Can, May)
- **Alethic Modality**: Necessity/Possibility ($P$ must be true).
- **Deontic Modality**: Obligation/Permission ($P$ is required).
- **Graph Mapping**: Add a `modality` property to the `Observation` node and/or the `Relation` edge.

## 2. Graph Schema Evolution

### Observation Node
- `tense`: (Already implemented)
- `modality`: `Optional[Literal["must", "can", "may", "should"]]`
- `subject_id`: `Optional[str]` (The agent whose belief/statement this is)

### Relation Edge
- `modality`: `Optional[Literal["must", "can", "may", "should"]]`

### New Edge Types
- `[:BELIEVES]`, `[:KNOWS]`, `[:SAYS]` pointing from `Entity` to `Observation`.

## 3. Parsing Strategy
1. **Attitude Pattern**: `r"^(.+?)\s+(believes?|knows?|says?|thinks?)\s+that\s+(.+)$"`
    - Group 1: Subject (Agent)
    - Group 2: Verb (Attitude)
    - Group 3: Proposition (to be parsed recursively by `add_belief`)
2. **Modal Pattern**: Update existing patterns to check for "must", "can", "may", "should" preceding the main verb or copula.

## 4. Logical Inference
- `evaluate_hypothesis` must be updated to:
    - Check for modal necessity: If checking "X is Y" and graph has "X must be Y", return Entailed.
    - Handle factive attitudes: If checking "P" and graph has "X knows that P", return Entailed.
    - Distinguish belief from fact: If checking "P" and graph only has "X believes that P", return Undetermined (unless querying Alice's specific beliefs).
