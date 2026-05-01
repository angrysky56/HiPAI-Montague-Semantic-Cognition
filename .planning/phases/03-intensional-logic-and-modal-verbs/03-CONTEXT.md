# Phase 3 Context: Intensional Logic and Modal Verbs

## Objective
Extend the HiPAI semantic cognition engine to handle intensionality and propositional attitudes. This includes modal verbs ("must", "can", "may") and attitude verbs ("believe", "know", "think"), allowing the system to represent and reason about different epistemic and deontic states within the FalkorDB world model.

## Current State
- **Phase 1**: Implemented "Disambiguate First" parsing and constraint pruning.
- **Phase 2**: Added tense (past/future) and basic Montague quantifiers (Some/No).
- **Limitation**: The system currently treats all observations as flat, extensional facts in the "actual world". It cannot represent nested beliefs (e.g., "Alice believes that Socrates is mortal") or modal requirements (e.g., "Socrates must be human").

## Scope
- **Modal Verbs**: Support for "must", "can", "may", "should".
- **Attitude Verbs**: Support for "believe", "know", "doubt".
- **Epistemic States**: Representation of "worlds" or "belief spaces" in the graph.
- **Intensional Parsing**: Expanding `synthesis.py` to handle "that" clauses and nested propositions.

## Constraints
- Must maintain compatibility with the existing `Observation` and `Relation` Pydantic models (expanding them if necessary).
- Must stay within the FalkorDB graph structure, potentially using subgraphs or labeled "epistemic spaces".
