# PLAN-2: Implement ambiguity resolution and context-aware parsing

## Objective
Refactor the parsing logic in `synthesis.py` (`HIPAIManager.add_belief`) to collect all possible logical interpretations of an input string rather than returning on the first regex match. Introduce a formal exception for ambiguity detection.

## Context
Currently, `add_belief` applies a series of string and regex checks sequentially, returning immediately when one matches. To support the "Disambiguate First, Parse Later" architecture, the engine must evaluate all patterns, collect all valid `Observation` objects, and determine if the input is ambiguous.

## Tasks
1. **Create `exceptions.py`**:
   - Define `AmbiguityDetectedError(Exception)`.
   - The exception should store the list of possible interpretations (e.g., list of `Observation` or dictionaries).

2. **Refactor `HIPAIManager.add_belief` (`synthesis.py`)**:
   - Instead of returning early, append each matched pattern's resulting `Observation` and metadata to a `possible_parses` list.
   - Example matches to check comprehensively: `"X is not a Y"`, `"X is a Y"`, `"All X are Y"`, `"X is not Y"`, `"X is Y"`, `"X has Y"`, `"X are Y"`, and the relational verbs.
   - If `len(possible_parses) > 1`, raise `AmbiguityDetectedError(possible_parses)`.
   - If `len(possible_parses) == 1`, proceed with `world_model.incorporate_observation` and return the success message.
   - If `len(possible_parses) == 0`, fallback to the unstructured belief creation.

## Validation
- The `HIPAIManager` should now successfully raise `AmbiguityDetectedError` when an input mathematically satisfies multiple parsing rules (e.g., if a sentence could be interpreted both as an entity property and a relation).
