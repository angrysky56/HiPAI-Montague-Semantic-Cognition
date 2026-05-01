# Phase 3 Verification Plan: Intensional Logic

## Automated Tests
- `tests/test_intensionality.py`: New test file.
    - `test_attitude_parsing`: "Alice believes that..."
    - `test_factive_entailment`: "Bob knows that P" -> P is Entailed.
    - `test_non_factive_belief`: "Alice believes that P" -> P is Undetermined.
    - `test_modal_parsing`: "Socrates must be human."
    - `test_modal_necessity`: "Must(P)" -> P is Entailed.

## Manual Verification
- Inspect FalkorDB using the CLI or a test script to ensure:
    - (Alice)-[:BELIEVES]->(ObsNode).
    - ObsNode has `modality="must"`.
