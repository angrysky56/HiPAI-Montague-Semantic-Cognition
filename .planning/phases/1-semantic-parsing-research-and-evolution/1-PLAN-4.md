# PLAN-4: Build a comprehensive test suite for parsing edge cases

## Objective
Develop a robust test suite using `pytest` to verify the ambiguity detection and resolution mechanisms.

## Context
We must ensure that the "Disambiguate First, Parse Later" architecture works as intended without regressing existing functionality. The tests should simulate the client LLM behavior by evaluating whether the correct ambiguity signals are emitted by the server.

## Tasks
1. **Ambiguity Detection Tests (`tests/test_synthesis.py` and `tests/test_hipai.py`)**:
   - Write tests that feed ambiguous statements into `HIPAIManager.add_belief`.
   - E.g., A sentence that could match both an "X is Y" pattern and a relational verb pattern. Wait, currently the patterns might not overlap, so we may need to define overlapping patterns or test specifically designed ambiguous phrases.
   - Assert that `AmbiguityDetectedError` is raised and contains the correct number of interpretations.

2. **MCP Tool Tests (`tests/test_mcp_server.py` - new file)**:
   - Test the `add_belief` tool wrapper directly.
   - Assert that when `AmbiguityDetectedError` is triggered, the tool returns the properly formatted "AmbiguityDetected" string instead of crashing.
   - Test a successful, unambiguous addition to ensure it returns the success payload.

3. **Pruning Tests (`tests/test_synthesis.py` or similar)**:
   - Create a test where multiple parses exist, but one is logically invalid (e.g., contradicts an existing strict axiom or graph constraint).
   - Assert that the invalid parse is pruned and the system gracefully accepts the remaining valid parse without raising an ambiguity error.

## Validation
- All `pytest` suites pass.
- Coverage for `synthesis.py` parsing logic and `mcp_server.py` `add_belief` wrapper is comprehensive, demonstrating both successful execution paths and proper ambiguity pushbacks.
