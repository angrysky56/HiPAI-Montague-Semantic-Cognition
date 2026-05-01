# PLAN-3: Refactor the Synthesis Engine's parsing layer and MCP integration

## Objective
Implement world-model constraint pruning for parsed interpretations and wire the ambiguity pushback loop into the MCP server interface.

## Context
If multiple parses are generated for a sentence, some might be logically invalid according to existing constraints (e.g., the Paraclete Protocol or ontological facts). We need to filter out invalid parses before alerting the client. Finally, `mcp_server.py` must catch `AmbiguityDetectedError` and format it appropriately for the client LLM.

## Tasks
1. **Prune Invalid Parses (`synthesis.py`)**:
   - In `HIPAIManager.add_belief`, after collecting `possible_parses`, iterate over them and check them against `world_model` constraints.
   - Wait, `world_model` constraint checking is primarily for relations (e.g., `check_constraint`). We can simulate or use `check_constraint` for relational observations, or simply check for glaring contradictions.
   - Filter `possible_parses` down to `valid_parses`.
   - Update the ambiguity logic: raise `AmbiguityDetectedError(valid_parses)` only if `len(valid_parses) > 1`.

2. **Update MCP Server (`mcp_server.py`)**:
   - In `add_belief(text: str)`, catch `AmbiguityDetectedError`.
   - Format the error into a structured string: `"AmbiguityDetected: The statement is ambiguous. Please call the tool again specifying your exact intent from these options:\nOption 1: ...\nOption 2: ..."`.
   - Ensure the options are formatted clearly so the LLM can understand the distinct logical forms (e.g., "Relation X -> Y" vs "Entity X has property Y").

## Validation
- When calling the `add_belief` tool via MCP, if an ambiguous string is provided, the tool should return the `AmbiguityDetected` string instead of throwing an unhandled exception or making a default choice.
- Pruning should successfully eliminate invalid parses so that an initially ambiguous sentence resolves cleanly if only one interpretation is logically consistent.
