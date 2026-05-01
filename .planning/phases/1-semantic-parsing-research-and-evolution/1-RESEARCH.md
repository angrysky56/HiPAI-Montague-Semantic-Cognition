# Phase 1: Semantic Parsing Research and Evolution - Research

## Standard Stack
- **Parsing Frameworks**: Since the system operates as an MCP server, the "Disambiguate First, Parse Later" approach should be **MCP-native**. The server does *not* need an onboard LLM. Instead, it relies on the client LLM that is making the tool calls to handle disambiguation.
- **Evaluation & Benchmarking**: `pytest` integrated with the AmP (Ambiguous Parsing) framework for testing various linguistic ambiguities (scope, attachment, vagueness). Ensure tests simulate an LLM client receiving ambiguity errors and re-attempting calls.
- **Interactive Disambiguation**: The server acts as a strict logical gatekeeper. When it detects ambiguity, it returns a structured error to the client LLM, forcing the client to choose the correct interpretation.

## Architecture Patterns
- **MCP-Native Two-Stage Parsing**:
  1. **Detection (Server-Side)**: The Montague parser attempts to parse the natural language input provided by the client LLM. If multiple valid logical interpretations can be generated, it does not pick a default.
  2. **Pushback (Server -> Client)**: The server returns an `AmbiguityDetected` response containing the plausible interpretations formatted clearly.
  3. **Disambiguation (Client-Side)**: The client LLM reads the options, uses its conversational context, and makes a new, precise tool call with the disambiguated logical form.
- **Contextual Grounding**: The server can evaluate the ambiguous parses against the FalkorDB world model. If an interpretation violates existing constraints, it is silently pruned. If only one valid interpretation remains, it proceeds without prompting the client.

## Don't Hand-Roll
- **LLM Disambiguation Agent**: Do not hand-roll a secondary LLM pipeline inside the MCP server to resolve ambiguity. Rely on the client LLM's context.
- **Hallucination/Ontology Gap Detection**: Do not silently fail or guess when a concept isn't in the FalkorDB schema. Explicitly return a schema error to the LLM so it knows it must define the concept first.
- **Complex Dependency Parsing**: Rely on the client LLM's natural language understanding rather than building a custom AST/dependency tree parser for raw text.

## Common Pitfalls
- **Silent Failures & Guessing**: Picking a "default" interpretation when the text is ambiguous. *Mitigation:* Always push back to the client LLM if multiple valid parses exist.
- **Unclear Error Messages**: Returning a generic parsing error to the LLM. *Mitigation:* Return structured options (e.g., "Did you mean Option A: ... or Option B: ...") so the LLM knows exactly how to correct its tool call.
- **Redundant Processing**: Re-evaluating known facts. *Mitigation:* Prune interpretations that contradict the world model before reporting ambiguity.

## Code Examples

### MCP-Native Ambiguity Resolution Pattern
```python
def add_belief_tool(text: str) -> str:
    # Attempt to parse into Montague logical forms
    possible_parses = montague_parser.parse_all(text)
    
    valid_parses = []
    # Prune parses that violate world model constraints
    for parse in possible_parses:
        if world_model.check_constraint(parse):
            valid_parses.append(parse)
            
    if len(valid_parses) > 1:
        # Push back to the client LLM
        options = "\n".join([f"Option {i+1}: {p.to_natural_language()}" for i, p in enumerate(valid_parses)])
        return f"AmbiguityDetected: The statement is ambiguous. Please call the tool again specifying your exact intent from these options:\n{options}"
    
    elif len(valid_parses) == 1:
        # Proceed with the single valid interpretation
        world_model.add_belief(valid_parses[0])
        return "Belief successfully added."
    
    else:
        return "Error: No valid interpretation found that consistent with the current world model."
```
