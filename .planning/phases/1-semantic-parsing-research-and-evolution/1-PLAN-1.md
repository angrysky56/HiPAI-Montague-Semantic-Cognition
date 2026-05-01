# PLAN-1: Research and prototype advanced semantic parsing models

## Objective
Research SOTA "Disambiguate First, Parse Later" parsing patterns and design the architecture for the MCP server. 

## Context
Semantic parsing often struggles with ambiguity. Instead of internally picking a default parse or hand-rolling a custom NLP disambiguation pipeline, the system should push ambiguity back to the client LLM via the MCP protocol.

## Tasks
1. **Research (COMPLETED)**:
   - Evaluated various parsing strategies.
   - Decided on MCP-Native Two-Stage Parsing (Detection -> Pushback -> Disambiguation by Client).
   - Documented in `1-RESEARCH.md`.

2. **Prototype**:
   - The prototyping of the logic is integrated into PLAN-2 and PLAN-3, which cover the concrete modifications to `synthesis.py` and `mcp_server.py`.

## Validation
- The research phase is complete and documented.
- Architecture decisions correctly guide the subsequent implementation plans.
