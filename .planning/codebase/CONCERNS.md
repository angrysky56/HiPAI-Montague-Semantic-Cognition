# Concerns

## Technical Debt
- **Large Files**: `world_model.py` (>900 lines) and `synthesis.py` (>700 lines) are becoming difficult to maintain and should be split into sub-modules (e.g., `graph/`, `reasoning/`).
- **High Complexity**: `WorldModel.check_constraint` and `WorldModel.calibrate_belief` have deep nesting (level 6) and complex conditional logic.

## Risks
- **FalkorDB Dependency**: No fallback or mock implementation for testing without a running database.
- **Performance**: KMeans clustering and vector embeddings may not scale well for extremely large world models without optimization.
- **Ambiguity**: Semantic parsing of natural language into formal observations is a brittle area prone to edge cases.
