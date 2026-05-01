# Testing

## Strategy
- **Unit Testing**: Focuses on `semantics.py` and `models.py` (no external dependencies).
- **Integration Testing**: Focuses on `world_model.py` and `synthesis.py`, requiring a FalkorDB instance.

## Execution
```bash
# Run all tests
pytest

# Run tests with uv
uv run pytest
```

## Coverage Areas
- **Semantic Composition**: Verifying lambda application and type-checking.
- **Graph Persistence**: Verifying node creation and Cypher query generation.
- **Clustering**: Verifying concept synthesis logic.
