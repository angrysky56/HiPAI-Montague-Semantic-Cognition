# Codebase Structure

```text
.
├── src/hipai/                # Core library
│   ├── mcp_server.py         # Entry point & MCP tool definitions
│   ├── world_model.py        # FalkorDB / Graph Persistence logic
│   ├── synthesis.py          # Reasoning & HIPAIManager
│   ├── semantics.py          # Montague types & Lambda logic
│   ├── models.py             # Pydantic data models
│   ├── bridge.py             # Integration bridge
│   └── sources.py            # Data ingestion
├── tests/                    # Test suite
│   ├── test_semantics.py     # Unit tests for formal logic
│   ├── test_world_model.py   # Integration tests for graph logic
│   └── integration_db.py     # DB-specific integration helpers
├── config/                   # Configuration files
├── docs/                     # Documentation
├── docker-compose.yml        # Infrastructure setup (FalkorDB)
├── pyproject.toml            # Project metadata & dependencies
└── uv.lock                   # Dependency lockfile
```
