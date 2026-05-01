# Conventions

## Coding Standards
- **Style**: PEP 8 compliant, enforced via Ruff.
- **Formatting**: 88 line length, double quotes.
- **Type Hints**: Mandatory for all function signatures and complex variables.
- **Docstrings**: Expected for classes and public functions (Google or Numpydoc style preferred).

## Patterns
- **Validation**: Use Pydantic `BaseModel` for all data interchange and persistence schemas.
- **Async**: `aiohttp` and `mcp` patterns for non-blocking I/O.
- **Errors**: Custom exception handling for semantic mismatches and graph inconsistencies.

## Tooling
- **Package Manager**: `uv` is the source of truth for dependencies.
- **Environment**: `.venv` in the project root.
