# Phase 2: Layer 2 Refactor & API Mirroring

## Objective
Refactor the core ingestion pipeline to replace brittle regex patterns with spaCy-based dependency parsing and transition the primary knowledge store from direct FalkorDB manipulation to an OWL-backed authoritative model using `owlready2`.

## Proposed Changes

### 1. Linguistic Parsing Layer (`src/hipai/parser.py`)
- **ClaimExtractor**: A new class that uses spaCy to convert natural language into `Observation` or `Belief` structures.
- **Dependency Patterns**:
    - `nsubj` + `AUX` + (`attr`|`acomp`) -> Type/Property attribution.
    - `nsubj` + `VERB` + `dobj` -> Binary relation.
    - `det` (`all`, `no`, `some`) -> Quantifier extraction.
    - `neg` -> Negation handling.
- **Ontology Alignment**: The parser should use `OntologyManager` to check if entities and relations already exist or need creation.

### 2. World Model Refactor (`src/hipai/world_model.py`)
- **OWL Integration**: Embed `OntologyManager` into `WorldModel`.
- **Primary Store Shift**: 
    - `owlready2` (SQLite) becomes the source of truth for logic and hierarchy.
    - FalkorDB remains as a "Read Model" for graph visualizations or fast proximity queries if needed, but its state must be derived from the OWL model.
- **Constraint Engine**: Re-implement `check_constraint` using `owlready2` consistency checks.

### 3. Synthesis Engine Bridge (`src/hipai/synthesis.py`)
- **HIPAIManager Refactor**: 
    - Update `add_belief` to delegate parsing to `ClaimExtractor`.
    - Mirror the legacy API signatures to ensure the MCP server and existing tests continue to function.
    - Implement "Paraclete Protocol" logic using the OWL model's reasoning capabilities.

## Verification Plan

### Automated Tests
- Run `tests/test_synthesis.py` and `tests/test_world_model.py`.
- Create `tests/test_parser.py` to specifically test the new spaCy logic.
- Verify that `Socrates is a man` correctly seeds a `man` Individual in the ontology.

### Manual Verification
- Use `sanity_check.py` to ensure the environment is still stable.
- Inspect `world.db` using `owlready2` to confirm individuals are being created with correct properties.

## Progress Tracking
- [ ] Implement `ClaimExtractor` in `src/hipai/parser.py` [TASK-1]
- [ ] Integrate `OntologyManager` into `WorldModel` [TASK-2]
- [ ] Refactor `HIPAIManager.add_belief` to use `ClaimExtractor` [TASK-3]
- [ ] Update `mcp_server.py` to use refactored components [TASK-4]
- [ ] Verify regression with existing test suite [TASK-5]
