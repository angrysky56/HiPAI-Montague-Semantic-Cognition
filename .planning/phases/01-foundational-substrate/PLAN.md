# Phase 1: Foundational Substrate

## Objective
Establish the core Python and Java environment required for the v0.5 architecture, and initialize the authoritative ontology.

## Plans

### PLAN-1: Dependency Setup
- [x] Update `pyproject.toml` with `spacy`, `owlready2`.
- [x] Install dependencies via `uv pip install`.
- [x] Download the `en_core_web_md` model for spaCy.
- [x] Verify environment with a simple sanity check script.

### PLAN-2: World Initialization
- [x] Create `src/ontology_manager.py` to handle `owlready2` initialization.
- [x] Setup persistent `world.db` (sqlite) in the root.
- [x] Implement the `OntologyManager.init_world()` method.

### PLAN-3: Axiom Seeding
- [x] Define the base T1 hierarchy: `Entity`, `Action`, `Agent`, `Patient`.
- [x] Define core properties: `harms`, `deceives`, `violates_agency`.
- [x] Seed initial `AllDisjoint` gates for the Paraclete substrate.

## Verification (UAT)
- [x] **UAT-1**: `uv run python src/ontology_manager.py --sanity` prints "Ontology Ready".
- [x] **UAT-2**: `world.db` exists and contains the seeded classes.
- [x] **UAT-3**: spaCy can parse a simple sentence "Socrates harms the Pig".
