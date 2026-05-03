# Contributing to HiPAI-Montague

## Development Workflow
We follow a Test-Driven Development (TDD) approach with a focus on empirical validation via spikes.

1.  **Architecture**: All changes must align with `DESIGN.md` (v0.5).
2.  **Environment**: Use `uv` for dependency management.
3.  **Spikes**: Significant architectural changes should be prototyped in `.planning/spikes/` first.
4.  **Tests**: New features must include unit tests in `tests/` and pass the 21-test regression suite.

## Technical Standards
- **Python**: Use 3.12+ features.
- **Ontology**: Every T1 constraint must have a corresponding `AllDisjoint` axiom.
- **Safety**: Never bypass the `IsolationWorld` pattern for reasoner passes on user-submitted claims.

## Branching & Commits
- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
- Main development occurs on `master` (or the feature branch as assigned).

---
*Operational ethics require that even the contribution process is traceable and consistent.*
