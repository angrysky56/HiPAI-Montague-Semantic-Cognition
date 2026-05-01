# Roadmap

## Phase 1: Semantic Parsing Research and Evolution
**Goal:** Research SOTA semantic parsing techniques and implement advanced ambiguity resolution to harden the parser against edge cases.

- [x] PLAN-1: Research and prototype advanced semantic parsing models
- [x] PLAN-2: Implement ambiguity resolution and context-aware parsing
- [x] PLAN-3: Refactor the Synthesis Engine's parsing layer to use the new approach
- [x] PLAN-4: Build a comprehensive test suite for parsing edge cases

## Phase 2: Logical Form Expansion and Montague Grammar Integration
**Goal:** Expand the parser's vocabulary and handle more complex linguistic structures (e.g., quantification and tense).

- [x] PLAN-1: Integrate Tense (past/future) into models and graph persistence
- [x] PLAN-2: Implement support for Montague quantifiers (Some, No)
- [x] PLAN-3: Resolve quantifier pattern ambiguity and harden synthesis

## Phase 3: Intensional Logic and Modal Verbs
**Goal:** Support propositional attitudes (Believe, Know) and modal necessity (Must, Can).

- [ ] PLAN-1: Implement attitude verb parsing and propositional graph reification
- [ ] PLAN-2: Implement modal auxiliary parsing and intensional entailment logic
- [ ] PLAN-3: Extend evaluation engine for factive and modal reasoning
