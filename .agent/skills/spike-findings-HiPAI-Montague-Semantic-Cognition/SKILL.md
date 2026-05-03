---
name: spike-findings-HiPAI-Montague-Semantic-Cognition
description: Implementation blueprint from spike experiments. Requirements, proven patterns, and verified knowledge for building HiPAI-Montague-Semantic-Cognition. Auto-loaded during implementation work.
---

<context>
## Project: HiPAI-Montague-Semantic-Cognition

Validate the architectural shift from regex-based NLP and manual property-tagging Cypher to a professional pipeline using spaCy (for lemmatization and dependency parsing) and owlready2 (for Description Logic reasoning via HermiT).

Spike sessions wrapped: 2026-05-03
</context>

<requirements>
## Requirements

- Must use spaCy for surface NLP (lemmatization, POS, dep-parse).
- Must use owlready2 for ontological reasoning and consistency checking.
- Must support transitive subsumption (e.g., Aristotle -> Philosopher -> Man -> Mortal).
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| NLP & Reasoning Integration | references/nlp-reasoning-integration.md | Transitive subsumption (Aristotle is Mortal) successfully inferred via spaCy/owlready2. |

## Source Files

Original spike source files are preserved in `sources/` for complete reference.
</findings_index>

<metadata>
## Processed Spikes

- 001-spacy-owlready2-integration
</metadata>
