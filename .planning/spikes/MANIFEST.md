# Spike Manifest

## Idea
Validate the architectural shift from regex-based NLP and manual property-tagging Cypher to a professional pipeline using spaCy (for lemmatization and dependency parsing) and owlready2 (for Description Logic reasoning via HermiT).

## Requirements
- Must use spaCy for surface NLP (lemmatization, POS, dep-parse).
- Must use owlready2 for ontological reasoning and consistency checking.
- Must support transitive subsumption (e.g., Aristotle -> Philosopher -> Man -> Mortal).

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | spacy-owlready2-integration | standard | Given logical claims in natural language, when parsed by spaCy and reasoned by owlready2, then transitive entails (like "Aristotle is mortal") are correctly inferred. | **VALIDATED ✓** | nlp, reasoning, owl |
