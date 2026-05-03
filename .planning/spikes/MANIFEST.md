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
| 002 | paraclete-deontic-restrictions | standard | Given "harms" restrictions, when "Socrates harms Pig", then Socrates is inferred as "MoralAgent". AND when a forbidden action is asserted, then `sync_reasoner()` raises `OwlReadyInconsistentOntologyError`. | **VALIDATED ✓** | nlp, reasoning, paraclete, owl |
| 003 | consistency-error-semantics | standard | Given T1 violations, can we identify the specific axiom IRI? Is the ontology recoverable after a block? How are simultaneous violations handled? | **VALIDATED ✓** | reasoning, errors, ebe-chain |
| 004 | neo4j-owl-sync | integration | [RE-SCOPE PENDING 002] Given OWL ontological state, determine if Neo4j remains necessary or if owlready2's quadstore handles performance and embeddings. | PENDING | integration, neo4j |
