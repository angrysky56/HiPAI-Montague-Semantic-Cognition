# Spike Conventions

## Stack
- **Language**: Python 3.13 (via `uv`)
- **NLP**: `spaCy` (model: `en_core_web_sm`)
- **Reasoning**: `owlready2` with `HermiT` reasoner (requires Java)

## Patterns
- **Extraction**: 
  - Subjects: `nsubj`, `nsubjpass`
  - Attributes/Adjectives: `attr`, `acomp`
  - Lemmatization should be used for class/individual names to normalize surface forms.
- **Ontology**: 
  - Dynamically create classes using `types.new_class(name, (Thing,))`.
  - Use `with onto: sync_reasoner()` to propagate inferences.
- **Verification**: 
  - Use `isinstance(individual, class_obj)` to verify membership, as it handles inferred classes automatically.
