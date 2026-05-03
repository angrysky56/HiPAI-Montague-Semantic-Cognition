---
spike: 001
name: spacy-owlready2-integration
type: standard
validates: "Given logical claims in natural language, when parsed by spaCy and reasoned by owlready2, then transitive entails (like 'Aristotle is mortal') are correctly inferred."
verdict: PENDING
related: []
tags: [nlp, reasoning, owl]
---

# Spike 001: spaCy and owlready2 Integration

## What This Validates
This spike validates whether we can replace regex-based patterns with a spaCy-driven NLP pipeline and owlready2-driven ontological reasoning to handle complex transitive subsumption.

## Research
- **spaCy**: Provides robust lemmatization and dependency parsing. "Socrates is a man" maps "Socrates" to `nsubj`, "is" to `ROOT`, and "man" to `attr`.
- **owlready2**: Pythonic interface to OWL 2. Supports HermiT reasoner (requires Java) for fixed-point subsumption computation.

## How to Run
```bash
# 1. Install dependencies
uv pip install spacy owlready2
python -m spacy download en_core_web_sm

# 2. Run the spike script
python .planning/spikes/001-spacy-owlready2-integration/spike.py
```

## What to Expect
- spaCy correctly identifies lemmatized subjects and attributes.
- owlready2 builds an ontology with classes (Man, Mortal, Philosopher) and individuals (Socrates, Aristotle).
- `sync_reasoner()` infers that Aristotle is mortal via the chain: Aristotle -> Philosopher -> Man -> Mortal.

## Investigation Trail
- [2026-05-03] Initial setup. Researching spaCy dependency tags for copula sentences.
- [2026-05-03] Verified Java 21 availability for HermiT.

## Results
**VERDICT: VALIDATED ✓**

The spike successfully demonstrated:
1. **spaCy parsing**: Correctly identified subjects and attributes in both individual membership ("X is a Y") and class inclusion ("All X are Y") sentences.
2. **owlready2 ontological mapping**: Mapped the extracted terms to OWL classes and individuals dynamically.
3. **HermiT Reasoning**: Correctly inferred that Aristotle is Mortal via the chain:
   - `Aristotle` (Individual) ∈ `Philosopher` (Class)
   - `Philosopher` (Class) ⊆ `Man` (Class)
   - `Man` (Class) ⊆ `Mortal` (Class)

The reasoner reparented the classes and confirmed `isinstance(Aristotle, Mortal)` is True.

### Surprises/Gotchas:
- **Python Scoping**: `class Thing(Thing)` inside a function causes `UnboundLocalError` if not handled correctly.
- **Verification**: `instance.is_a` only shows direct classes; `isinstance(instance, Class)` is required to check inferred memberships.
- **spaCy Tags**: Adjectives like "mortal" in "All men are mortal" are tagged as `acomp` (adjective complement), while nouns are tagged as `attr` (attribute).
