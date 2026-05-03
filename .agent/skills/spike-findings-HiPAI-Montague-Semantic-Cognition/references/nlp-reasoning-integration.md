# NLP & Reasoning Integration

## Requirements
- Must use spaCy for surface NLP (lemmatization, POS, dep-parse).
- Must use owlready2 for ontological reasoning and consistency checking.
- Must support transitive subsumption (e.g., Aristotle -> Philosopher -> Man -> Mortal).

## How to Build It

### 1. Setup Dependencies
```python
# Required packages
# uv pip install spacy owlready2
# python -m spacy download en_core_web_sm
```

### 2. NLP Pipeline (spaCy)
Use spaCy dependency parsing to extract Subject and Attribute/Attribute Complement.
```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_logic_claim(sentence):
    doc = nlp(sentence)
    subject = None
    attribute = None
    
    for token in doc:
        # Extract Subject
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject = token.lemma_.capitalize()
        # Extract Attribute (Noun) or Adjective Complement
        elif token.dep_ in ["attr", "acomp"]:
            attribute = token.lemma_.capitalize()
            
    return subject, attribute
```

### 3. Ontological Mapping (owlready2)
Map extracted terms to OWL classes and individuals.
```python
from owlready2 import *

onto = get_ontology("http://test.org/ontology.owl")

def add_claim_to_onto(subject, attribute, is_universal=False):
    with onto:
        if is_universal:
            # "All X are Y" -> X subClassOf Y
            class_sub = types.new_class(subject, (Thing,))
            class_super = types.new_class(attribute, (Thing,))
            class_sub.is_a.append(class_super)
        else:
            # "X is a Y" -> X Individual of Class Y
            class_attr = types.new_class(attribute, (Thing,))
            individual = class_attr(subject)
```

### 4. Reasoning
Run the reasoner to compute transitive subsumption and check consistency.
```python
def run_reasoner():
    with onto:
        sync_reasoner()

def verify_membership(individual_name, class_name):
    ind = onto.search_one(iri=f"*{individual_name}")
    cls = onto.search_one(iri=f"*{class_name}")
    if ind and cls:
        # Use isinstance to check both direct and inferred classes
        return isinstance(ind, cls)
    return False
```

## What to Avoid
- **Shadowing `Thing`**: Do not redefine `class Thing(Thing)` inside functions or local scopes; it causes `UnboundLocalError`.
- **Direct Class Checks**: Avoid checking `individual.is_a` directly for membership; it only contains direct parents. Use `isinstance(individual, Class)`.
- **Regex Parsing**: Stop using regex for verb stemming or plural handling. spaCy's `.lemma_` handles this structurally.

## Constraints
- **Java Runtime**: HermiT (the default reasoner) requires a JVM.
- **Model Loading**: spaCy models must be downloaded before use.

## Origin
Synthesized from spikes: 001
Source files available in: sources/001-spacy-owlready2-integration/
