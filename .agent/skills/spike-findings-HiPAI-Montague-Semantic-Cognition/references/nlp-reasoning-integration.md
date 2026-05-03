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
- **Missing Disjointness**: Never omit `AllDisjoint` when creating safety gates. Without explicit disjointness between `RestrictedAction` and `PermittedAction`, the reasoner will simply infer that the action is both, failing to raise the required `OwlReadyInconsistentOntologyError`.

## Hard Patterns
- **Mandatory AllDisjoint**: Every T1 restriction must be paired with an `AllDisjoint` declaration.

### Pattern: Mandatory AllDisjoint
Every T1 restriction must be paired with an `AllDisjoint` declaration.
  ```python
  with onto:
      class RestrictedAction(Action):
          equivalent_to = [Action & property.some(Range)]
      AllDisjoint([RestrictedAction, PermittedAction])
  ```

### Pattern: Relational Typing
Leverage domain/range restrictions for "free" subject typing (e.g., a subject of `harms` is automatically typed as a `MoralAgent` in the same reasoner pass).

### Pattern: Snapshot Reasoning
Always perform reasoner passes on a World snapshot/clone (e.g., using a temporary SQLite DB) if an inconsistency block is possible. This prevents "dirtying" the main world model and provides transaction-per-claim semantics.

### Pattern: Relaxed-Sync Diagnostics
To identify *why* a block occurred:
1. Catch `OwlReadyInconsistentOntologyError`.
2. Temporarily `.destroy()` the `AllDisjoint` gate axioms.
3. Re-run `sync_reasoner()` on the relaxed world.
4. Inspect the individual's `.is_a` or `isinstance()` to see which `RestrictedAction` classes were inferred.

### Pattern: Transitive Projection (Neo4j Integration)
When projecting memberships to Neo4j, walk `cls.ancestors()` to ensure the full type hierarchy is queryable via Cypher.

### Pattern: Downstream Invariant
Projection to Neo4j must only occur after a successful `sync_reasoner()` pass. Neo4j is a read-model for inferred truth; OWL is the source of truth for reasoning.

### Pattern: Execution Metadata
- **Java Runtime**: HermiT (the default reasoner) requires a JVM.
- **Model Loading**: spaCy models must be downloaded before use.

## Origin
Synthesized from spikes: 001
Source files available in: sources/001-spacy-owlready2-integration/
