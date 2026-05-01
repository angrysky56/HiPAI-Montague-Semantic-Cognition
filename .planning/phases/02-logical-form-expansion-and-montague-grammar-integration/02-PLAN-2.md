---
wave: 2
depends_on: ["02-PLAN-1.md"]
files_modified:
  - src/hipai/synthesis.py
  - src/hipai/world_model.py
  - tests/test_ambiguity.py
autonomous: true
---

# Phase 2, Wave 2: Quantifier Support

## Objective
Expand the semantic parser to support existential ("some", "a few") and negative universal ("no") quantifiers, mapping them into the FalkorDB graph correctly according to Montague semantics.

## Tasks

<task>
<read_first>
- `src/hipai/synthesis.py`
</read_first>
<action>
1. In `src/hipai/synthesis.py` (`add_belief`), add a new parsing block for Existential Quantifiers ("Some X are Y" or "Some X is Y").
   - Match sentences starting with "Some " and containing " are " or " is ".
   - Extract the Subject Class (`X`) and Object Property (`Y`).
   - Create an anonymous `Observation` containing an `Individual` with `name=f"anonymous_{uuid4().hex}"`, `properties=[Y]`.
   - Add to `possible_parses` with `type="existential_belief"`, `concept_name=f"Concept_{X.capitalize()}"`.
2. Add a new parsing block for Negative Universal Quantifiers ("No X are Y").
   - Match sentences starting with "No " and containing " are " or " is ".
   - Extract the Subject Class (`X`) and Object Property (`Y`).
   - Add to `possible_parses` with `type="negative_universal_belief"`, `concept_name=f"Concept_{X.capitalize()}"`, `property_key=f"prop_not_{Y}"`.
</action>
<acceptance_criteria>
- `synthesis.py` contains patterns matching "Some " and "No ".
- Existential parses use anonymous UUIDs for `Individual` generation.
</acceptance_criteria>
</task>

<task>
<read_first>
- `src/hipai/synthesis.py`
- `src/hipai/world_model.py`
</read_first>
<action>
1. In `src/hipai/synthesis.py`, update the `if len(valid_parses) == 1:` block to handle graph incorporation for the new metadata types.
2. For `type == "existential_belief"`:
   - Call `self.world_model.incorporate_observation(parse["observation"])` to create the anonymous entity.
   - Run a cypher query to link the anonymous entity to the Concept:
     `MATCH (e:Entity {id: $subject_id}) MERGE (c:Concept {name: $concept_name}) MERGE (e)-[:INSTANCE_OF]->(c)`
3. For `type == "negative_universal_belief"`:
   - Similar to `universal_belief`, create the Concept if missing, and set the property on the Concept:
     `MATCH (c:Concept {name: $concept_name}) SET c.$property_key = true`
</action>
<acceptance_criteria>
- The single-parse resolution logic handles `existential_belief` and executes graph queries linking the anonymous entity to a Concept.
- The logic handles `negative_universal_belief` and sets the `not_Y` property on the Concept.
</acceptance_criteria>
</task>

<task>
<read_first>
- `tests/test_ambiguity.py`
</read_first>
<action>
1. Add a test in `tests/test_ambiguity.py` (or a new `test_quantifiers.py`) to verify the new parses.
2. Test "Some humans are immortal" -> Verify an anonymous entity is created that is an INSTANCE_OF `Concept_Human` with property `immortal`.
3. Test "No dogs are cats" -> Verify `Concept_Dog` has property `prop_not_cats = true`.
</action>
<acceptance_criteria>
- New tests for "Some" and "No" quantifiers are present.
- `pytest tests/test_ambiguity.py` (or the new test file) exits 0.
</acceptance_criteria>
</task>

## Verification
<must_haves>
- [ ] Existential quantifiers ("Some") result in an anonymous entity linked to the correct Concept.
- [ ] Negative universal quantifiers ("No") result in a negative property assignment on the Concept.
- [ ] Graph tests confirm correct creation of anonymous nodes and Concept properties.
</must_haves>
