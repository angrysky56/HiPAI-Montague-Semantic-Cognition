---
wave: 1
depends_on: []
files_modified:
  - src/hipai/models.py
  - src/hipai/world_model.py
  - src/hipai/synthesis.py
  - tests/test_synthesis.py
autonomous: true
---

# Phase 2, Wave 1: Tense Integration

## Objective
Update the semantic models and the parsing engine to support temporal context (tense). Sentences like "Socrates was a man" or "Alice will visit Bob" should be parsed with past/future tense metadata.

## Tasks

<task>
<read_first>
- `src/hipai/models.py`
</read_first>
<action>
1. In `src/hipai/models.py`, import `Literal` from `typing` if not already present.
2. Update the `Relation` model:
   - Add a new field `tense: Literal["past", "present", "future"] = Field(default="present", description="The temporal context of the relation.")`
3. Update the `Observation` model:
   - Add a new field `tense: Literal["past", "present", "future"] = Field(default="present", description="The temporal context of the overall observation.")`
</action>
<acceptance_criteria>
- `src/hipai/models.py` contains `tense: Literal["past", "present", "future"]` in both `Relation` and `Observation`.
- Tests related to pydantic model instantiation still pass.
</acceptance_criteria>
</task>

<task>
<read_first>
- `src/hipai/world_model.py`
</read_first>
<action>
1. In `src/hipai/world_model.py`, locate `incorporate_observation`.
2. When creating relations (edges) in the graph, ensure the `tense` field from the `Relation` model is saved as a property on the edge.
   - Modify the `cypher` query for relations: `MERGE (source)-[r:{rel_type}]->(target) SET r.tense = $tense`
   - Pass `rel.tense` in the parameters dict.
3. (Optional but recommended) Do the same for property assignment nodes, or store the observation tense as a top-level node property if needed.
</action>
<acceptance_criteria>
- `world_model.py`'s `incorporate_observation` uses the `tense` property when creating edges via Cypher.
</acceptance_criteria>
</task>

<task>
<read_first>
- `src/hipai/synthesis.py`
</read_first>
<action>
1. In `src/hipai/synthesis.py`'s `add_belief` method, update the string-matching logic to detect basic tense.
2. For Pattern 2 ("X is a Y"), also check for "X was a Y" and "X will be a Y".
   - If " was a ", set `tense="past"`.
   - If " will be a ", set `tense="future"`.
   - Update the `Observation` instantiation to include the detected tense.
3. Update Pattern 5 ("X is Y") similarly for "was" and "will be".
4. Update Pattern 8 (Relational verbs) to detect basic auxiliary verbs: "will [verb]", "did [verb]", "had [verb]". 
   - Extract the tense and assign it to both the `Observation` and the `Relation` objects.
</action>
<acceptance_criteria>
- `synthesis.py` contains checks for " was a " and " will be a ".
- `Observation` and `Relation` objects created in these conditions have the correct `tense` assigned.
</acceptance_criteria>
</task>

<task>
<read_first>
- `tests/test_synthesis.py`
</read_first>
<action>
1. In `tests/test_synthesis.py`, add a new test `test_tense_parsing`.
2. The test should call `manager.add_belief("Socrates was a man")` and verify that the resulting graph state contains the `past` tense metadata.
3. Test a future relational belief: `manager.add_belief("Alice will visit Bob")` and verify the `tense` property is `"future"`.
</action>
<acceptance_criteria>
- `tests/test_synthesis.py` contains `test_tense_parsing`.
- `pytest tests/test_synthesis.py` exits 0.
</acceptance_criteria>
</task>

## Verification
<must_haves>
- [ ] Models `Relation` and `Observation` explicitly define `tense`.
- [ ] The `incorporate_observation` function stores `tense` on graph edges.
- [ ] `add_belief` parses past and future verbs correctly.
- [ ] The test suite verifies tense attributes on graph extraction.
</must_haves>
