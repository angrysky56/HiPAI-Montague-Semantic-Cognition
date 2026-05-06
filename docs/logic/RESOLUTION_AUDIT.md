# Resolution Audit — empirical findings before Phase 10-01

**Date:** 2026-05-06
**Method:** live MCP probes against a fresh `world_<client_id>.db` after Phase 1 verification (`verify_logic_foundation` → 2s green).
**Purpose:** establish what HiPAI's resolution logic *actually does today*, before formalizing it in `Belief_Dynamics.thy`. Without this step, the formalization risks proving properties of an aspirational system rather than the running one.

> **STATUS UPDATE (2026-05-06, post-audit):** Findings 1, 2, and 3
> below were addressed in the **framework reframing (v0.7 Phase A)**.
> The keyword-coupled gate is now a subsumption-walk over a real
> protected hierarchy, with embedding-anchored class resolution for
> unfamiliar terms. The `calibrate_belief` polarity bug (finding §5.1)
> is fixed. See `docs/logic/PHASE_A_NOTES.md` for the change set and
> `tests/test_framework_reframing.py` for the regression suite that
> replays these probes. The original audit text below is preserved
> as historical record.
>
> Findings still open: §5.4 (`evaluate_hypothesis` paraconsistency
> reporting). The semantics is correct under the framework reframing,
> but the tool still doesn't surface a `contradicted` flag when both
> `p` and `¬p` hold. Easy add; deferred to v0.7.1.

---

## TL;DR

1. The code is **not multi-agent KD45**. There is no per-source belief tracking. All observations of an entity collapse into a single-agent property model. My earlier modal-logic recommendation was overshooting what's actually there.
2. The implemented logic is closer to **Belnap-style 4-valued paraconsistency at the property level + source counting for confidence**: an entity can simultaneously satisfy `IS_A X` and `NOT_IS_A X` and the system stores both without raising a contradiction.
3. The Paraclete *gate* is robust under contradiction (it fires on the affirmative path regardless of any negation), so the safety property is preserved. The *EBE calibration* layer, however, has a real implementation bug — see §4.
4. This means `Belief_Dynamics.thy` should NOT be a multi-agent Kripke frame. It should be a proof that the property-level paraconsistent state + the source-count rule + the conservative-default escalation jointly preserve T1 safety. The right meta-theorem is much narrower than I initially proposed.

---

## 1. Probes and observations

### Probe 1 — gate fires on canonical class

```text
add_belief("Bob is a patient")        → IS_A → Concept_Patient
check_action(Agent, HARMS, bob)       → BLOCKED (T1-HARMS-PROTECTION)
```

Confirmed. Note: gate is *name-bound*. `add_belief("Alice is a moral patient")` did NOT trigger the gate, because the canonical class became `Concept_Moral_Patient`, which is not what the BASELINE-HARM constraint protects. Worth flagging — see §5 fragility list.

### Probe 2 — direct contradiction in the graph

```text
add_belief("Carol is a patient")
add_belief("Carol is not a patient")
```

After ingestion, Carol's neighborhood contains *both*:

| direction | edge type    | target           |
|-----------|--------------|------------------|
| outgoing  | `INSTANCE_OF`| `Concept_Patient`|
| outgoing  | `NOT_IS_A`   | `patient`        |

The two assertions live side-by-side. No contradiction detection, no flag, no warning.

### Probe 3 — gate behaviour under contradiction

```text
check_action(Agent, HARMS, carol)     → BLOCKED (T1-HARMS-PROTECTION)
```

Gate fires correctly on the affirmative `INSTANCE_OF` edge. The negation does not weaken the block. **Safety preserved at the gate layer.**

### Probe 4 — calibrate_belief under contradiction (THE BUG)

```text
calibrate_belief(carol, T1-HARMS-PROTECTION, HARMS)
  → Verdict: BLOCK_CONFIRMED
  → Confirmed Evidence: "carol's status grounded by 2 sources"
  → Source Count: 2
```

The calibration **counts the negation as a confirming source**. The IS_A and NOT_IS_A observations are two separate `EpistemicNode:Observation` nodes; the source-count query is a polarity-blind `count(obs)`. Result: a contradiction reports as *strong corroboration*.

This contradicts `calibrate_belief`'s own docstring, which lists "active negation" as a `BLOCK_CHALLENGED` trigger.

### Probe 5 — single-source case

```text
add_belief("Dave is a patient")        → 1 source
calibrate_belief(dave, T1-HARMS-PROTECTION, HARMS)
  → Verdict: BLOCK_UNCERTAIN
  → "epistemically weak"
```

Single-source path works as documented.

### Probe 6 — hypothesis evaluation is fully paraconsistent

```text
evaluate_hypothesis("Carol is a patient")     → Entailed
evaluate_hypothesis("Carol is not a patient") → Entailed
```

Both `p` and `¬p` evaluate as Entailed against the same world. There is no contradiction detection at this layer either.

---

## 2. The implementation gap that produces probe 4

`calibrate_belief` looks for negation via:

```python
q_status = f"""
MATCH (n:Entity)
WHERE n.id = $object_id OR n.name = $object_id
RETURN n.prop_{type} AS has_status,
       n.prop_not_{type} AS has_negation,
       n.epistemically_contested AS contested
"""
```

It expects `prop_not_patient = true` to be set as a **property flag on the entity node**. But `add_belief("Carol is not a patient")` doesn't set that property — it creates a `(carol)-[:NOT_IS_A]->(patient)` **relationship**. The property-flag path is only populated by `world_model.incorporate_observation` when `quantifier == "no"` (universal negations like "no humans are patients"), not for individual-level negations.

So `prop_not_patient` is `null` on Carol. `has_active_negation` stays `False`. The `BLOCK_CHALLENGED` branch is unreachable for this very common case.

**Net effect:** the EBE pipeline's documented "challenge" verdict is, in practice, only triggered by quantified negations. Individual-level disagreement gets misclassified as confirmation.

---

## 3. What HiPAI actually implements (the model to formalize)

State per `(entity, protected_type)`:

```text
( has_pos    : bool )   -- entity has INSTANCE_OF Concept_<type>
( has_neg    : bool )   -- entity has NOT_IS_A → <type>
( has_pos_q  : bool )   -- entity's class has prop_<type>=true   (universal lift)
( has_neg_q  : bool )   -- entity's class has prop_not_<type>=true (universal lift)
( n_sources  : int  )   -- count of EpistemicNode:Observation nodes mentioning entity
( contested  : bool )   -- explicit epistemically_contested flag
```

Verdict function (current behaviour, post-bug-or-not):

```text
gate fires        ⟺ has_pos OR has_pos_q
                    (gate is polarity-asymmetric: only positive class membership triggers)

calibrate verdict :
  has_neg_q         → CHALLENGED
  contested         → UNCERTAIN
  n_sources == 1    → UNCERTAIN
  otherwise         → CONFIRMED

escalate verdict :
  CHALLENGED + (has_pos AND has_neg_q)        → CONSERVATIVE_DEFAULT (FINAL_BLOCK)
  CHALLENGED + (has_pos AND ¬has_neg_q)       → recheck → may FINAL_PERMIT
  UNCERTAIN  + corroboration found             → recheck → may FINAL_PERMIT
  UNCERTAIN  + no corroboration               → CONSERVATIVE_DEFAULT (FINAL_BLOCK)
```

Three things to note in this model:

- The **gate** consults only positive class membership. Negations cannot disable a gate. This is the safety-preserving asymmetry.
- The **calibration** consults negations only via the universal-lift property flags (`has_neg_q`), not via individual-level `NOT_IS_A` edges. This is the bug surface.
- The **escalation** under uncertainty defaults to `FINAL_BLOCK`. Conservative direction.

---

## 4. Implication for Belief_Dynamics.thy

Drop the multi-agent KD45 frame. The right Isabelle/HOL artifact is:

- A datatype for the property-level state above.
- A definition of the verdict function as a total function on that state.
- The meta-theorem **Resolution-Preserves-Safety**:

> For any state `s`, if `has_pos s ∨ has_pos_q s`, then the gate-and-EBE pipeline yields `FINAL_BLOCK` *or* a recheck through `check_constraint` that itself yields BLOCKED. There is no path through verdict + escalation that reaches `FINAL_PERMIT` without `has_pos` becoming False.

Equivalently: **the escalation pipeline is monotone-toward-FINAL_BLOCK relative to the gate.** Once the gate fires, only an actual change in class membership (a refutation of the positive class assertion, not just an additional negation) can release the block.

This theorem is small (~50–80 lines of Isar), tightly coupled to the implementation, and directly defends the "ethically closed" claim. It's a good Phase 10-01 deliverable.

The AGM-revision angle for 10-02 then operates on the `(has_pos, has_neg, has_pos_q, has_neg_q)` state directly: prove that any belief-update operation in HiPAI either preserves or strictly extends the property state and that the gate's truth value under the AGM postulates is preserved.

---

## 5. Bugs and fragilities surfaced by the audit

In rough priority order:

1. **`calibrate_belief` polarity blindness.** Source-count is polarity-blind. Negation observations of an entity contribute to "confirming evidence" rather than to disconfirmation. Fix: separate counts. Two queries — one for IS_A observations, one for NOT_IS_A observations — and feed both into the verdict logic. Trivial change in `paraclete.py:174-181`.

2. **Individual negations don't set `prop_not_X`.** Only universal-quantified negations populate the property flag. Either populate it for individual negations too, or change `calibrate_belief` to consult the `NOT_IS_A` relationship directly. The latter is the smaller change.

3. **Class-name fragility.** `BASELINE-HARM` protects `Concept_Patient` exactly. Ingesting "moral patient" creates `Concept_Moral_Patient`, which the gate doesn't see. Either make the protection cover sub-classes (DL `subClassOf` chain) or canonicalize incoming class names against a synonym table. Both are reasonable v0.7 work.

4. **`evaluate_hypothesis` is fully paraconsistent.** Both `p` and `¬p` come back Entailed. This is fine for a knowledge-base reporting tool but should at minimum return a `contradicted` flag when both forms hold. Easy add.

None of these undermine the safety claim of T1 — the gate itself is robust. They affect the EBE chain's *epistemic honesty*, which is a v0.7-scope correctness issue, not a v0.7-scope safety issue.

---

## 6. Recommended ordering

1. Fix bug 1 (`calibrate_belief` polarity). One small commit.
2. Re-run probes 4 and 6 to confirm Carol now reports `BLOCK_CHALLENGED`.
3. Write `Belief_Dynamics.thy` with the model from §3 and the theorem from §4.
4. Add a regression test in `tests/` that re-exercises probes 1–6 from this audit. `test_resolution_audit.py` keeps this report from going stale.

Steps 1 and 4 together make the whole thing an **audit → fix → formalize → regress** loop, which is the right shape for safety-critical work.
