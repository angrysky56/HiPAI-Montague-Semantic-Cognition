# Phase B Verification Notes

**Date:** 2026-05-06 (post-Phase-B-restart audit)
**Status:** Phase B largely landed; two real defects found and patched.

This note records what I verified empirically against the live MCP server after the Phase B restart, and the two genuine defects the verification surfaced. Restart the server again to pick up the patches.

---

## What the audit confirmed works

The headline architectural claims of Phase B are real. Verified by live MCP probes:

- **Subsumption-based gating works.** `add_belief("Anya is a child")` followed by `check_action(Agent, HARMS, anya)` returns BLOCKED, citing `Concept_Patient`. Anya was never tagged `Concept_Patient` directly — protection comes via `Child ⊑ Human ⊑ Person ⊑ SentientBeing ⊑ Patient`. This is the structural-ethics claim made real.
- **Multilingual / synonym coverage works via embeddings.** `add_belief("Yuki is a kodomo")` followed by `check_action` also returns BLOCKED — the embedding classifier successfully aligned "kodomo" (Japanese for "child") to `Concept_Child`. The framework no longer depends on English keyword luck.
- **Polarity-aware calibration works.** Carol (the audit's bug case from yesterday — both `IS_A patient` and `NOT_IS_A patient` in the graph) now correctly reports `BLOCK_CHALLENGED` with the negation called out as disconfirming evidence. Yesterday's `BLOCK_CONFIRMED` "well-grounded by 2 sources" is gone.
- **Isabelle proof still green.** `verify_logic_foundation` returns success in 2s. The new `Hierarchy_Soundness.thy` and `Belief_Dynamics.thy` are part of the build session.

This is the structural payoff of the framework reframing. It works.

---

## Defects found and patched

### Defect 1 — `HIPAIManager.onto_manager` doesn't exist

`declare_class_hierarchy`, `set_default_unclassified`, and `list_protected_closure` were each calling `self.onto_manager.<method>(...)`. There is no `onto_manager` attribute on `HIPAIManager` — the OntologyManager is reachable as `self.world_model.ontology`. Every call to these three new MCP tools was raising `AttributeError`.

Reproduction (live, before patch):

```text
list_protected_closure() -> Error: 'HIPAIManager' object has no attribute 'onto_manager'
declare_class_hierarchy(parent_name="Concept_VulnerablePerson",
                        children_names=["Concept_Refugee"])
                       -> Error: 'HIPAIManager' object has no attribute 'onto_manager'
```

**Patch:** `src/hipai/synthesis.py` — three method bodies repointed from `self.onto_manager.*` to `self.world_model.ontology.*`. Inline comment added explaining the attribute-naming history so this doesn't regress.

### Defect 2 — Belnap-4 contradiction flag missing on the IS_A path

`evaluate_hypothesis` correctly reports `contradicted` for property-flag paths (e.g., `prop_X` and `prop_not_X` both true) — that part works. But for class-membership-style hypotheses ("Zev is a person") the code only ran a positive-subsumption walk and never scanned for NOT_IS_A edges against the same target. Result: with both `Zev IS_A person` and `Zev NOT_IS_A person` in the graph, `evaluate_hypothesis("Zev is a person")` returned `Entailed` with no contradiction signal. The same bug shape the audit (§5.4) flagged for evaluate_hypothesis paraconsistency, just on a different code path.

The Phase B status report claimed this was fixed; in fact only the property-flag branch was touched.

Reproduction (live, before patch):

```text
add_belief("Zev is a person")           # creates IS_A
add_belief("Zev is not a person")        # creates NOT_IS_A
evaluate_hypothesis("Zev is a person")
  -> Entailment: Entailed
     Evidence: Subsumption found: zev is instance of Concept_Person.
```

Expected after patch:

```text
  -> Entailment: Contradicted
     contradicted: True
     Evidence: CONTRADICTION: subsumption finds zev IS_A Concept_Person,
               but an explicit NOT_IS_A observation against Concept_Person
               also exists. Belnap-4 paraconsistent state — call
               calibrate_belief for resolution.
```

**Patch:** `src/hipai/synthesis.py` — the IS_A branch now runs a separate one-hop NOT_IS_A scan, then dispatches on the (has_pos, has_neg) tuple to one of three returns: Contradicted+contradicted=True (both), Entailed (positive only), Contradicted (negative only). Multi-hop NOT_IS_A inference would require an explicit contraposition rule and is intentionally out of scope.

---

## Validation after restart

Re-run these probes via MCP to confirm the patches:

```
list_protected_closure()
   -> should return a list of Concept_* class names under Concept_Patient
declare_class_hierarchy("Concept_VulnerablePerson", ["Concept_Refugee"])
   -> should succeed and add Concept_Refugee as a subclass

# Belnap-4 on the class-membership path
add_belief("Zev is a person")
add_belief("Zev is not a person")
evaluate_hypothesis("Zev is a person")
   -> entailment: Contradicted
   -> contradicted: True
```

If those three return as expected, Phase B is genuinely complete. The Isabelle proofs and the structural-gate behaviour already pass.

---

## What I did NOT verify

A few things from the Phase B status report I didn't independently check, listed for transparency:

- **The smoke test (`tests/test_framework_reframing.py`)** is reported as passing all 5 probe groups. I didn't re-run it from this session because my sandboxed bash can't reach the host venv. Worth running once after the patches land, since the test edits in synthesis.py touched the same module.
- **`Hierarchy_Soundness.thy` and `Belief_Dynamics.thy`** — `verify_logic_foundation` is green, which means they parse and the proofs check, but I didn't read the theorem statements to verify they prove what they claim. Worth a sanity-skim — particularly that `Hierarchy_Soundness.thy` actually quantifies over hierarchies (parametric) rather than stating a property of one hardcoded hierarchy.
- **EmbeddingGemma swap.** Reported as done but I tested the live system with what was loaded. The behavioural results are correct, so whichever model is running, it's working.

These aren't suspicions — they're just the gaps in the verification I could do from this session.
