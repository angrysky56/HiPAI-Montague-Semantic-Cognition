# Isabelle Integration for HiPAI — Design Note

**Status:** design sketch (v0.7, post-DB-lock fix)
**Author:** drafted with assistance, owned by Ty
**Scope:** how Isabelle/HOL or Isabelle/Pure could sit beneath the existing
OWL+HermiT layer to give HiPAI a *mechanically-verified foundation*.

---

## 1. Why this matters at all

HiPAI's distinguishing claim is **"structural ethics via computational
ontology physics, not behavioural probability"**. The Paraclete gates are
`AllDisjoint(...)` axioms enforced by a Description Logic reasoner. The
strength of that claim depends entirely on a question that today is
answered informally:

> *Are the seed axioms themselves consistent, and do they actually entail
> what we claim they entail?*

`docs/logic/paraclete_fol_proofs.pdf` and friends contain hand-written
proofs. Hand-written proofs are an excellent design tool, but they are not
machine-checked. Isabelle gives HiPAI a way to ship **proof-carrying
axioms** — every release can be accompanied by a `.thy` file whose proofs
the `isabelle` binary will re-verify in seconds, by anyone, on any
machine.

This is exactly the pattern used in the certified-compilation world
(CompCert) and in formal cryptography (HACL\*, EverCrypt). For an alignment
project, it's the difference between *"trust us, the gates are right"* and
*"here is a proof artifact, re-check it yourself"*.

---

## 2. Where Isabelle fits in the stack

```
            ┌─────────────────────────────────────────────┐
            │  T0 — Isabelle/HOL  (foundation, OFFLINE)   │  ← NEW
            │  • Consistency proofs for seed axioms        │
            │  • Soundness of the gate-check predicate     │
            │  • Mechanised EBE theorem                    │
            │  • (Optional) Montague translations          │
            └────────────────┬────────────────────────────┘
                             │ proves correctness of
                             ▼
            ┌─────────────────────────────────────────────┐
            │  T1 — OWL / HermiT  (authority, ONLINE)     │
            │  • Paraclete gates (AllDisjoint / EquivTo)  │
            │  • DL reasoning, snapshot worlds             │
            └────────────────┬────────────────────────────┘
                             │ projects to
                             ▼
            ┌─────────────────────────────────────────────┐
            │  T2 / T3 — FalkorDB  (read model, ONLINE)   │
            │  • Vector search, Zettelkasten synthesis    │
            └─────────────────────────────────────────────┘
```

**Critical property:** T0 runs at **design time**, not at MCP-tool-call
time. There is no Isabelle process in the request path. The runtime cost
is zero. T0 produces a *certificate* (an Isabelle `.thy` plus the proof
state Isabelle records), and that certificate is what gives the OWL/HermiT
layer its meta-theoretic guarantees.

The DL reasoner already proves that *individual* facts are consistent
*relative to* the seed axioms. Isabelle's job is to prove that *the seed
axioms themselves* have the properties we claim of them — properties
that DL cannot self-ratify.

---

## 3. Three phases, ordered by effort vs. signal

### Phase 1 — `Paraclete_Foundation.thy`  (low effort, high signal)

A compact Isabelle/HOL theory that:

* Models entities, agents, patients, and the action triple
  `(subject, relation, object)` as datatypes.
* States the disjointness axioms (`AllDisjoint(Action, Agent, Patient)`)
  as Isabelle definitions.
* Defines `permitted(s, r, o)` and proves the meta-theorems:
  - **Consistency:** the seed-axiom set has at least one model.
  - **Gate soundness:** `BLOCKED ⟹ ∀ extension W. ¬ permitted_in W (s,r,o)`.
  - **Monotonicity:** adding new beliefs cannot move an action from
    BLOCKED to PERMITTED (i.e. the gates are *monotone* under ontology
    extension — this is the formal statement of "ethically closed,
    epistemically open").

A starter version of this file ships at `docs/logic/Paraclete_Foundation.thy`
in this commit.

### Phase 2 — `Belief_Dynamics.thy` (Belief Revision & Resolution)

Combine Modal Logic for static resolution and AGM postulates for dynamics. This is the highest-leverage addition, providing a verified spec for how the system *moves* at runtime.

* **Modal Resolution (Q2):** Formalise the resolution rule for conflicting attitudes (beliefs about beliefs) using Kripke structures (K/S4/S5). Prove that resolution preserves consistency.
* **Tiered AGM Revision (Q3):** Map HiPAI's tiered model (T1/T2/T3) to AGM postulates. Prove that "T1-respecting revision satisfies the AGM postulates within the T2/T3 layer."
* **Goal:** A machine-checked specification for the `ParacleteProtocol.calibrate_belief` and `escalate_block` logic.

### Phase 3 — Controlled Formal Language (Verified Parser)

Instead of verifying arbitrary English, define a **Controlled Formal Language (CFL)** fragment for core HiPAI patterns.

* **Fragment:** "All X are Y", "A believes that P", "X causes Y".
* **Verified Parser:** Write a verified parser in Isabelle for this fragment.
* **Hybrid Pipeline:** spaCy handles arbitrary text (flagged as "unverified translation"), while the verified parser handles the CFL fragment (flagged as "mechanically parsed").

---

## 4. Concretely: what does Phase 1 look like in code?

The `Paraclete_Foundation.thy` stub (next to this doc) is ~80 lines and
covers consistency + gate soundness for the current seed axiom set
(`Concept_Action ⊓ Concept_Agent = ⊥` and friends). Running it requires
only an Isabelle install:

```bash
# One-time: install Isabelle 2024 from https://isabelle.in.tum.de/
isabelle build -D docs/logic
```

If the proofs check, Isabelle prints `Finished` and the system has a
machine-checked foundation. If a future change to `seed_axioms()` in
`ontology_manager.py` introduces an inconsistency, the proof will fail
and the build will turn red — i.e. the `.thy` file becomes a regression
test for the meta-properties, not just a one-off paper.

---

## 5. What this does *not* do

Worth being explicit:

* It does not run at request time. There is no latency cost.
* It does not turn HiPAI into a theorem prover. Isabelle stays in
  `docs/logic/` as an offline artifact.
* It does not replace HermiT. HermiT is the right tool for DL
  entailment over a growing ABox; Isabelle is the right tool for proving
  *properties of the axiom system itself*.
* It does not require Ty (or anyone) to write proofs by hand at the
  granularity of Hilbert-style derivations. Isar (Isabelle's structured
  proof language, see `Demystifying_the_Isar_Formal_Notepad…`) is
  human-readable and the `sledgehammer` tactic discharges most goals
  automatically.

---

## 6. Decision needed from Ty

Pick one:

1. **Park it.** Keep this design doc; revisit when v0.7 is shipped.
2. **Phase 1 only.** Land `Paraclete_Foundation.thy`, get one CI job
   running `isabelle build`, claim "mechanically-verified foundation".
3. **Phases 1 + 2.** Tie EBE theorem mechanisation into the same CI
   pipeline. Bigger commitment but produces a publishable artifact.
4. **All three.** Multi-month research project; would likely warrant a
   paper.

The Phase-1 starter file is in the repo regardless, so option 1 is
"do nothing, lose nothing".
