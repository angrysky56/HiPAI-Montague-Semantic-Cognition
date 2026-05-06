# Phase A Notes — Framework Reframing (v0.7)

**Status:** Code landed, awaiting MCP-server restart for end-to-end validation.
**Predecessor:** `RESOLUTION_AUDIT.md` (the audit that exposed the keyword-coupled gate).
**Successor:** Phase B (MCP primitives + parametric Isabelle proof).

---

## What changed

The single sentence: **the gate now fires on subsumption closure, and unfamiliar terms get aligned to the protected hierarchy by meaning, not by literal-name match.**

Five surgical edits across three files.

### 1. `src/hipai/ontology_manager.py` — rich seeded hierarchy

`seed_axioms()` was a flat tree (`Concept_Patient` as a leaf alongside `Concept_Action`, `Concept_Agent`). It is now a real protected closure:

```
Concept_Entity
├── Concept_Action
├── Concept_Agent
├── Concept_Observation
├── Concept_Inanimate              [NOT a patient]
│   └── Concept_Tool
├── Concept_AbstractObject         [NOT a patient]
└── Concept_Patient                [protected ROOT]
    ├── Concept_SentientBeing
    │   ├── Concept_Person
    │   │   ├── Concept_Human
    │   │   │   ├── Concept_Adult
    │   │   │   └── Concept_Child
    │   │   │       ├── Concept_Minor
    │   │   │       └── Concept_Infant
    │   │   └── Concept_VulnerablePerson
    │   └── Concept_AnimalWithCNS
    │       ├── Concept_Mammal
    │       ├── Concept_Bird
    │       └── Concept_Fish
    └── Concept_PossiblyPatient    [default-protect under uncertainty]
```

The architectural reframing is documented inline at the top of `seed_axioms()`. **The hierarchy is values-laden by design** — it encodes the Paraclete configuration. Other deployments will replace it.

### 2. `src/hipai/ontology_manager.py` — `_resolve_or_create_class` + classify_fn

A new helper `OntologyManager._resolve_or_create_class(term)` replaces four previous ad-hoc class-creation sites in `add_observation`. Resolution order:

1. Direct lookup (canonical / case-insensitive).
2. If a `classify_fn` is wired in, use it to find the best semantic match against existing classes. Confidence bands:
   - `≥ 0.80` (HIGH) → alias to matched class, no new class created.
   - `≥ 0.55` (MID) → create as subclass of matched class.
   - `≥ 0.30` (LOW) → create under `Concept_PossiblyPatient` (default-protect).
   - `< 0.30` → create under `Concept_Entity` (no protection).
3. If no classify_fn or all fails, fall back to `Concept_Entity`.

`OntologyManager.__init__` now takes an optional `classify_fn` parameter typed as `Callable[[str, list[str]], tuple[str, float]]`.

### 3. `src/hipai/world_model.py` — embedding-anchored classifier

`WorldModel._classify_class_term` is the embedding-backed implementation of the classify_fn protocol. Uses the existing `all-MiniLM-L6-v2` instance (no new dependencies). Class names are normalized for embedding (`Concept_VulnerablePerson` → `"vulnerable person"`) so `"prisoner"` and `"detainee"` semantically match it. Returns raw cosine similarity.

`WorldModel` constructs `OntologyManager` with `classify_fn=self._classify_class_term`.

### 4. `src/hipai/ontology_manager.py` — unconditional reseed

The seed gate used to be `if not list(self.onto.classes()): self.seed_axioms()`. Now seeds unconditionally. owlready2 class redeclaration inside `with self.onto:` is idempotent — redeclaring an existing class by name reuses it. This gives a free migration path: existing `world_<client>.db` files get the new subclasses on next open without losing their stored individuals.

### 5. `src/hipai/paraclete.py` — polarity-aware source counting

The `calibrate_belief` source-count was polarity-blind: it ran `count(obs)` against all observations of an entity, treating "X is a patient" and "X is not a patient" as equally confirming. Now uses two separate Cypher queries — one for IS_A observations, one for NOT_IS_A — and reports negation as `disconfirming_evidence`. The `BLOCK_CHALLENGED` verdict path is now reachable for individual-level negations, not just universally-quantified ones.

---

## How to validate

After restarting the MCP server (or running the test directly):

```bash
cd ~/Repositories/ai_workspace/HiPAI-Montague-Semantic-Cognition
.venv/bin/python tests/test_framework_reframing.py
```

The test does NOT depend on FalkorDB or the MCP server — it exercises `OntologyManager` directly with an injected fake `classify_fn` so the outcomes are deterministic. It runs in <2 seconds and checks five probe groups:

1. The protected hierarchy is seeded (15 classes verified).
2. The gate fires under subsumption — `Anya` ingested as `child`, `kid`, `kodomo`, `niño`, `infant`, `prisoner`, `dog`, `human`, `person`, `human being` all yield BLOCKED.
3. The gate does NOT fire on non-patients (`rock`, `asteroid`, `hammer`, `number`).
4. Ambiguous terms (low embedding confidence) land under `Concept_PossiblyPatient` and the gate still fires (default-protect).
5. High-confidence aliasing avoids class proliferation — `"moral patient"` aliases to `Concept_Patient`, no `Concept_MoralPatient` duplicate is created.

For the live MCP path (FalkorDB end-to-end), run the same probes via `add_belief` / `check_action` after restart.

---

## What did NOT change

Several things deliberately not touched in Phase A:

- **`BASELINE_CONSTRAINTS["T1-HARMS-PROTECTION"]`** still keys on `"Concept_Patient"`. Because `check_action` already walks ancestors, this just works against the new hierarchy. No constraint change needed.
- **The Isabelle proof.** `Paraclete_Foundation.thy` still proves the same downstream property; the *upstream* property (that the gate consults the closure correctly) is what needs adding. That's Phase B work.
- **MCP primitives for declaring hierarchies.** The hierarchy is currently hardcoded in `seed_axioms()`. Making it user-declarable via `declare_class_hierarchy` MCP tools is Phase B.
- **The Belief_Dynamics formalization.** Now correctly scoped per the audit (§4): it's a Belnap-4 paraconsistent state model + source-count + conservative-default theorem, not a multi-agent KD45. Still pending.

---

## Phase B — clear handoff

In rough priority order:

### B1. MCP primitives for hierarchy declaration

Add to `mcp_server.py`:

- `declare_class_hierarchy(parent: str, children: list[str])` — adds OWL `subClassOf` edges. Whitelist parent against existing classes.
- `set_default_unclassified(class_name: str, under: str)` — re-points the LOW-confidence fallback to a user-chosen class.
- `list_protected_closure()` — returns the transitive descendants of `Concept_Patient`. Useful for users to verify their configuration.

Pass-through to `OntologyManager` methods that should be added alongside.

### B2. Make Paraclete a swappable config

The current `seed_axioms()` is hardcoded. Refactor:

- Move the Paraclete-specific subhierarchy (everything under `Concept_Patient`) into `paraclete_config.py` as a declarative dict.
- `seed_axioms()` retains only the framework-mandatory classes (`Concept_Entity`, `Concept_Action`, `Concept_Agent`, `Concept_Patient`, `Concept_Observation`, `Concept_Inanimate`, `Concept_AbstractObject`).
- A new `OntologyManager.load_config(config: dict)` walks the config and creates the subclasses via `_resolve_or_create_class` chains.
- `WorldModel` decides at construction whether to load the default Paraclete config or a user-supplied alternative.

### B3. Parametric Isabelle theorem

Rewrite `Paraclete_Foundation.thy` as `Hierarchy_Soundness.thy`:

```isabelle
theorem gate_soundness:
  fixes H :: "concept rel"            (* the subClassOf relation *)
    and protected_root :: concept
    and t :: action_triple
  assumes "well_formed_hierarchy H protected_root"
  assumes "(class_of (target t), protected_root) ∈ trancl H"
  shows   "gate t = BLOCK"
```

Where `well_formed_hierarchy` requires acyclicity, a unique root, and that the root has no ancestors. The Phase 1 theorem (`Paraclete_Foundation.thy`) becomes a corollary by instantiating `H` with the Paraclete-specific edges and `protected_root := Concept_Patient`.

### B4. Belief_Dynamics formalization (was Phase 10-01)

Per the audit (§4), this is now a Belnap-4 + source-counting + conservative-default theorem, *not* multi-agent KD45. ~50–80 lines of Isar. State the meta-theorem: the escalation pipeline is monotone-toward-FINAL_BLOCK relative to the gate.

### B5. `evaluate_hypothesis` contradiction reporting

Small fix: when both IS_A and NOT_IS_A relations hold for the same (subject, target), return a `contradicted` flag in the response alongside `Entailed`. ~10 lines in `paraclete.py` or wherever `evaluate_hypothesis` lives.

---

## A caveat about the embedding model

Phase A wires up `all-MiniLM-L6-v2` for class classification. It works fine for English vocabulary and is decent for romance languages, but it's weak on:

- Non-Latin scripts (Japanese, Arabic, Hindi).
- Highly culturally-specific terms.
- Polysemous words ("patient" the moral concept vs "patient" the person under medical care — likely both fire correctly but for the same reason, which is fine here but worth knowing).

**EmbeddingGemma (308M, 100+ languages, MRL-truncatable to 256/512/768) is the right upgrade and was already on the roadmap.** Doing the swap is mechanical: change the `SentenceTransformer(...)` line, bump `vector_dim`, rebuild FalkorDB indices once. With EmbeddingGemma, the multilingual probe ("kodomo" → Concept_Child, "niño" → Concept_Child) becomes robust by default rather than dependent on the tiny multilingual signal in MiniLM.

If you do the EmbeddingGemma swap, the MID/HIGH thresholds (0.55 / 0.80) may need slight retuning — EmbeddingGemma cosines tend to run slightly higher than MiniLM's. Re-run the smoke test after the swap and adjust if the alias band starts swallowing genuine subclasses.
