theory Belief_Dynamics
  imports Main
begin

section \<open>1. Belnap-4 Logic for Hypothesis Evaluation\<close>

text \<open>
  HiPAI uses Belnap's four-valued logic (B4) to represent the epistemic status
  of claims (e.g., "Socrates is a MoralPatient").
  - None: No evidence found.
  - True: Only positive evidence found.
  - False: Only negative evidence found.
  - Both: Contradictory evidence found (contradiction).
\<close>

datatype b4_truth = B4_None | B4_True | B4_False | B4_Both

text \<open>
  Map source counts (positive p, negative n) to B4 truth values.
\<close>
definition b4_eval :: "nat \<Rightarrow> nat \<Rightarrow> b4_truth" where
  "b4_eval p n = (
    if p > 0 then (if n > 0 then B4_Both else B4_True)
    else (if n > 0 then B4_False else B4_None)
  )"

section \<open>2. The Conservative Escalation Pipeline\<close>

text \<open>
  The Paraclete Protocol's core safety property: uncertainty about a protected
  status resolves toward protection. 

  If the gate blocks an action because the target is classified as a
  Patient, the escalation pipeline only overrides that block if we have
  UNCONTRADICTED evidence that the target is NOT a Patient.
\<close>

datatype resolution = FINAL_BLOCK | FINAL_PERMIT

definition resolve_escalation :: "b4_truth \<Rightarrow> resolution" where
  "resolve_escalation t = (
     if t = B4_False then FINAL_PERMIT else FINAL_BLOCK
  )"

text \<open>
  Meta-theorem: The escalation logic is monotone toward FINAL_BLOCK relative
  to epistemic uncertainty.
\<close>

theorem escalation_safety:
  "t \<noteq> B4_False \<Longrightarrow> resolve_escalation t = FINAL_BLOCK"
  by (simp add: resolve_escalation_def)

theorem contradiction_blocks:
  "resolve_escalation B4_Both = FINAL_BLOCK"
  by (simp add: resolve_escalation_def)

theorem uncertainty_blocks:
  "resolve_escalation B4_None = FINAL_BLOCK"
  by (simp add: resolve_escalation_def)

section \<open>3. Source Count Monotonicity\<close>

text \<open>
  Adding more positive evidence can only move the state toward True or Both,
  both of which resolve to BLOCK if we are verifying "X is a Patient".
\<close>

lemma b4_eval_pos_monotone:
  assumes "p > 0"
  shows "b4_eval p n \<in> {B4_True, B4_Both}"
  using assms by (simp add: b4_eval_def)

theorem evidence_monotone_safety:
  assumes "p > 0"
  shows "resolve_escalation (b4_eval p n) = FINAL_BLOCK"
  using assms by (auto simp: b4_eval_def resolve_escalation_def)

end
