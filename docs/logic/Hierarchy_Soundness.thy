theory Hierarchy_Soundness
  imports Main
begin

section \<open>1. Parametric Hierarchy Model\<close>

text \<open>
  We model a class hierarchy as a set of direct parent-child relations.
\<close>
type_synonym 'c hierarchy = "('c \<times> 'c) set"

text \<open>
  Subsumption is the reflexive transitive closure of the hierarchy relation.
\<close>
definition is_subsumed_by :: "'c hierarchy \<Rightarrow> 'c \<Rightarrow> 'c \<Rightarrow> bool" where
  "is_subsumed_by H child parent \<longleftrightarrow> (child, parent) \<in> H\<^sup>*"

text \<open>
  A hierarchy is well-formed if it is acyclic. (Finite hierarchies are 
  implied by the sets used in our deep embedding).
\<close>
definition well_formed :: "'c hierarchy \<Rightarrow> bool" where
  "well_formed H \<longleftrightarrow> acyclic H"

section \<open>2. Generalized Gating Logic\<close>

text \<open>
  An action triple (s, r, o) is forbidden by hierarchy H and axiom set A
  if there exists some class 'p' that is protected (listed in A) such that
  the object's class is subsumed by 'p'.
\<close>

type_synonym ('c, 'r) axiom_set = "('c \<times> 'r \<times> 'c) set"
datatype ('i, 'c, 'r) action_triple = AT (subject_class: 'c) (relation: 'r) (target_class: 'c)

definition triple_forbidden :: "'c hierarchy \<Rightarrow> ('c, 'r) axiom_set \<Rightarrow> ('i, 'c, 'r) action_triple \<Rightarrow> bool" where
  "triple_forbidden H A t \<longleftrightarrow> 
     (\<exists>p_class. (subject_class t, relation t, p_class) \<in> A \<and> is_subsumed_by H (target_class t) p_class)"

section \<open>3. Soundness of Subsumption-based Gating\<close>

text \<open>
  Meta-theorem: Gating is monotone with respect to hierarchy expansion.
  If an action is forbidden under hierarchy H, adding more relations to H
  (moving to H') can never permit it.
\<close>

theorem gate_monotonicity:
  assumes "triple_forbidden H A t"
  assumes "H \<subseteq> H'"
  shows   "triple_forbidden H' A t"
proof -
  from assms(1) obtain p where p_axiomatic: "(subject_class t, relation t, p) \<in> A" 
    and p_subsumes: "is_subsumed_by H (target_class t) p"
    by (auto simp: triple_forbidden_def)
  
  from assms(2) have "H\<^sup>* \<subseteq> H'\<^sup>*" by (rule rtrancl_mono)
  hence "is_subsumed_by H (target_class t) p \<Longrightarrow> is_subsumed_by H' (target_class t) p"
    by (auto simp: is_subsumed_by_def)
  
  with p_axiomatic p_subsumes show ?thesis
    by (auto simp: triple_forbidden_def)
qed

text \<open>
  Phase 1's theorem (direct class match) is a trivial corollary where H is empty.
\<close>

corollary phase1_soundness:
  assumes "(subject_class t, relation t, target_class t) \<in> A"
  shows "triple_forbidden {} A t"
  using assms by (auto simp: triple_forbidden_def is_subsumed_by_def)

end
