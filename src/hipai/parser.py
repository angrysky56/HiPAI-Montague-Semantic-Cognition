"""
Linguistic parsing engine for HiPAI-Montague.

Uses spaCy dependency parsing to extract structured claims (Observations)
from natural language, replacing brittle regex patterns.
"""

import spacy
from typing import Optional
from .models import Observation, Individual, Relation

class ClaimExtractor:
    def __init__(self, model: str = "en_core_web_md"):
        """
        Initialize the ClaimExtractor with a spaCy model.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Fallback or alert if model not found
            raise ImportError(f"spaCy model {model} not found. Please run 'uv run python -m spacy download {model}'")

    def extract(self, text: str) -> Observation:
        """
        Extract an Observation from a natural language sentence.
        """
        doc = self.nlp(text)
        observation = Observation(text_source=text)
        
        # We assume one primary claim per sentence for now
        # In the future, we can split by sentence or handle multiple ROOTs
        
        for sent in doc.sents:
            self._process_sentence(sent, observation)
            
        return observation

    def _process_sentence(self, sent, observation: Observation):
        """
        Process a single sentence and update the observation.
        """
        try:
            self._populate_observation(sent.root, observation)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def _get_full_name_and_id(self, token):
        """Extract full name and generated ID for a token including compounds."""
        # Get all tokens in the subtree that are part of the name
        name_tokens = [t for t in token.subtree if t.dep_ in ("compound", "flat", "nsubj", "dobj", "obj", "pobj")]
        # Ensure we at least include the token itself if it's not in subtree (rare)
        if token not in name_tokens:
            name_tokens.append(token)
        name_tokens.sort(key=lambda t: t.i)
        
        name = " ".join([t.text for t in name_tokens])
        id = name.lower().replace(" ", "_")
        return name, id

    def _populate_observation(self, root, observation: Observation):
        """
        Populate an observation from a root token and its subtree.
        """
        # 1. Find Subject
        # We look for nsubj in the children of the current root
        subjects = [t for t in root.children if t.dep_ == "nsubj"]
        if not subjects:
            return
            
        subj_token = subjects[0]
        subj_name, subject_id = self._get_full_name_and_id(subj_token)
        
        # Extract Quantifier
        quantifier = self._extract_quantifier(subj_token)
        
        # Add subject as individual if not exists
        if not any(ind.id == subject_id for ind in observation.individuals):
            observation.individuals.append(Individual(id=subject_id, name=subj_name))

        # 2. Handle Predicate based on ROOT type
        if root.pos_ == "AUX" or (root.pos_ == "VERB" and root.lemma_ == "be"):
            self._handle_attribution(root, subject_id, observation, quantifier)
        elif root.pos_ == "VERB":
            self._handle_action(root, subject_id, observation, quantifier)

    def _extract_quantifier(self, subj_token) -> Optional[str]:
        """
        Extract quantifier (all, some, no) from the subject's determiner.
        """
        for child in subj_token.children:
            if child.dep_ == "det":
                text = child.text.lower()
                if text in ("all", "every", "each"):
                    return "all"
                if text in ("some", "a", "an", "the"):
                    return "some"
                if text in ("no", "none"):
                    return "no"
        return None

    def _handle_attribution(self, root, subject_id: str, observation: Observation, quantifier: Optional[str] = None):
        """
        Handle 'S is A' or 'S is a C' patterns.
        """
        # Look for attr (noun) or acomp (adj)
        attrs = [t for t in root.children if t.dep_ in ("attr", "acomp")]
        if not attrs:
            return
            
        attr_token = attrs[0]
        
        # Check for negation
        is_negated = any(t.dep_ == "neg" for t in root.children) or (quantifier == "no")
        
        # Get individual
        subject = next(ind for ind in observation.individuals if ind.id == subject_id)
        
        if attr_token.pos_ == "NOUN":
            prop_name = attr_token.text
            if is_negated:
                prop_name = f"not_{prop_name}"
            subject.properties.append(prop_name)
        elif attr_token.pos_ == "ADJ":
            prop_name = attr_token.text
            if is_negated:
                prop_name = f"not_{prop_name}"
            subject.properties.append(prop_name)

    def _handle_action(self, root, subject_id: str, observation: Observation, quantifier: Optional[str] = None):
        """
        Handle 'S [verb] O' patterns.
        """
        objs = [t for t in root.children if t.dep_ in ("dobj", "obj", "npadvmod", "pobj")]
        ccomps = [t for t in root.children if t.dep_ == "ccomp"]
        
        if not objs and not ccomps:
            return
            
        subj_ind = next(ind for ind in observation.individuals if ind.id == subject_id)
        
        is_negated = any(t.dep_ == "neg" for t in root.children) or (quantifier == "no")
        rel_type = root.lemma_.upper()
        if is_negated:
            rel_type = f"NOT_{rel_type}"
            
        tense = self._detect_tense(root)
        modality = self._detect_modality(root)
        
        # Handle standard objects
        for obj_token in objs:
            obj_name, obj_id = self._get_full_name_and_id(obj_token)
            if not any(ind.id == obj_id for ind in observation.individuals):
                observation.individuals.append(Individual(id=obj_id, name=obj_name))
            
            observation.relations.append(Relation(
                source_id=subject_id,
                target_id=obj_id,
                relation_type=rel_type,
                tense=tense,
                modality=modality
            ))

        # Handle clausal complements (attitudes)
        for cc_token in ccomps:
            # Extract nested observation
            subtree_text = " ".join([t.text for t in cc_token.subtree])
            inner_obs = Observation(text_source=subtree_text)
            self._populate_observation(cc_token, inner_obs)
            
            relation = Relation(
                source_id=subject_id,
                target_observation=inner_obs,
                relation_type=rel_type,
                tense=tense,
                modality=modality
            )
            observation.relations.append(relation)
        
        observation.tense = tense
        observation.modality = modality

    def _detect_tense(self, root) -> str:
        """Detect tense from the root verb/aux."""
        if any(t.text.lower() in ("will", "shall") for t in root.children):
            return "future"
        if any(t.text.lower() in ("was", "were", "did", "had") for t in root.children) or root.tag_ in ("VBD", "VBN"):
            return "past"
        return "present"

    def _detect_modality(self, root) -> Optional[str]:
        """Detect modality (must, can, should) from the root's children."""
        for t in root.children:
            if t.dep_ == "aux" and t.text.lower() in ("must", "can", "should", "may"):
                return t.text.lower()
        return None
