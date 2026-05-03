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
        root = sent.root
        
        # 1. Find Subject
        subjects = [t for t in sent if t.dep_ == "nsubj"]
        if not subjects:
            return
            
        subj_token = subjects[0]
        subject_id = subj_token.text.lower()
        
        # Extract Quantifier
        quantifier = self._extract_quantifier(subj_token)
        
        # Add subject as individual if not exists
        if not any(ind.name.lower() == subject_id for ind in observation.individuals):
            # Use name.lower() as ID for consistency
            ind_id = subj_token.text.lower().replace(" ", "_")
            observation.individuals.append(Individual(id=ind_id, name=subj_token.text))

        # 2. Handle Predicate based on ROOT type
        if root.pos_ == "AUX" or (root.pos_ == "VERB" and root.lemma_ == "be"):
            # Attribution: "Socrates is a man" or "Socrates is mortal"
            self._handle_attribution(root, subj_token, observation, quantifier)
        elif root.pos_ == "VERB":
            # Action: "Socrates harms the Pig"
            self._handle_action(root, subj_token, observation, quantifier)

    def _extract_quantifier(self, subj_token) -> Optional[str]:
        """
        Extract quantifier (all, some, no) from the subject's determiner.
        """
        for child in subj_token.children:
            if child.dep_ == "det":
                det_text = child.text.lower()
                if det_text in ("all", "every", "each"):
                    return "all"
                if det_text in ("some", "a", "an"):
                    return "some"
                if det_text in ("no", "none"):
                    return "no"
        return None

    def _handle_attribution(self, root, subj_token, observation: Observation, quantifier: Optional[str] = None):
        """
        Handle 'is a' or 'is [adj]' patterns.
        """
        # Look for attr (noun) or acomp (adj)
        attrs = [t for t in root.children if t.dep_ in ("attr", "acomp")]
        if not attrs:
            return
            
        attr_token = attrs[0]
        
        # Check for negation
        is_negated = any(t.dep_ == "neg" for t in root.children) or (quantifier == "no")
        
        subject = next(ind for ind in observation.individuals if ind.name.lower() == subj_token.text.lower())
        
        if attr_token.pos_ == "NOUN":
            # "Socrates is a man" -> Add 'man' to properties
            prop_name = attr_token.text
            if is_negated:
                prop_name = f"not_{prop_name}"
            
            if isinstance(subject.properties, list):
                subject.properties.append(prop_name)
        elif attr_token.pos_ == "ADJ":
            # "Socrates is mortal" -> Add 'mortal' to properties
            prop_name = attr_token.text
            if is_negated:
                prop_name = f"not_{prop_name}"
            
            if isinstance(subject.properties, list):
                subject.properties.append(prop_name)

    def _handle_action(self, root, subj_token, observation: Observation, quantifier: Optional[str] = None):
        """
        Handle 'S [verb] O' patterns.
        """
        # Look for dobj (direct object) or other object-like deps
        objs = [t for t in root.children if t.dep_ in ("dobj", "obj", "npadvmod")]
        if not objs:
            return
            
        obj_token = objs[0]
        obj_id = obj_token.text.lower()
        
        # Add object as individual if not exists
        if not any(ind.name.lower() == obj_id for ind in observation.individuals):
            ind_id = obj_token.text.lower().replace(" ", "_")
            observation.individuals.append(Individual(id=ind_id, name=obj_token.text))
            
        # Check for negation
        is_negated = any(t.dep_ == "neg" for t in root.children) or (quantifier == "no")
        rel_type = root.lemma_.upper()
        if is_negated:
            rel_type = f"NOT_{rel_type}"
            
        # Create relation
        subj_ind = next(ind for ind in observation.individuals if ind.name.lower() == subj_token.text.lower())
        obj_ind = next(ind for ind in observation.individuals if ind.name.lower() == obj_id)
        
        tense = self._detect_tense(root)
        modality = self._detect_modality(root)
        
        relation = Relation(
            source_id=subj_ind.id,
            target_id=obj_ind.id,
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
