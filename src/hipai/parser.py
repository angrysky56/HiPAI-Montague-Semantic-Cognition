import spacy

from .models import Individual, Observation, Relation


class ClaimExtractor:
    def __init__(self, model: str = "en_core_web_md"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import os

            os.system(f"python -m spacy download {model}")
            self.nlp = spacy.load(model)

    def extract(self, text: str) -> Observation:
        doc = self.nlp(text)
        observation = Observation(text_source=text)

        # 1. Identify Individuals and Basic Properties
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                self._process_subject_cluster(token, observation)

        # 2. Identify Relations
        for token in doc:
            if token.pos_ == "VERB" or (token.pos_ == "AUX" and token.dep_ == "ROOT"):
                self._process_verb_cluster(token, observation)

        return observation

    def _process_subject_cluster(self, token, observation: Observation):
        # Handle quantifiers
        quantifier = None
        for child in token.children:
            if child.dep_ == "det" and child.lemma_.lower() in (
                "all",
                "no",
                "every",
                "some",
                "any",
            ):
                quantifier = child.lemma_.lower()
            elif child.dep_ == "quantmod" or child.dep_ == "nummod":
                quantifier = child.text.lower()

        name, id = self._get_full_name_and_id(token)

        # Check if already exists
        if not any(ind.id == id for ind in observation.individuals):
            observation.individuals.append(
                Individual(name=name, id=id, quantifier=quantifier)
            )

    def _get_full_name_and_id(self, token) -> tuple[str, str]:
        # Collect compound parts
        name_tokens = [token]
        for child in token.children:
            if child.dep_ == "compound":
                name_tokens.append(child)

        name_tokens.sort(key=lambda t: t.i)

        text_name = " ".join([t.text for t in name_tokens])
        id = text_name.lower().replace(" ", "_")

        if token.pos_ == "NOUN":
            name = " ".join([t.lemma_ for t in name_tokens])
        else:
            name = text_name

        return name, id

    def _process_verb_cluster(self, root, observation: Observation):
        # Find subject
        subject_id = None
        quantifier = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                _, subject_id = self._get_full_name_and_id(child)
                # Check for quantifier on subject
                for grand in child.children:
                    if grand.dep_ == "det" and grand.lemma_.lower() in ("all", "no"):
                        quantifier = grand.lemma_.lower()
                break

        if not subject_id:
            return

        # Tense and Modality
        tense = "present"
        modality = None

        # Simple tense check
        if root.tag_ in ("VBD", "VBN"):
            tense = "past"
        elif any(t.lemma_ == "will" for t in root.children):
            tense = "future"

        # Modality check
        modal_tokens = [
            t
            for t in root.children
            if t.pos_ == "AUX" and t.lemma_ not in ("be", "have", "do", "will")
        ]
        if modal_tokens:
            modality = modal_tokens[0].lemma_

        # Set observation-level metadata
        observation.tense = tense
        observation.modality = modality or "assertive"
        observation.subject_id = subject_id

        # 1. Copular (is-a or property)
        if root.lemma_ == "be":
            self._handle_copula(
                root, subject_id, observation, quantifier, tense, modality
            )
        else:
            self._handle_action(
                root,
                subject_id,
                observation,
                quantifier,
                tense=tense,
                modality=modality,
            )

    def _handle_copula(
        self,
        root,
        subject_id: str,
        observation: Observation,
        quantifier: str | None = None,
        tense: str = "present",
        modality: str | None = None,
    ):
        """
        Handles 'Socrates is a man' or 'Dogs are animals' or 'The apple is red'.
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
            attr_name, attr_id = self._get_full_name_and_id(attr_token)
            rel_type = "IS_A"
            if is_negated:
                rel_type = "NOT_IS_A"

            # Ensure the attribute individual is also tracked
            if not any(ind.id == attr_id for ind in observation.individuals):
                observation.individuals.append(Individual(name=attr_name, id=attr_id))

            rel = Relation(
                source_id=subject_id,
                target_id=attr_id,
                relation_type=rel_type,
                tense=tense,
                modality=modality,
            )
            observation.relations.append(rel)

        elif attr_token.pos_ == "ADJ":
            prop_name = attr_token.lemma_
            if is_negated:
                prop_name = f"not_{prop_name}"
            subject.properties.append(prop_name)

    def _handle_action(
        self,
        root,
        subject_id: str,
        observation: Observation,
        quantifier: str | None = None,
        max_depth: int = 5,
        tense: str = "present",
        modality: str | None = None,
    ):
        """
        Handles 'Socrates drinks hemlock' or 'Dogs chase cats'.
        """
        # Check for negation
        is_negated = any(t.dep_ == "neg" for t in root.children) or (quantifier == "no")

        rel_type = root.lemma_.upper()
        if is_negated:
            rel_type = f"NOT_{rel_type}"

        # Find objects (dobj, prep/pobj, etc.)
        targets = []

        for child in root.children:
            if child.dep_ in ("dobj", "obj", "npadvmod"):
                targets.append(child)
            elif child.dep_ == "prep":
                for grand in child.children:
                    if grand.dep_ == "pobj":
                        targets.append(grand)

        for target_token in targets:
            target_name, target_id = self._get_full_name_and_id(target_token)

            # Ensure target individual exists
            if not any(ind.id == target_id for ind in observation.individuals):
                observation.individuals.append(
                    Individual(name=target_name, id=target_id)
                )

            observation.relations.append(
                Relation(
                    source_id=subject_id,
                    target_id=target_id,
                    relation_type=rel_type,
                    tense=tense,
                    modality=modality,
                )
            )

        # 3. Clausal Complements (Attitudes)
        ccomps = [t for t in root.children if t.dep_ == "ccomp"]
        for ccomp in ccomps:
            # Get text from subtree
            ccomp_text = " ".join([t.text for t in ccomp.subtree])
            # Remove 'that ' if it starts with it
            if ccomp_text.lower().startswith("that "):
                ccomp_text = ccomp_text[5:]

            # Recursively extract
            sub_obs = self.extract(ccomp_text)
            if sub_obs:
                # Merge sub-observation individuals into main observation
                for sub_ind in sub_obs.individuals:
                    if not any(ind.id == sub_ind.id for ind in observation.individuals):
                        observation.individuals.append(sub_ind)

                # Factivity check
                factive_verbs = ("know", "realize", "regret", "understand")
                is_factive_rel = root.lemma_ in factive_verbs

                observation.relations.append(
                    Relation(
                        source_id=subject_id,
                        target_observation=sub_obs,
                        relation_type=rel_type,
                        tense=tense,
                        modality=modality,
                        is_factive=is_factive_rel,
                    )
                )
