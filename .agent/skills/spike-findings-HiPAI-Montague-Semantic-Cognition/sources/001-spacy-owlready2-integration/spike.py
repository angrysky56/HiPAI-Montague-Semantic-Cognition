import spacy
from owlready2 import *
import sys

def run_spike():
    print("--- spaCy + owlready2 Spike ---")
    
    # 1. Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # 2. Setup Ontology
    onto = get_ontology("http://test.org/spacy_owl_spike.owl")

    sentences = [
        "Socrates is a man",
        "All men are mortal",
        "All philosophers are men",
        "Aristotle is a philosopher"
    ]

    print("\n[1] Parsing sentences with spaCy and building OWL ontology...")
    
    for sent in sentences:
        doc = nlp(sent)
        print(f"  Sentence: '{sent}'")
        
        subject = None
        attribute = None
        is_all = sent.lower().startswith("all")
        
        for token in doc:
            # print(f"    Token: {token.text} | Dep: {token.dep_} | Lemma: {token.lemma_}")
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subject = token.lemma_.capitalize()
            elif token.dep_ in ["attr", "acomp"]:
                attribute = token.lemma_.capitalize()

        if not subject or not attribute:
            print(f"    WARNING: Could not extract subject or attribute. Subject={subject}, Attribute={attribute}")
            continue

        if is_all:
            # "All X are Y" -> X and Y are classes, X subClassOf Y
            print(f"    Mapping: Class({subject}) ⊆ Class({attribute})")
            with onto:
                class_sub = types.new_class(subject, (Thing,))
                class_super = types.new_class(attribute, (Thing,))
                class_sub.is_a.append(class_super)
        else:
            # "X is a Y" -> X is Individual of Class Y
            print(f"    Mapping: Individual({subject}) ∈ Class({attribute})")
            with onto:
                class_attr = types.new_class(attribute, (Thing,))
                individual = class_attr(subject)

    print("\n[2] Running HermiT Reasoner...")
    try:
        with onto:
            sync_reasoner()
    except Exception as e:
        print(f"Reasoner error: {e}")
        return

    print("\n[3] Verification: Is Aristotle mortal?")
    
    # Check Aristotle
    try:
        aristotle = onto.search_one(iri="*Aristotle")
        mortal_class = onto.search_one(iri="*Mortal")
        
        if aristotle and mortal_class:
            # Check if Aristotle is an instance of Mortal (includes inferred)
            is_mortal = isinstance(aristotle, mortal_class)
            print(f"  Aristotle's direct classes: {[c.name for c in aristotle.is_a]}")
            print(f"  Is Aristotle mortal? {'YES ✓' if is_mortal else 'NO ✗'}")
        else:
            print("  Error: Aristotle or Mortal class not found in ontology.")
    except Exception as e:
        print(f"  Verification error: {e}")

if __name__ == "__main__":
    run_spike()
