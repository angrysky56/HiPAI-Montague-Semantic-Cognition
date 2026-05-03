import spacy
import owlready2
import sys

def main():
    print("Checking spaCy...")
    try:
        nlp = spacy.load("en_core_web_md")
        doc = nlp("Socrates is a man.")
        print(f"spaCy loaded. Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
    except Exception as e:
        print(f"spaCy failure: {e}")
        sys.exit(1)

    print("Checking owlready2...")
    try:
        onto = owlready2.get_ontology("http://test.org/onto.owl")
        with onto:
            class Man(owlready2.Thing): pass
        print("owlready2 loaded. Created class 'Man'.")
    except Exception as e:
        print(f"owlready2 failure: {e}")
        sys.exit(1)

    print("\nEnvironment Ready.")

if __name__ == "__main__":
    main()
