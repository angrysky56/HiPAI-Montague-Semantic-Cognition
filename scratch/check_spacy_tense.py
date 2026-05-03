import spacy
nlp = spacy.load("en_core_web_md")
doc = nlp("Socrates was a man")
for token in doc:
    print(f"{token.text} | {token.lemma_} | {token.pos_} | {token.tag_} | {token.dep_}")
