import spacy

def test():
    nlp = spacy.load("en_core_web_md")
    text = "Alice believes that Bob is happy"
    doc = nlp(text)
    print(f"Text: {text}")
    for token in doc:
        print(f"Token: {token.text:10} | Dep: {token.dep_:10} | Head: {token.head.text:10} | POS: {token.pos_:5}")

if __name__ == "__main__":
    test()
