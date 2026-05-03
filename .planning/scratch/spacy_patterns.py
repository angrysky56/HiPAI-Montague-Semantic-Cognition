import spacy

nlp = spacy.load("en_core_web_md")

test_sentences = [
    "Socrates is a man",
    "Socrates is mortal",
    "Socrates harms the Pig",
    "All men are mortal",
    "No man is an island",
    "Social media exploits attention",
    "Alice believes that Bob is happy",
]


def analyze(text):
    print(f"\nSentence: {text}")
    doc = nlp(text)
    for token in doc:
        print(
            f"  {token.text:<12} | {token.pos_:<6} | {token.dep_:<10} | head: {token.head.text}"
        )


if __name__ == "__main__":
    for sent in test_sentences:
        analyze(sent)
