from hipai.parser import ClaimExtractor

def test_parser():
    extractor = ClaimExtractor()
    
    test_cases = [
        "Socrates is a man",
        "Socrates is mortal",
        "Socrates harms the Pig",
        "Socrates does not harm the Pig",
        "Socrates is not a god"
    ]
    
    for text in test_cases:
        obs = extractor.extract(text)
        print(f"\nText: {text}")
        print(f"Individuals: {[(ind.name, ind.properties) for ind in obs.individuals]}")
        print(f"Relations: {[(rel.relation_type, rel.source_id, rel.target_id) for rel in obs.relations]}")

if __name__ == "__main__":
    test_parser()
