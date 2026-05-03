import spacy

from hipai.models import Observation
from hipai.parser import ClaimExtractor


def test_attitude():
    extractor = ClaimExtractor()
    text = "Alice believes that Bob is happy"
    obs = extractor.extract(text)

    print(f"Observation for: {text}")
    print(f"Individuals: {[ind.name for ind in obs.individuals]}")
    print(f"Relations: {len(obs.relations)}")

    for rel in obs.relations:
        print(
            f"  Relation: {rel.source_id} -[{rel.relation_type}]-> {rel.target_id or 'NESTED'}"
        )
        if rel.target_observation:
            print(
                f"    Nested Obs Individuals: {[ind.name for ind in rel.target_observation.individuals]}"
            )
            for ind in rel.target_observation.individuals:
                print(f"      Individual {ind.name} Properties: {ind.properties}")
            for n_rel in rel.target_observation.relations:
                print(
                    f"      Nested Relation: {n_rel.source_id} -[{n_rel.relation_type}]-> {n_rel.target_id}"
                )


if __name__ == "__main__":
    test_attitude()
