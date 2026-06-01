import pytest

from hipai.models import DeontologicalAxiom
from hipai.parser import ClaimExtractor
from hipai.world_model import WorldModel


@pytest.fixture
def wm():
    model = WorldModel(graph_name="test_v051", db_path="test_v051.db")
    model.clear_database()
    model.ontology.seed_axioms()
    yield model
    model.close()


def test_nationality_lemmatization(wm):
    extractor = ClaimExtractor()

    # 1. Individual assertion with proper noun class
    obs1 = extractor.extract("Diogenes is a Greek")
    wm.incorporate_observation(obs1)

    # 2. Universal assertion with plural proper noun
    obs2 = extractor.extract("All Greeks are philosophers")
    wm.incorporate_observation(obs2)

    # Verify Diogenes IS_A Greek (singular/lemmatized)
    res = wm.query_graph(
        "MATCH (n:Entity {id: 'diogenes'})-[:INSTANCE_OF]->(c) RETURN c.name"
    )
    assert res == [["Concept_Greek"]]

    # Verify only ONE Greek concept exists
    res = wm.query_graph(
        "MATCH (c:Concept) WHERE c.name CONTAINS 'Greek' RETURN c.name"
    )
    assert len(res) == 1
    assert res[0][0] == "Concept_Greek"

    # Verify Diogenes -> Greek -> Philosopher chain (transitive subsumption)
    res = wm.query_graph(
        "MATCH (n:Entity {id: 'diogenes'})-[:INSTANCE_OF]->(:Concept {name: 'Concept_Greek'})-[:SUBCLASS_OF]->(p:Concept) RETURN p.name"
    )
    assert res == [["Concept_Philosopher"]]


def test_action_gate_baseline(wm):
    extractor = ClaimExtractor()
    wm.incorporate_observation(extractor.extract("Eve is a patient"))

    # Baseline HARM protection for Patient
    result = wm.check_constraint("carol", "harm", "eve")
    assert result["permitted"] is False
    assert "Patient" in result["reasoning"] or "patient" in result["reasoning"]


def test_action_gate_custom_axiom(wm):
    extractor = ClaimExtractor()

    # Load a custom axiom A3 that protects a NOVEL type the baseline
    # hierarchy does not cover (an ecosystem against pollution). This is the
    # honest test of custom-axiom functionality: the baseline HARM/Patient
    # protection does not apply here, so any block MUST be attributable to A3.
    a3 = DeontologicalAxiom(
        tier="T1",
        subject_type="Any",
        relation_type="pollute",
        object_type="ecosystem",
        constraint="FORBIDDEN",
        source_axiom="A3",
    )
    wm.incorporate_axiom(a3)

    wm.incorporate_observation(extractor.extract("Wetland is an ecosystem"))
    wm.incorporate_observation(extractor.extract("Factory is an agent"))

    # The custom axiom A3 must block and be named in the reasoning.
    result = wm.check_constraint("factory", "pollute", "wetland")
    assert result["permitted"] is False
    assert "A3" in result["reasoning"]
    assert result["blocking_axiom"] == "A3"
    assert "A3" in result.get("blocking_axioms", [])

