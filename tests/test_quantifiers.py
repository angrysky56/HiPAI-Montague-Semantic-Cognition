def test_negative_universal(manager):
    # Test "No dogs are cats"
    manager.add_belief("No dogs are cats")

    # Verify Concept_Dog has prop_not_cat = true
    res = manager.world_model.query_graph(
        "MATCH (c:Concept {name: 'Concept_Dog'}) RETURN c.prop_not_cat"
    )
    assert res[0][0] is True


def test_existential(manager):
    # Test "Some humans are immortal"
    manager.add_belief("Some humans are immortal")

    # Verify an anonymous entity is created that is an INSTANCE_OF
    # Concept_Human with property immortal
    res = manager.world_model.query_graph(
        "MATCH (e:Entity)-[:INSTANCE_OF]->(c:Concept {name: 'Concept_Human'}) "
        "WHERE e.id STARTS WITH 'anonymous_' "
        "RETURN e.prop_immortal"
    )
    assert res[0][0] is True


def test_singular_quantifiers(manager):
    # Test "No dog is a bird"
    manager.add_belief("No dog is a bird")
    res = manager.world_model.query_graph(
        "MATCH (c:Concept {name: 'Concept_Dog'}) RETURN c.prop_not_bird"
    )
    assert res[0][0] is True

    # Test "Some cat is fluffy"
    manager.add_belief("Some cat is fluffy")
    res = manager.world_model.query_graph(
        "MATCH (e:Entity)-[:INSTANCE_OF]->(c:Concept {name: 'Concept_Cat'}) "
        "RETURN e.prop_fluffy"
    )
    assert res[0][0] is True
