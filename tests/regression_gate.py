import logging
import os
import unittest

from hipai.parser import ClaimExtractor
from hipai.world_model import WorldModel

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)


class RegressionGateTest(unittest.TestCase):
    def setUp(self):
        self.db_path = "world_test.db"
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass
        # Use a unique world_id for test isolation
        self.wm = WorldModel(db_path=self.db_path, graph_name="hipai_test")
        self.wm.clear_graph()
        self.parser = ClaimExtractor()

    def tearDown(self):
        # Clean up
        if hasattr(self.wm, "onto"):
            self.wm.onto.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass

    def add_belief(self, text):
        obs = self.parser.extract(text)
        self.wm.incorporate_observation(obs)

    def test_1_baseline_blocks_head_noun_patient(self):
        """Test that the built-in baseline blocks HARM on a 'patient'."""
        self.add_belief("Eve is a patient.")
        # Baseline should work even without explicit axioms
        res = self.wm.check_action("Antigravity", "HARM", "eve")
        print(f"\nTest 1 Result: {res}")
        self.assertFalse(res["permitted"], "Baseline should block HARM on patient")
        self.assertEqual(res["blocking_axiom"], "T1-HARMS-PROTECTION")

    def test_2_custom_axiom_blocks_compound_class(self):
        """Test that a custom axiom on 'moral patient' blocks HARM on Dave."""
        # 1. Setup world state
        self.add_belief("Dave is a moral patient.")

        # 2. Add custom axiom via paraclete
        axiom = {
            "source_axiom": "CUSTOM-MORAL-PROTECTION",
            "relation_type": "HARM",
            "object_type": "moral patient",
            "subject_type": "Any",
            "tier": "T1",
            "constraint": "FORBIDDEN",
        }
        self.wm.paraclete.incorporate_axiom(axiom)

        # 3. Check action
        res = self.wm.check_action("Antigravity", "HARM", "dave")
        print(f"\nTest 2 Result: {res}")
        self.assertFalse(
            res["permitted"], "Custom axiom should block HARM on moral patient"
        )
        self.assertEqual(res["blocking_axiom"], "CUSTOM-MORAL-PROTECTION")

    def test_3_depth_2_syllogism(self):
        """Test that reasoning still works (Syllogism)."""
        self.add_belief("All men are mortal.")
        self.add_belief("Socrates is a man.")

        # Check if Socrates is Mortal via the graph hierarchy
        # Use a variable-length path to account for subclasses
        # Note: Graph uses INSTANCE_OF for Entity -> Concept links
        query = """
        MATCH (s:Entity {id: 'socrates'})-[:INSTANCE_OF]->(:Concept)-[:SUBCLASS_OF*0..]->(c:Concept {name: 'Concept_Mortal'})
        RETURN c
        """
        res = self.wm.query_graph(query)
        print(f"\nTest 3 Graph Result: {res}")
        self.assertTrue(
            len(res) > 0, "Socrates should be linked to Concept_Mortal via hierarchy"
        )


if __name__ == "__main__":
    unittest.main()
