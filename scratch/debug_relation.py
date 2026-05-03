from hipai.synthesis import HIPAIManager
from hipai.models import Observation
from hipai.parser import ClaimExtractor

mgr = HIPAIManager(db_path=":memory:")
parser = ClaimExtractor()

text = "Alice will visit Bob"
obs = parser.extract(text)
print(f"Parsed obs: {obs}")
print(f"Relations: {obs.relations}")

mgr.world_model.incorporate_observation(obs)

q = "MATCH (a)-[r]->(b) RETURN labels(a), a.id, type(r), labels(b), b.id, r.tense"
res = mgr.world_model.query_graph(q)
print(f"Graph relations: {res}")
