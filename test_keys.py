import asyncio
from falkordb import FalkorDB

db = FalkorDB(host='localhost', port=6380)
graph = db.select_graph("test_synthesis")
res = graph.query("MATCH (n:Entity {id: 'Socrates'}) RETURN keys(n)")
if res.result_set:
    print(res.result_set)
else:
    print("Empty result")
