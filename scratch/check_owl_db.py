import owlready2
from owlready2 import World
import os

world = World(filename=":memory:")
print("Type of World.graph.db:", type(world.graph.db))
world.close()
