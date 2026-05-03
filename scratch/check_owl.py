import owlready2
from owlready2 import World
import os

world = World(filename=":memory:")
print("Attributes of World:", dir(world))
world.close()
