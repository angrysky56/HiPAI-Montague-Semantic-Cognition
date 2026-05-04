import os

import owlready2
from owlready2 import World

world = World(filename=":memory:")
print("Attributes of World:", dir(world))
world.close()
