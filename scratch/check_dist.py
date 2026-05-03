from hipai.world_model import WorldModel
wm = WorldModel()
v1 = wm._get_embedding("Alice")
v2 = wm._get_embedding("Socrates")
import numpy as np
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

dist = cosine_distance(v1, v2)
print(f"Distance between Alice and Socrates: {dist}")
