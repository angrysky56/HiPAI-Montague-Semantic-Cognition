import owlready2
from owlready2 import *

# Use in-memory world
default_world.set_backend(filename=":memory:")
onto = get_ontology("http://test.org/onto")

with onto:

    class Entity(Thing):
        pass

    class Animal(Entity):
        pass

    dog = Entity("MyDog")
    cat = Animal("MyCat")

print(f"dog is_a: {dog.is_a}")
print(f"cat is_a: {cat.is_a}")

# Try adding individual to is_a
try:
    dog.is_a.append(cat)
    print(
        "Added individual to is_a successfully (Wait, this shouldn't be right for OWL)"
    )
except Exception as e:
    print(f"Failed to add individual to is_a: {e}")

# Check types
print(f"Type of Animal class: {type(Animal)}")
print(f"Type of dog individual: {type(dog)}")

# Try issubclass on individual in is_a
print("\nTesting issubclass on is_a elements:")
for item in dog.is_a:
    print(f"Item: {item}, Type: {type(item)}")
    try:
        # Check if item is a class or instance
        is_cls = isinstance(item, ThingClass)
        print(f"  Is ThingClass? {is_cls}")
        if is_cls:
            print(f"  issubclass(item, Thing): {issubclass(item, Thing)}")
        else:
            print(f"  Item is NOT a class, skipping issubclass check")
            # This is where we might have the error if we don't check
            print(f"  Forcing issubclass(item, Thing)...")
            issubclass(item, Thing)
    except TypeError as e:
        print(f"  issubclass failed: {e}")
    except Exception as e:
        print(f"  Other error: {e}")
