import time

from kg.kg import KnowledgeGraph

kg = KnowledgeGraph()
# Test set
# bodies = ["The Universe", "The Solar System", "Planet Jupiter"]

bodies = ["The Solar System", "Planet Pluto", "Planet Venus", "Planet Earth", "Planet Mars", "Planet Jupiter", "Planet Saturn", "The moons of Saturn", "Planet Neptune", "Planet Uranus", "The Sun"]

# Build a knowledge graph
# for body in bodies:
#     start_time = time.time()
#     print(f"Processing '{body}'")
#     kg.add_knowledge(body)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Time taken: {elapsed_time:.2f} seconds")

# kg.add_knowledge( "The Planet Venus")

response = kg.interrogate("Could humans live on Venus")
print("My response:", response)