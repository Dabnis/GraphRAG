import time

from kg.kg import KnowledgeGraph

add_knowledge = True

kg = KnowledgeGraph()
# Test set
# bodies = ["LLM context compression", "LLM context size", "neo4j", "cypher queries"]

if add_knowledge:
    # Build a knowledge graph about our solar systems.
    bodies = ["The Solar System", "Planet Pluto", "Planet Venus", "Planet Earth", "Planet Mars", "Planet Jupiter", "Planet Saturn", "The moons of Saturn", "The moons of Saturn", "Planet Neptune", "Planet Uranus", "The Sun"]

    # Build a knowledge graph
    for body in bodies:
        start_time = time.time()
        print(f"Processing '{body}'")
        kg.add_knowledge(body)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")


response = kg.interrogate("Apart from Earth what is the most likely planet that humans could inhabit")
print("My response:", response)