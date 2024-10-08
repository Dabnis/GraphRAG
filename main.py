import time

from kg.kg import KnowledgeGraph

add_knowledge = True

kg = KnowledgeGraph()
# Test set
# bodies = ["LLM context compression", "LLM context size", "neo4j", "cypher queries"]

if add_knowledge:
    # Build a knowledge graph about our solar systems.
    core_subject = ["The Solar System", "The Planet Pluto", "The Planet Venus", "The Planet Earth", "The Planet Mars",
              "The Planet Jupiter", "The Planet Saturn", "The moons of Saturn", "The moons of Saturn",
              "The Planet Neptune", "The Planet Uranus", "The Sun"]

    # Build the knowledge graph
    for subject in core_subject:
        start_time = time.time()
        print(f"Processing '{subject}'")
        kg.add_knowledge(subject)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")


response = kg.interrogate("Apart from Earth what is the most likely planet that humans could inhabit")
print("My response:", response)