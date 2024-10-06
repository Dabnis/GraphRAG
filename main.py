import time

from kg.kg import KnowledgeGraph

kg = KnowledgeGraph()

# bodies = ["The Universe", "The Solar System", "Planet Pluto", "Planet Venus", "Planet Earth", "Planet Mars", "Planet Jupiter", "Planet Saturn", "The moons of Saturn", "The moons of Saturn", "Planet Neptune", "Planet Uranus", "The Sun of our solar system"]
bodies = ["The Universe", "The Solar System"]
# Build a knowledge graph
for body in bodies:
    start_time = time.time()
    print(f"Processing '{body}'")
    # Replace 'kg.add_knowledge("The Universe")' with your actual code here
    kg.add_knowledge(body)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

response = kg.interrogate("Of all of our solar system planets and moons, which is the most likely to have some form of life?")
print(response)