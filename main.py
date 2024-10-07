import time

from kg.kg import KnowledgeGraph

kg = KnowledgeGraph()
# Test set
bodies = ["The Universe", "The Solar System", "Planet Jupiter"]

# bodies = ["The Universe", "The Solar System", "Planet Pluto", "Planet Venus", "Planet Earth", "Planet Mars", "Planet Jupiter", "Planet Saturn", "The moons of Saturn", "The moons of Saturn", "Planet Neptune", "Planet Uranus", "The Sun", "Astronomy and mythology", "Known planets outside our solar system"]

# Build a knowledge graph
# for body in bodies:
#     start_time = time.time()
#     print(f"Processing '{body}'")
#     # Replace 'kg.add_knowledge("The Universe")' with your actual code here
#     kg.add_knowledge(body)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Time taken: {elapsed_time:.2f} seconds")


# print("You asked:", text_input)
response = kg.interrogate("What size is the universe")
print("My response:", response)