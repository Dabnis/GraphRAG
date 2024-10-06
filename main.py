import time

from kg.kg import KnowledgeGraph

kg = KnowledgeGraph()
# Test set
# bodies = ["The Universe", "The Solar System"]

bodies = ["The Universe", "The Solar System", "Planet Pluto", "Planet Venus", "Planet Earth", "Planet Mars", "Planet Jupiter", "Planet Saturn", "The moons of Saturn", "The moons of Saturn", "Planet Neptune", "Planet Uranus", "The Sun of our solar system"]

# Build a knowledge graph
# for body in bodies:
#     start_time = time.time()
#     print(f"Processing '{body}'")
#     # Replace 'kg.add_knowledge("The Universe")' with your actual code here
#     kg.add_knowledge(body)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Time taken: {elapsed_time:.2f} seconds")

# kg.add_knowledge("Astronomy and mythology")
# kg.add_knowledge("Known planets outside of our solar system")


# print("You asked:", text_input)
response = kg.interrogate("Explain what solar mass is, and give the solar mass of the earth")
print("My response:", response)