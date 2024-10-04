from kg.kg import KnowledgeGraph

kg = KnowledgeGraph()
kg.add_knowledge("The Roman Empire")
response = kg.interrogate("Who was it's first emperor, when were they given this title and why?")
print(response)