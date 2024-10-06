from kg.kg import KnowledgeGraph

kg = KnowledgeGraph()
# kg.add_knowledge("Italian geography")
# kg.add_knowledge("Italian language")
response = kg.interrogate("give me a list of locations where Roman emperors' have lived, if possible giving dated to from?")
print(response)