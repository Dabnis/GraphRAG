# GraphRAG
Neo4j DB RAG implementation

"""
       Explanation of what the Cypher query does:
       The Cypher query  uses Neo4j to search and retrieve relationships between nodes based on a full-text
       search and then finds related nodes with specific relationships.
       
       Full-Text Search on Nodes:
       'CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
       YIELD node, score'
       This calls a full-text search function db.index.fulltext.queryNodes on the full-text index named 'entity',
       searching for nodes that match the $query parameter (likely a string search).
       It retrieves up to 2 nodes that match the search query, and YIELD node, score provides the matched node
       and its relevance score.

       Extracting Relationships:
       'CALL (node) {
           WITH node
           MATCH (node)-[r:!MENTIONS]->(neighbor)
           RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
           UNION ALL
           WITH node
           MATCH (node)<-[r:!MENTIONS]-(neighbor)
           RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
       }'
       For each node returned by the full-text search, this subquery is called.
       The first part attempts to find any relationships from the node to other nodes with a relationship :!MENTIONS:

       'MATCH (node)-[r:!MENTIONS]->(neighbor)
       RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output'

       Here, :!MENTIONS is used as a placeholder for relationships with names that include "MENTIONS".
       The MATCH finds all relationships from the node to its neighbors, and the RETURN statement generates
       an output string that includes the IDs of the node, relationship type, and the neighboring node.
       The UNION ALL combines the results with a similar query for relationships that point towards the node:

       'MATCH (node)<-[r:!MENTIONS]-(neighbor)
       RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output

       Output results.
       'WITH output
       RETURN output LIMIT 50'

       This collects the output strings generated from the relationship matches and limits the results to 50.

       The overall query:
       Searches for up to 2 nodes that match a full-text search query.
       For each found node, it retrieves the relationships of type !MENTIONS going both ways (to and from the node).
       It formats these relationships into strings that describe the direction and type.
       It then limits the final output to a maximum of 50 results.

       This query helps to find specific entities based on a text search, explore how they are connected
       to other nodes through "MENTIONS" relationships, and return a description of these connections.
       """
