import re
import time
import warnings

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple
import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough, RunnableBranch, RunnableLambda,
)

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

warnings.filterwarnings("ignore")

LLM = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")


# **REM: A Pydantic model used to collect 'entities' that will eventually become neo4j nodes.
class Entities(BaseModel):
    """
    Giving hints to what we are interested in  (generally)
    The LLM 'should' read this description to get instruction on what entities to extract from
    our text chunks. **REM 'entities' will become neo4j nodes
    E.G. If our knowledge graph was to be used by a crew that was tasked with processes relating
    to the 'auto industry' the description below would be more like:
    Make; Model; Colour; Engine type, etc.
    N.B. At present we have one set of entities 'names', but this could be expanded.
    """
    names: List[str] = Field(
        None,
        description="All the people, organization, or business entities that appear in the text",
    )

class KnowledgeGraph:
    def __init__(self):
        self.llm = LLM
        self.neo4j = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )
        self.graph_transformer = LLMGraphTransformer(self.llm)
        self.vector_index = None

    def add_knowledge(self, subject: str) -> None:
        # One example of how we can 'gain'/add knowledge to our knowledge graph.
        raw_docs = WikipediaLoader(query=subject, load_max_docs=25).load()
        # CSV loader; pdf loader; web scraping loader; MySql loader
        # Set the chunking/text splitting strategy
        """
        **REM: This is a main area of my present research. Splitting on a fixed chunk size bears no
        relationship to how a context is structured within a text block. A more intelligent splitting
        process that maintains context edges will I believe significantly improve accuracy, stability,
        while at the same time reducing the size of context, concentrating the context. Both I believe
        facilitating the use of smaller LLMs, etc. N.B. Smaller contexts place less stress on an LLM
        and its required resources.
        """
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        # Here we just take the first 3 chunks as an example. I need to look at making this more dynamic & relative
        # to subject in hand.
        docs = text_splitter.split_documents(raw_docs[:20])
        # Transform our chunks -> graph documents
        graph_docs = self.graph_transformer.convert_to_graph_documents(docs)
        # ADD to our neo4j
        self.neo4j.add_graph_documents(
            graph_docs,
            include_source=True,
            baseEntityLabel=True,
        )
        # Vectorise Document node text content. Presently all
        # create vector index
        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
            index_name='dabnis',
        )
        # Initialise full text index. Cypher Query!
        self.neo4j.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    def _gen_full_txt_qry(self, ip: str) -> str:
        """
            Generate a full-text search query for a given input string.

            This function constructs a query string suitable for a full-text search.
            It processes the input string by splitting it into words and appending a
            similarity threshold (~2 changed characters) to each word, then combines
            them using the AND operator. Useful for mapping entities from user questions
            to database values, and allows for some misspellings.
            """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(ip).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Fulltext index query
    def _structured_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organizations, people, locations entities from the text.",
                ),
                (
                    "human",
                    "Extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        chain = prompt | LLM.with_structured_output(Entities)
        # Invoke our entity chain
        entities = chain.invoke({"question": question})
        # Init result string
        result = ""
        for entity in entities.names:
            # Fix for deprecation warnings.
            response = self.neo4j.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL (node) {
                    WITH node
                    MATCH (node)-[r:!MENTIONS]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r:!MENTIONS]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                WITH output
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )

            # Collate the response
            result += "\n".join([el["output"] for el in response])
        return result

    # Combine full text & vector similarity search outputs
    def _retriever(self, question: str) -> str:
        # Full text search, nearest neighbour, etc
        structured_data = self._structured_retriever(question)
        # Vector search
        v_results = self.neo4j.query(
            """
            WITH genai.vector.encode(
            $question, "OpenAI", { token: $api_key }) AS userEmbedding
            CALL db.index.vector.queryNodes('vector', 6, userEmbedding)
            YIELD node, score
            RETURN node.text, score
            """,
            {"question": question, "api_key": OPENAI_API_KEY}
        )
        # Set a threshold of acceptability of response.
        threshold = .89
        # Filter results based on the threshold
        filtered_results = [result for result in v_results if result['score'] > threshold]
        # Sort the filtered results by score in descending order
        sorted_data = sorted(filtered_results, key=lambda x: x['score'], reverse=True)
        # Build a single text block.
        unstructured_data = " ".join(result['node.text'] for result in sorted_data)

        # unstructured_data = [
        #     el.page_content for el in self.vector_index.similarity_search(question)
        # ]
        resp = f"""
        Structured data: {structured_data}
        Unstructured data: {unstructured_data}
        """
        response = self.remove_ws(resp)
        return response

    def _format_chat_history(self,chat_history: List[Tuple[str, str]]) -> List:
        """
        Collates chat history
        :param chat_history:
        :return:
        """
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def remove_ws(self,text: str) -> str:
        # Regular expression pattern to match multiple consecutive whitespace characters
        pattern = r"\s+"

        # Replace multiple consecutive whitespace characters with a single space
        cleaned = re.sub(pattern, " ", text)

        return cleaned

    def interrogate(self, query: str) -> str:
        # Condense a chat history and follow-up question into a standalone question
        _template = """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """

        condense_question_prompt = PromptTemplate.from_template(_template)
        # Create our search query
        search_query = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),  # Condense follow-up question and chat into a standalone_question
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | condense_question_prompt
                | ChatOpenAI(temperature=0)
                | StrOutputParser(),
            ),
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x: x["question"]),
        )
        # Now for our final chain!
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        """
        Take note of how we create the complete 'context', in real time by combining the results of
        our full text and vector searches of our knowledge graph.
        The 'important point to take note of is:
        Over time you never loose info from the context as it's generated in real time AND composed
        of ALL relevant information related to the incoming query.
        This will improve stability; accuracy; repeatability. Repeatability is the key to interfacing
        AI systems with existing 'logical' processes. :)
        """

        # context = search_query | self._retriever

        chain = (
                RunnableParallel(
                    {
                        # Here we use our real time dynamic context
                        "context": search_query | self._retriever,
                        "question": RunnablePassthrough(),
                    }
                )
                | prompt
                | LLM
                | StrOutputParser()
        )
        # Invoke the chain
        return  chain.invoke(
            {
                "question": query,
                # "chat_history": [
                #     ("Who was the first emperor?", "Augustus was the first emperor.")
                # ],
            }
        )

