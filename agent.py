"""
Agent for RAG system with tool usage
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from tqdm import tqdm

from src.config.constants import LOGGER as logger
from src.config.models import AgentConfig, EmbedderConfig
from src.embeddings.embedder import Embedder, EmbedSparse
from src.helpers.create_vector_db import VectorDatabase


class RAGAgent:
    """
    RAG Agent that uses embedder for retrieval and LLM for response generation
    """

    def __init__(
            self,
            embedder: Embedder,
            config: AgentConfig,
            llm: ChatOllama
    ):
        """
        Initialize RAG Agent

        Args:
            embedder (Embedder): Embedder instance for retrieval
            config (AgentConfig): Agent configuration
            llm (Optional[ChatOllama]): Language model for generation
        """
        self.embedder = embedder
        self.config = config
        self.llm = llm
        self.tools = embedder.get_tools()

        logger.info("RAG Agent initialized with model: %s", config.llm_model_name)

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}({self.config=!r})"

    def search_child_chunks(self, question: str) -> str:
        """
        Search for relevant child chunks based on query.

        Args:
            question (str): Search query

        Returns:
            str: Formatted child chunks with metadata or "NO RELEVANT CHUNKS FOUND"
        """
        logger.info("Searching for chunks related to: %s", question)
        search_tool = self.tools[0]
        if not search_tool:
            return "The search tool is not available"
        search_result = search_tool.invoke({
                    "query": question,
                    "limit": self.config.retrieval_k
                })
        if search_result == "NO RELEVANT CHUNKS FOUND":
            return "No relevant information was found to answer the question."
        return search_result

    def search_parent_chunk(self, child_search_result: str) -> str:
        """
            Retrieve full parent chunks using IDs from child search results.

            Args:
                child_search_result (str): Output from search_child_chunks()

            Returns:
                str: Combined parent chunks or error message if none found
        """
        parent_tool = self.tools[1]
        lines = child_search_result.split("\n")
        parent_chunks = []
        unique_parents = []
        for i in range(0, len(lines), 5):
            parent_id = lines[i].replace("Parent ID: ", "").strip()
            doc_id = lines[i + 1].replace("Document ID: ", "").strip()
            if (parent_id, doc_id) not in unique_parents:
                unique_parents.append((parent_id, doc_id))

        for parent_id, doc_id in tqdm(unique_parents, desc="Retrieving parent chunks"):
            result = parent_tool.invoke({
                "parent_id": parent_id,
                "document_id": doc_id
            })
            if result not in ["NO PARENT COLLECTION", "PARENT_CHUNK_NOT_FOUND"]:
                parent_chunks.append(result)

        if not parent_chunks:
            return "No parent chunks could be retrieved."

        logger.info("Successfully retrieved %d parent chunks", len(parent_chunks))
        return "\n\n---\n\n".join(parent_chunks)

    def needs_parent_chunk(self, question: str, child_chunk: str) -> bool:
        """
        Determine if parent chunk is needed to answer the question.

        Args:
            question (str): User question
            child_chunk (str): Child chunk content

        Returns:
            bool: True if parent chunk is needed, False otherwise
        """
        prompt = f"""
                    Based on the question and the available text chunk, 
                    determine if you need MORE CONTEXT to properly answer the question.

                    Question: {question}
                
                    Available text chunk: {child_chunk}
                
                    Do you need the full parent chunk (larger context) to answer this question?
                    Answer ONLY "yes" or "no".
                    """
        need_parent = self.llm.invoke(prompt)
        if str(need_parent.content).lower() == 'yes':
            return True
        return False

    def respond(self, context: str, question: str) -> str:
        """
        Generate answer using LLM based on retrieved context.

        Args:
            question (str): User question
            context (str): Retrieved context information

        Returns:
            str: Generated response based on context
        """
        prompt = f"""
            You are a helpful AI assistant that answers questions based on the provided context.
                                
                Rules:
                    1. Only use information from the provided context to answer questions
                    2. If the context doesn't contain enough information, say so honestly
                    3. Be specific and cite relevant parts of the context
                    4. Keep your answers clear and concise
                    5. If you're unsure, admit it rather than guessing
                                
                Context: {context}
                                
                Question: {question}
                                
                Answer based on the context above:
                """
        response_to_return = self.llm.invoke(prompt)
        return str(response_to_return.content)


if __name__ == '__main__':
    QUESTION = str(input('Введите запрос: '))
    embedder_config = EmbedderConfig(
        parent_chunk_size=1024,
        parent_chunk_overlap=256,
        child_chunk_size=512,
        child_chunk_overlap=128,
    )
    embedder_for_agent = Embedder(config=embedder_config,
                        embeddings_model=HuggingFaceEmbeddings,
                        sparse_model=EmbedSparse,
                        vector_db=VectorDatabase,
                        recreate_collection=True)
    agent_config = AgentConfig(llm_model_name="gpt-3.5-turbo",
                               temperature=0.7,
                               retrieval_k=4
                               )
    agent_llm = ChatOllama(model=agent_config.llm_model_name,
                           temperature=agent_config.temperature)
    agent = RAGAgent(embedder=embedder_for_agent, config=agent_config,
                     llm=agent_llm)
    context_child = agent.search_child_chunks(QUESTION)
    if agent.needs_parent_chunk(QUESTION, context_child) is True:
        CONTEXT_PARENT = agent.search_parent_chunk(context_child)
        print(agent.respond(CONTEXT_PARENT, QUESTION))
    elif agent.needs_parent_chunk(QUESTION, context_child) is False:
        print(agent.respond(context_child, QUESTION))
