"""
Agent for RAG system with tool usage
"""
from langchain_community.llms.vllm import VLLM
from langchain_huggingface import HuggingFaceEmbeddings

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
            llm
    ):
        """
        Initialize RAG Agent

        Args:
            embedder (Embedder): Embedder instance for retrieval
            config (AgentConfig): Agent configuration
            llm (Optional[ChatOpenAI]): Language model for generation
        """
        self.embedder = embedder
        self.config = config
        self.llm = llm

        # Get tools from embedder
        self.tools = embedder.get_tools()

        logger.info("RAG Agent initialized with model: %s", config.llm_model_name)

    def search_and_respond(self, question: str) -> str:
        """
        Simple search and respond workflow

        Args:
            question (str): User question

        Returns:
            str: Response
        """
        try:
            # Step 1: Search for relevant chunks
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

            # Step 2: Extract parent IDs and get full context if needed
            lines = search_result.split("\n")
            context_parts = []
            for i in range(0, len(lines), 5):  # Each chunk has 4 lines + separator
                if i + 3 < len(lines):
                    parent_id = lines[i].replace("Parent ID: ", "").strip()
                    doc_id = lines[i + 1].replace("Document ID: ", "").strip()
                    content = lines[i + 3].replace("Content: ", "").strip()
                    context_parts.append(f"Context (parent {parent_id}): {content}")

                    # Optionally get full parent chunk
                    if self.config.use_full_context:
                        parent_tool = self.tools[1]
                        parent_result = parent_tool.invoke({
                                "parent_id": parent_id,
                                "document_id": doc_id
                            })
                        if parent_result not in ["NO PARENT COLLECTION",
                                                     "PARENT_CHUNK_NOT_FOUND"]:
                            context_parts.append(f"Full context: {parent_result}")

                    # Step 3: Generate response with LLM
            context = "\n\n".join(context_parts)
            prompt = f"""
                            You are a helpful AI assistant that answers questions based on the provided context.
                                
                                Rules:
                                1. Only use information from the provided context to answer questions
                                2. If the context doesn't contain enough information, say so honestly
                                3. Be specific and cite relevant parts of the context
                                4. Keep your answers clear and concise
                                5. If you're unsure, admit it rather than guessing
                                
                                Context:
                                {context}
                                
                                Question: {question}
                                
                                Answer based on the context above:
                                """
            response_to_return = self.llm.invoke(prompt)
            return response_to_return.content

        except (ValueError, TypeError) as e:
            logger.error("Error in search_and_respond: %s", str(e))
            return f"Request processing error: {str(e)}"


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
                               max_tokens=500,
                               retrieval_k=4,
                               similarity_threshold=0.6,
                               use_tools=True,
                               max_iterations=5,
                               use_full_context=False,)
    agent = RAGAgent(embedder=embedder_for_agent, config=agent_config,
                     llm=VLLM)
    response = agent.search_and_respond(QUESTION)
