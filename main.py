"""
Main module of the program
"""

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama
# from langchain_huggingface import HuggingFaceEmbeddings
# from src.config.constants import PROMPT_TEMPLATE
# from src.embeddings.embedder import Embedder, EmbedSparse
# from src.embeddings.models import EmbedderConfig
# from src.helpers.create_vector_db import VectorDatabase
# if __name__ == "__main__":
#     embedder_config = EmbedderConfig(
#         parent_chunk_size=2048,
#         parent_chunk_overlap=512,
#         child_chunk_size=1024,
#         child_chunk_overlap=256,
#     )

#     embedder = Embedder(
#         config=embedder_config,
#         embeddings_model=HuggingFaceEmbeddings,
#         sparse_model=EmbedSparse,
#         vector_db=VectorDatabase,
#         recreate_collection=True,
#     )

#     with open("pinker.md", "r", encoding="utf-8") as file:
#         data = file.read()

#     embedder.add_documents(
#         texts=[
#             data,
#         ],
#     )
#     tools = embedder.get_tools()
#     query_text = "Did Noam Chomsky support the ideas of behaviorism or descriptivism?"
#     results = embedder.similarity_search_with_score(query_text, 4)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     model = ChatOllama(model="mistral")
#     model_with_tools = model.bind_tools(tools)

#     response_text = model_with_tools.invoke(prompt)
#     sources = [doc.metadata.get("chunk_id", None) for doc, _ in results]

#     formatted_response = f"Response: {response_text.text}, Sources: {sources}"
#     print(formatted_response)
