"""
Main module of the program
"""
from embeddings.embedder import Embedder
from embeddings.models import EmbedderConfig
from langchain_ollama import ChatOllama
if __name__ == "__main__":
    embedder_config = EmbedderConfig(
        parent_chunk_size=2048,
        parent_chunk_overlap=512,
        child_chunk_size=1024,
        child_chunk_overlap=256,
    )

    embedder = Embedder(config=embedder_config, recreate_collection=True)

    with open("data/raw_texts/md_storage/Steven_Pinker_-_The_Language_Instinct_HarperCollins.md", "r", encoding="utf-8") as file:
        data = file.read()

    embedder.add_documents(texts=[data,],)
    tools = embedder.get_tools()

    model = ChatOllama(model="mistral")
    model_with_tools = model.bind_tools(tools)

    response_text = model.invoke("What is X bar?")

