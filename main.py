"""
Main module of the program
"""

from embeddings.embedder import Embedder
from embeddings.models import EmbedderConfig
from raw_texts_processor.processor import Processor
from config.constants import PathsStorage

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
    results = embedder.similarity_search("Specific Language Impairment", k=4)
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
