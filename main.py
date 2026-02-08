"""
Main module of the program
"""

from embeddings.embedder import Embedder
from embeddings.models import EmbedderConfig
from raw_texts_processor.processor import Processor
from config.constants import PathsStorage

if __name__ == "__main__":
    embedder_config = EmbedderConfig(
        parent_chunk_size=1024,
        parent_chunk_overlap=256,
        child_chunk_size=512,
        child_chunk_overlap=128,
    )

    embedder = Embedder(config=embedder_config, recreate_collection=True)

    embedder.add_documents(texts=["""### 4 #### How Language Works Journalists say that when a dog bites a man that is not news, but when a man bites a dog that is news. This is the essence of the language instinct: language conveys news. The streams of words called are not just memory prods, reminding you of man and man's best friend and letting you fill in the rest; they tell you who in fact did what to whom. Thus we get more from most stretches of language than Woody Allen got from _War and Peace,_ which he read in two hours after taking speed-reading lessons: Language allows us to know how octopuses make love and how to remove cherry stains and why Tad was heartbroken, and whether the Red Sox will win the World Series without a good relief pitcher and how to build an atom bomb in your basement and how Catherine the Great died, among other things. When scientists see some apparent magic trick in nature, like bats homing in on insects in pitch blackness or salmon returning to breed in their natal stream, they look for the engineering principles behind it. For bats, the trick turned out to be sonar; for salmon, it was locking in to a faint scent trail. What is the trick behind the ability of _Homo_ _sapiens_ to convey that man bites dog? In fact there is not one trick but two, and they are associated with the names of two European scholars who wrote in the nineteenth century. The first principle, articulated by the Swiss linguist Ferdinand de Saussure, is "the arbitrariness of the sign," the wholly conventional pairing of a sound with a meaning. The word _dog_ does not look like a dog, walk like a dog, or woof like a dog, but it means""",],)
    results = embedder.similarity_search("fish", k=2)
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
