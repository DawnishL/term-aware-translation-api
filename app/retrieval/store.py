from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.retrieval.embeddings import Embedder

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "glossary"

class TermStore:

    def __init__(self, 
                 embedder = Embedder,
                 collection_name: str = COLLECTION_NAME,
                 url: str = QDRANT_URL,) -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)

    def rebuild(self, glossary: list[dict[str,str]]) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name),

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.dimension,
                distance=Distance.COSINE,
            ),
        )

        sources = [entry["source"] for entry in glossary]
        vectors = self.embedder.embed_passages(sources)

        points: list[PointStruct] = []

        for index, entry in enumerate(glossary):
            points.append(
                PointStruct(id=index, vector=vectors[index], payload=entry)
            )
        
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, text:str, top_k: int=5) -> list:
        query_vector = self.embedder.embed_query(text)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results: list[dict] = []
        
        for point in response.points:
            results.append({
                "source": point.payload["source"],
                "target": point.payload["target"],
                "score": point.score,
            })
        
        return results

if __name__ == "__main__":
    from app.retrieval.embeddings import get_embedder
    from app.retrieval.glossary import load_glossary

    store = TermStore(get_embedder())
    store.rebuild(load_glossary())
    count = store.client.count(store.collection_name).count
    print("已写入", count, "条术语")

    queries = [
        "The phone gets warm while it is recharging.",
        "屏幕太暗了，看不清。",
        "How long does it take to fill up the battery?",
    ]
    for query in queries:
        print(query)
        for hit in store.search(query, top_k=5):
            print(f"  {hit['score']:.3f}  {hit['source']}  {hit['target']}")