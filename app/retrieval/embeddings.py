from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

        if "e5" in model_name:
            self.passage_prefix = "passage: "
            self.query_prefix = "query: "
        else:
            self.passage_prefix = ""
            self.query_prefix = ""
    
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        inputs = [self.passage_prefix + text for text in texts]
        vectors = self.model.encode(inputs, normalize_embeddings=True)
        return vectors.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(self.query_prefix + text, normalize_embeddings=True)
        return vector.tolist()
    
@lru_cache
def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    return Embedder(model_name)

if __name__ == "__main__":
    embedder = get_embedder()
    print("模型", embedder.model_name, "维度", embedder.dimension)

    terms = [
        "charging cable",
        "power adapter",
        "wireless charging",
        "screen brightness",
        "warranty card",
    ]
    term_vectors = embedder.embed_passages(terms)

    queries = [
        "The phone gets warm while it is recharging.",
        "屏幕太暗了，看不清。",
    ]
    for query in queries:
        query_vector = embedder.embed_query(query)
        scores = []
        for term, vector in zip(terms, term_vectors):
            score = sum(a * b for a, b in zip(query_vector, vector))
            scores.append((score, term))

        print(query)
        for score, term in sorted(scores, reverse=True):
            print(f"  {score:.3f}  {term}")