from functools import lru_cache

from app.config import Settings, get_settings

from app.retrieval.embeddings import get_embedder
from app.retrieval.glossary import load_glossary
from app.retrieval.matching import ExactMatcher
from app.retrieval.store import TermStore
from app.schemas import TermHit, SearchResponse

def merge_results(must_use:list[dict],
                  vector_hits:list[dict],
                  top_k:int) -> list[dict]:
    must_use_source = [entry["source"] for entry in must_use]

    reference: list[dict] = []
    for hit in vector_hits:
        if hit["source"] not in must_use_source:
            reference.append(hit)
    return reference[:top_k]

class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        glossary = load_glossary(settings.glossary_path)
        self.matcher = ExactMatcher(glossary)
        self.store = TermStore(get_embedder(settings.embedding_model),
                               collection_name=settings.qdrant_collection,
                               url=settings.qdrant_url,
                               )
        self.store.rebuild(glossary)
    
    def search(self, text: str, top_k:int):
        must_use = self.matcher.find(text)
        vector_hits = self.store.search(text, top_k=top_k + len(must_use))

        reference = merge_results(must_use, vector_hits, top_k)

        must_use_hits: list[TermHit] = []
        for hit in must_use:
            must_use_hits.append(
                TermHit(
                    source=hit["source"],
                    target=hit["target"],
                    match_type="exact",
                )
            )
        reference_hits: list[TermHit] = []
        for hit in reference:
            reference_hits.append(
                TermHit(
                    source=hit["source"],
                    target=hit["target"],
                    match_type="vector",
                    score=hit["score"],
                )
            )
        
        return SearchResponse(query=text, must_use=must_use_hits, reference=reference_hits)

@lru_cache
def get_retrieval_service() -> RetrievalService:
    """返回全局唯一的检索服务，整个程序只创建一次。"""
    return RetrievalService(get_settings())


if __name__ == "__main__":
    from app.config import get_settings

    service = RetrievalService(get_settings())
    result = service.search("Plug the charger into the charging port.", top_k=5)
    print(result.model_dump_json(indent=2))