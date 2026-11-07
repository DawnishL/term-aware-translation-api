from functools import lru_cache
from app.config import get_settings
from app.schemas import TranslateResponse
from app.llm import OllamaClient
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.service import RetrievalService, get_retrieval_service

class TranslationService:
    def __init__(self, retrieval: RetrievalService, llm: OllamaClient) -> None:
        self.retrieval = retrieval
        self.llm = llm
    
    def translate(self, text: str, top_k: int) -> TranslateResponse:
        terms = self.retrieval.search(text, top_k)
        user_prompt = build_user_prompt(text, terms.must_use, terms.reference)
        translation = self.llm.chat(system=SYSTEM_PROMPT, user=user_prompt)

        return TranslateResponse(
            query=text,
            translation=translation,
            must_use=terms.must_use,
            reference=terms.reference,
        )
    
@lru_cache
def get_translation_service() -> TranslationService:
    return TranslationService(
        retrieval=get_retrieval_service(),
        llm=OllamaClient(get_settings()),
    )

if __name__ == "__main__":
    service = get_translation_service()
    result = service.translate(
        "Press and hold the power button to restart the control hub.", top_k=5
    )
    print(result.model_dump_json(indent=2))