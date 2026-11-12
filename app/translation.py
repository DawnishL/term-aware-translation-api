from functools import lru_cache
from app.config import get_settings
from app.schemas import TranslateResponse
from app.llm import OllamaClient
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.graph.workflow import build_translation_graph

MAX_RETRIES = 2

class TranslationService:
    def __init__(self, retrieval: RetrievalService, llm: OllamaClient, max_retries=MAX_RETRIES) -> None:
        self.graph = build_translation_graph(retrieval=retrieval, llm=llm, max_retries=max_retries)
    
    def translate(self, text: str, top_k: int) -> TranslateResponse:
        final_state = self.graph.invoke({"text": text, "top_k": top_k})
        return TranslateResponse(
            query=text,
            translation=final_state["translation"],
            must_use=final_state["must_use"],
            reference=final_state["reference"],
            adherence=final_state["adherence"],
            retries=final_state["retries"],
            missing=final_state["missing"],
        )

@lru_cache
def get_translation_service() -> TranslationService:
    return TranslationService(
        retrieval=get_retrieval_service(),
        llm=OllamaClient(get_settings()),
    )

if __name__ == "__main__":

    class FakeLLM:
        """假的模型，第一次故意漏用术语，之后才用对。"""

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, system: str, user: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "长按电源按钮以重启控制中枢。"
            return "长按开关键以重启多功能网关。"

    service = TranslationService(retrieval=get_retrieval_service(), llm=FakeLLM())
    result = service.translate(
        "Press and hold the power button to restart the control hub.", top_k=3
    )
    print(result.model_dump_json(indent=2))