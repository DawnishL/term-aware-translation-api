from functools import lru_cache
from app.config import get_settings
from app.schemas import TranslateResponse
from app.llm import OllamaClient
from app.prompts import SYSTEM_PROMPT, build_user_prompt, build_retry_prompt
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.checking import adherence_rate, find_missing_terms

MAX_RETRIES = 2

class TranslationService:
    def __init__(self, retrieval: RetrievalService, llm: OllamaClient, max_retries=MAX_RETRIES) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.retries = max_retries
    
    def translate(self, text: str, top_k: int) -> TranslateResponse:
        terms = self.retrieval.search(text, top_k)
        must_use = terms.must_use
        reference = terms.reference
        user_prompt = build_user_prompt(text, terms.must_use, terms.reference)
        translation = self.llm.chat(system=SYSTEM_PROMPT, user=user_prompt)
        missing = find_missing_terms(translation=translation, must_use=must_use)

        retries = 0
        while missing and retries < self.retries:
            retry_prompt = build_retry_prompt(
                text=text,
                must_use=must_use,
                reference=reference,
                previous_translation=translation,
                missing=missing,
            )
            translation = self.llm.chat(system=SYSTEM_PROMPT, user=retry_prompt)
            missing = find_missing_terms(translation, must_use)
            retries += 1

        return TranslateResponse(
            query=text,
            translation=translation,
            must_use=must_use,
            reference=reference,
            adherence=adherence_rate(missing, must_use),
            retries=retries,
            missing=missing,
        )

    
@lru_cache
def get_translation_service() -> TranslationService:
    return TranslationService(
        retrieval=get_retrieval_service(),
        llm=OllamaClient(get_settings()),
    )

if __name__ == "__main__":

    class FakeLLM:
        """假的模型，第一次故意漏用术语，之后才用对，用来测试重译逻辑。"""

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, system: str, user: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "长按电源按钮以重启控制中枢。"
            return "长按开关键以重启多功能网关。"

    text = "Press and hold the power button to restart the control hub."

    print("===== 假模型，测试重译逻辑 =====")
    fake_service = TranslationService(retrieval=get_retrieval_service(), llm=FakeLLM())
    print(fake_service.translate(text, top_k=3).model_dump_json(indent=2))

    print("===== 真模型 =====")
    real_service = get_translation_service()
    print(real_service.translate(text, top_k=3).model_dump_json(indent=2))