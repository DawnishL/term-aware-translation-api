from app.checking import find_missing_terms, adherence_rate
from app.graph.state import TranslationState
from app.llm import OllamaClient
from app.prompts import build_retry_prompt, build_user_prompt, SYSTEM_PROMPT
from app.retrieval.service import RetrievalService

class TranslationNodes:
    """翻译流程的四个节点和一个判断函数，需要的服务从外面交进来。"""
    def __init__(self, retrieval: RetrievalService, llm: OllamaClient, max_retries: int) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.max_retries = max_retries
    
    def retrieve(self, state: TranslationState) -> dict:
        """检索术语，并把重译次数设为 0。"""
        terms = self.retrieval.search(state["text"], state["top_k"])
        return {
            "must_use": terms.must_use,
            "reference": terms.reference,
            "retries": 0
        }
    
    def translate(self, state: TranslationState) -> dict:
        """第一次用普通 prompt 翻译，之后带着上次的译文和漏用的术语重译。"""

        if "translation" not in state:
            user_prompt = build_user_prompt(
                state["text"], 
                state["must_use"],
                state["reference"],)
            translation = self.llm.chat(SYSTEM_PROMPT, user_prompt)
            return {"translation": translation}


        retry_prompt = build_retry_prompt(
            state["text"], 
            state["must_use"],
            state["reference"],
            state["translation"],
            state["missing"]
        )
        translation = self.llm.chat(SYSTEM_PROMPT, retry_prompt)
        return {
            "translation": translation,
            "retries": state["retries"] + 1
        }
    
    def check(self, state: TranslationState) -> dict:
        """找出译文里漏用的必须使用术语。"""
        missing = find_missing_terms(state["translation"], state["must_use"])
        return {"missing": missing}
    
    def should_retry(self, state: TranslationState) -> dict:
        """检查之后决定下一步，返回下一个节点的名字。"""
        if state["missing"] and state["retries"] < self.max_retries:
            return "translate"
        return "output"
    
    def output(self, state: TranslationState) -> dict:
        """计算术语遵循率。"""
        return {"adherence": adherence_rate(state["must_use"], state["missing"])}


if __name__ == "__main__":
    from app.retrieval.service import get_retrieval_service

    class FakeLLM:
        """假的模型，第一次故意漏用术语，之后才用对。"""

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, system: str, user: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "长按电源按钮以重启控制中枢。"
            return "长按开关键以重启多功能网关。"

    nodes = TranslationNodes(get_retrieval_service(), FakeLLM(), max_retries=2)

    state = {
        "text": "Press and hold the power button to restart the control hub.",
        "top_k": 3,
    }

    state.update(nodes.retrieve(state))
    print("检索后　必须使用", [t.target for t in state["must_use"]], "重译次数", state["retries"])

    state.update(nodes.translate(state))
    print("翻译后　", state["translation"])

    state.update(nodes.check(state))
    print("检查后　漏掉", [t.target for t in state["missing"]], "下一步", nodes.should_retry(state))

    state.update(nodes.translate(state))
    print("重译后　", state["translation"], "重译次数", state["retries"])

    state.update(nodes.check(state))
    print("检查后　漏掉", [t.target for t in state["missing"]], "下一步", nodes.should_retry(state))

    state.update(nodes.output(state))
    print("输出　　遵循率", state["adherence"])