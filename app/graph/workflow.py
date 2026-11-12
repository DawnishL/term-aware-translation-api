from langgraph.graph import START, END, StateGraph

from app.graph.state import TranslationState
from app.graph.nodes import TranslationNodes
from app.llm import OllamaClient
from app.retrieval.service import RetrievalService

def build_translation_graph(
        retrieval: RetrievalService, llm: OllamaClient, max_retries: int
):
    """把检索、翻译、检查、输出四个节点连成流程，编译后交出去。"""
    nodes = TranslationNodes(
        retrieval=retrieval, llm=llm, max_retries=max_retries
    )

    graph = StateGraph(TranslationState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("translate", nodes.translate)
    graph.add_node("check", nodes.check)
    graph.add_node("output", nodes.output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "translate")
    graph.add_edge("translate", "check")

    graph.add_conditional_edges(
        "check",
        nodes.should_retry,
        {"translate": "translate", "output": "output"}
    )

    graph.add_edge("output", END)

    return graph.compile()

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

    retrieval = get_retrieval_service()
    start_state = {
        "text": "Press and hold the power button to restart the control hub.",
        "top_k": 3,
    }

    print("第一部分　看流程经过了哪些节点")
    graph = build_translation_graph(retrieval, FakeLLM(), max_retries=2)
    for step in graph.stream(start_state, stream_mode="updates"):
        for node_name in step:
            print("　经过", node_name)

    print("第二部分　直接拿最终结果")
    graph = build_translation_graph(retrieval, FakeLLM(), max_retries=2)
    final_state = graph.invoke(start_state)
    print("　译文", final_state["translation"])
    print("　重译次数", final_state["retries"])
    print("　漏掉", [t.target for t in final_state["missing"]])
    print("　遵循率", final_state["adherence"])

    print("第三部分　流程图")
    print(graph.get_graph().draw_mermaid())