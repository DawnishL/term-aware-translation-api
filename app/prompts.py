from app.schemas import TermHit

SYSTEM_PROMPT = """你是一名专业的英译中技术翻译，负责翻译消费电子产品的说明文字。
请遵守以下规则：
1. 只输出中文译文，不要解释，不要加引号，不要重复原文。
2. “必须使用的术语”必须严格使用给定的中文译法。
3. “参考术语”仅供参考，只在原文确实表达了该意思时使用。
4. 译文要准确、通顺，符合中文技术文档的表达习惯。"""

def format_terms(hits: list[TermHit]) -> str:
    """把术语列表排成一行一条的文字，没有术语时返回“（无）”。"""
    if not hits:
        return "（无）"
    
    lines = [f"- {hit.source} -> {hit.target}" for hit in hits]
    return "\n".join(lines)

def build_user_prompt(text: str, must_use: list[TermHit], reference: list[TermHit]) -> str:
    """把原文和两组术语组装成发给模型的用户消息。"""
    return (
        f"必须使用的术语：\n{format_terms(must_use)}\n\n"
        f"参考术语：\n{format_terms(reference)}\n\n"
        f"原文：\n{text}\n\n"
        "译文："
    )

if __name__ == "__main__":
    from app.config import get_settings
    from app.retrieval.service import get_retrieval_service
    from app.llm import OllamaClient

    retrieval = get_retrieval_service()
    client = OllamaClient(get_settings())
    sentences = [
        "Press and hold the power button to restart the control hub.",
        "Replace the ear tips if they no longer fit.",
        "This device has an IP rating of IP67.",
    ]

    first = retrieval.search(sentences[0], top_k=5)
    print("===== 第一句发给模型的用户消息 =====")
    print(build_user_prompt(sentences[0], first.must_use, first.reference))
    print()

    for text in sentences:
        result = retrieval.search(text, top_k=5)
        plain = client.chat(system=SYSTEM_PROMPT, user=f"原文：\n{text}\n\n译文：")
        with_terms = client.chat(SYSTEM_PROMPT, 
                            build_user_prompt(text, result.must_use, result.reference))

        print("原文　　", text)
        print("无术语　", plain)
        print("有术语　", with_terms)
        print()