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

def build_context(text: str, must_use: list[TermHit], reference: list[TermHit]) -> str:
    return (
        f"必须使用的术语：\n{format_terms(must_use)}\n\n"
        f"参考术语：\n{format_terms(reference)}\n\n"
        f"原文：\n{text}"
    )

def build_user_prompt(text: str, must_use: list[TermHit], reference: list[TermHit]) -> str:
    """把原文和两组术语组装成发给模型的用户消息。"""
    return f"{build_context(text, must_use, reference)}\n\n译文："

def build_retry_prompt(text: str, 
                       must_use: list[TermHit], 
                       reference: list[TermHit], 
                       previous_translation: str, 
                       missing: list[TermHit]) -> str:
    """重译时发给模型的用户消息，附上上一次的译文和漏用的术语。"""
    return (
        f"{build_context(text, must_use, reference)}\n\n"
        f"上一次的译文：\n{previous_translation}\n\n"
        "上一次的译文漏用了以下必须使用的术语。"
        "请在保持原意和通顺的前提下修改译文，确保使用这些译法：\n"
        f"{format_terms(missing)}\n\n"
        "修改后的译文："
    )
    

if __name__ == "__main__":
    from app.checking import find_missing_terms
    from app.config import get_settings
    from app.llm import OllamaClient

    text = "Press and hold the power button to restart the control hub."
    must_use = [
        TermHit(source="power button", target="开关键", match_type="exact"),
        TermHit(source="control hub", target="多功能网关", match_type="exact"),
    ]
    reference: list[TermHit] = []
    previous = "长按电源按钮以重启控制中枢。"

    missing = find_missing_terms(previous, must_use)
    retry_prompt = build_retry_prompt(text, must_use, reference, previous, missing)

    print("===== 重译时发给模型的用户消息 =====")
    print(retry_prompt)
    print()

    client = OllamaClient(get_settings())
    revised = client.chat(system=SYSTEM_PROMPT, user=retry_prompt)
    print(revised)
    print("仍然漏掉：", [term.target for term in find_missing_terms(revised, must_use)])