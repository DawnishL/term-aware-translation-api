from typing import TypedDict

from app.schemas import TermHit

class TranslationState(TypedDict, total=False):
    """翻译流程中各节点共享的数据，每个节点读取需要的栏目，写回自己负责的栏目。"""
    text: str
    top_k: int
    must_use: list[TermHit]
    reference: list[TermHit]
    translation: str
    retries: int
    adherence: float
    missing: list[TermHit]