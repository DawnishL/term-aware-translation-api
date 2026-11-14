import sys
from pathlib import Path

from app.config import get_settings
from app.llm import OllamaClient
from app.retrieval.service import get_retrieval_service
from app.schemas import TranslateResponse
from app.translation import TranslationService

SENTENCES_PATH = Path("data/test_sentences.txt")
TOP_K = 5


def load_sentences(path: Path) -> list[str]:
    """读测试句子，一行一句，跳过空行。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def count_used(result: TranslateResponse) -> int:
    """数一下用上了几个必须使用的术语。"""
    return len(result.must_use) - len(result.missing)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SENTENCES_PATH
    sentences = load_sentences(path)

    retrieval = get_retrieval_service()
    llm = OllamaClient(get_settings())
    baseline = TranslationService(retrieval=retrieval, llm=llm, max_retries=0)
    full = TranslationService(retrieval=retrieval, llm=llm, max_retries=2)

    counted = 0
    total_terms = 0
    used_before = 0
    used_after = 0
    perfect_before = 0
    perfect_after = 0
    retried = 0

    for number, sentence in enumerate(sentences, start=1):
        before = baseline.translate(sentence, top_k=TOP_K)
        after = full.translate(sentence, top_k=TOP_K)

        print(f"\n[{number}] {sentence}")
        if not before.must_use:
            print("　没有找到必须使用的术语，不计入统计，请换一句")
            continue

        print("　必须使用", [f"{t.source} -> {t.target}" for t in before.must_use])
        print(f"　重译前　{before.translation}　遵循率 {before.adherence:.2f}")
        print(
            f"　重译后　{after.translation}　遵循率 {after.adherence:.2f}"
            f"　重译 {after.retries} 次"
        )
        if after.missing:
            print("　仍然漏掉", [t.target for t in after.missing])

        counted += 1
        total_terms += len(before.must_use)
        used_before += count_used(before)
        used_after += count_used(after)
        if not before.missing:
            perfect_before += 1
        if not after.missing:
            perfect_after += 1
        if after.retries > 0:
            retried += 1

    print("\n===== 汇总 =====")
    if total_terms == 0:
        print("没有可以统计的句子")
        return

    print(f"计入统计的句子　{counted} 句")
    print(f"必须使用的术语　{total_terms} 个")
    print(
        f"术语遵循率　　　重译前 {used_before / total_terms:.1%}"
        f"　重译后 {used_after / total_terms:.1%}"
    )
    print(f"全部用对的句子　重译前 {perfect_before} 句　重译后 {perfect_after} 句")
    print(f"触发重译的句子　{retried} 句")


if __name__ == "__main__":
    main()