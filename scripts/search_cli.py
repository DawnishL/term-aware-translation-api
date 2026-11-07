from app.retrieval.matching import ExactMatcher
from app.retrieval.embeddings import get_embedder
from app.retrieval.glossary import load_glossary
from app.retrieval.store import TermStore
from app.retrieval.service import merge_results

TOP_K = 5

def main():
    print("正在加载模型和术语表，请稍等……")
    glossary = load_glossary()
    matcher = ExactMatcher(glossary)
    store = TermStore(get_embedder())
    store.rebuild(glossary)
    print("准备好了。输入英文原文查询，直接回车退出。")

    while True:
        text = input("\n原文> ").strip()

        if not text:
            break

        must_use = matcher.find(text)
        vector_hits = store.search(text, top_k=TOP_K + len(must_use))
        
        print("必须使用")
        if not must_use:
            print("无")
        for entry in must_use:
            print(f"{entry['source']} -> {entry['target']}")

        print("参考术语")
        reference = merge_results(must_use, vector_hits, TOP_K)
        for entry in reference:
            print(f"{entry['score']:.3f}  {entry['source']} -> {entry['target']}")

if __name__ == "__main__":
    main()