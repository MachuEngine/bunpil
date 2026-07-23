#!/usr/bin/env python
"""tools.py의 search_standards/search_regulations가 쓰는 top_k(현재 3)를 2로
낮췄을 때 검색 품질(Recall)이 얼마나 나빠지는지 측정.

BGE 임베딩/리랭킹만 쓰고 LLM(Ollama)과 무관 — eval_exam.py의 Recall@5(top_k=5)와는
다른 값으로, 실제 에이전트가 도구 호출 시 받는 top_k(2 또는 3) 기준으로 직접
Recall@k를 계산한다. retrieval_golden_final.json(reviewed=true, n=21) 사용.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "retrieval_golden_final.json")


def load_golden() -> list[dict]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [item for item in data if item.get("reviewed")]


def eval_recall_at_k(retriever: RAGRetriever, golden: list, top_k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    for item in golden:
        col = item["source_collection"]
        results = retriever.retrieve(item["query"], col, top_k=top_k, n_candidates=20)
        preview = item["chunk_preview"].strip()
        found_rank = None
        for rank, r in enumerate(results, 1):
            if preview and preview[:80] in r["text"]:
                found_rank = rank
                break
        if found_rank is not None:
            hits += 1
            rr_sum += 1.0 / found_rank
    n = len(golden)
    return {"top_k": top_k, "recall": hits / n, "mrr": rr_sum / n, "n": n}


def main() -> None:
    golden = load_golden()
    store = RAGStore()
    embedder = BGEEmbedder()
    reranker = BGEReranker()
    retriever = RAGRetriever(store, embedder, reranker)

    print(f"=== top_k별 Recall 비교 (n={len(golden)}, tools.py 현재값=3) ===\n")
    results = {}
    for k in (2, 3, 5):
        r = eval_recall_at_k(retriever, golden, k)
        results[f"top_k={k}"] = r
        print(f"top_k={k}: Recall={r['recall']:.3f}, MRR={r['mrr']:.3f}")

    gap = results["top_k=3"]["recall"] - results["top_k=2"]["recall"]
    print(f"\ntop_k 3→2 축소 시 Recall 하락폭: {gap:+.3f}")
    if gap >= 0.05:
        print("→ 하락폭 0.05 이상: top_k 축소 보류 권장(Recall 손실 유의미)")
    else:
        print("→ 하락폭 0.05 미만: top_k=2로 축소해도 무방할 수준")

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_topk_recall_compare.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
