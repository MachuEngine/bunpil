#!/usr/bin/env python
"""'예시 문제 문장' 쿼리 vs 기존 '주제어' 쿼리의 standards 검색 정합성 비교.

배경: retrieval_golden_final.json의 쿼리는 성취기준 문서 문체(주제어/서술형)를
따라 만들어졌다. 실제로는 교사가 실제 시험 문제 문장(구체적 질문형, passage_text)을
그대로 검색에 사용할 수 있는데, 이 문체가 standards 컬렉션(성취기준 해설 문체)과
얼마나 잘 매칭되는지 검증된 적이 없다.

data/golden/example_question_retrieval_test.json의 각 항목은 실제 문제 문장
스타일 query를 담고 있으나, expected_chunk_id/chunk_preview 라벨링은
사람이 직접 한다(reviewed: false인 동안은 eval_retrieval() 채점에서 자동 제외).

라벨링을 돕기 위해, reviewed 여부와 무관하게 모든 쿼리의 top-5 검색 후보를
출력한다 — 정답 판정은 사람이 하고, 이 스크립트는 후보만 보여준다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from eval_lib import _load_retrieval_golden, eval_retrieval
from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

_NEW_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "golden", "example_question_retrieval_test.json"
)


def _load_example_golden() -> list[dict]:
    with open(_NEW_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def print_candidates(retriever: RAGRetriever, golden: list[dict]) -> None:
    print("\n--- 라벨링용 top-5 후보 (사람이 직접 정답 판정할 것) ---")
    for item in golden:
        results = retriever.retrieve(item["query"], item["source_collection"], top_k=5, n_candidates=20)
        print(f"\n[{item['id']}] query: {item['query']}")
        for rank, r in enumerate(results, 1):
            source = r["metadata"].get("source", "?")
            preview = r["text"][:120].replace("\n", " ")
            print(f"  {rank}. ({source}, score={r['score']:.3f}) {preview}")


def main() -> None:
    store = RAGStore()
    embedder = BGEEmbedder()
    reranker = BGEReranker()
    retriever = RAGRetriever(store, embedder, reranker)

    baseline_golden = _load_retrieval_golden()
    example_golden = _load_example_golden()
    example_reviewed = [item for item in example_golden if item.get("reviewed")]

    print("=== 예시 문제 → 성취기준 검색 정합성 비교 ===\n")

    print(f"1. 기존 골든셋(주제어 쿼리, n={len(baseline_golden)})")
    baseline_result = eval_retrieval(retriever, baseline_golden)
    print(f"   Recall@5={baseline_result['recall_at_5']:.3f}, MRR={baseline_result['mrr']:.3f}")

    print(f"\n2. 신규 골든셋(실제 문제 문장 쿼리, 라벨링 완료 n={len(example_reviewed)}/{len(example_golden)})")
    if not example_reviewed:
        print("   라벨링된 항목 없음 — expected_chunk_id/chunk_preview를 채우고 reviewed:true로 바꾼 뒤 재실행하세요.")
        example_result = None
    else:
        example_result = eval_retrieval(retriever, example_reviewed)
        print(f"   Recall@5={example_result['recall_at_5']:.3f}, MRR={example_result['mrr']:.3f}")

    if example_result is not None:
        recall_gap = baseline_result["recall_at_5"] - example_result["recall_at_5"]
        mrr_gap = baseline_result["mrr"] - example_result["mrr"]
        print(f"\n3. 격차: Recall@5 {recall_gap:+.3f}, MRR {mrr_gap:+.3f}")
        if recall_gap >= 0.1 or mrr_gap >= 0.1:
            print("   → 0.1 이상 격차: 문체 격차가 실제 문제라는 근거로 기록 권장")
        else:
            print("   → 격차 작음: 문체 격차 우려는 기각, 다른 원인(청킹 등)에 집중 권장")

    print_candidates(retriever, example_golden)


if __name__ == "__main__":
    main()
