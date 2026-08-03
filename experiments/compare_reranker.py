#!/usr/bin/env python
"""리랭커 조사 — 하이브리드 검색 도입(2026-08-03) 후 드러난 병목을 정량화한다.

배경(EVAL.md 12절): BM25 융합으로 dense가 후보에조차 못 올리던 `ret_015`를
후보 12위로 찾아냈으나, `bge-reranker-base`가 top-5로 올리지 못해 최종
Recall@5는 그대로였다. 즉 병목이 검색기 → 리랭커로 옮겨갔다.

MODEL_SELECTION.md 4절의 열린 질문("리랭커의 실제 기여도가 한 번도 정량
측정된 적 없음")도 여기서 함께 답한다.

세 가지를 측정한다 (전부 LLM 미사용 · 결정론적):
  A. 리랭커 ablation — 리랭커를 아예 빼고 융합 순서를 그대로 쓰면?
  B. n_candidates 스윕 — 후보를 더 많이 주면 리랭커가 정답을 건져 올리나?
  C. (선택) 리랭커 모델 교체 — BGE_RERANK_MODEL 환경변수로 지정
     예: BGE_RERANK_MODEL=BAAI/bge-reranker-v2-m3 python experiments/compare_reranker.py

실행:
    CHROMA_PERSIST_DIR=./chroma_db python experiments/compare_reranker.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "golden", "retrieval_golden_final.json"
)


def load_golden() -> list[dict]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def _rank_of_answer(results: list[dict], preview: str) -> int | None:
    """정답 청크가 결과 몇 번째에 있는지(1-base). 없으면 None."""
    anchor = preview.strip()[:80]
    if not anchor:
        return None
    for rank, r in enumerate(results, 1):
        if anchor in r["text"]:
            return rank
    return None


def score(golden: list, retrieve_fn, top_k: int = 5) -> dict:
    """retrieve_fn(query, collection) -> list[dict] 를 받아 Recall@5·MRR 계산."""
    hits = 0
    rr_sum = 0.0
    per_item: dict[str, int | None] = {}
    for item in golden:
        results = retrieve_fn(item["query"], item["source_collection"])
        rank = _rank_of_answer(results[:top_k], item["chunk_preview"])
        per_item[item["id"]] = rank
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
    n = len(golden)
    return {
        "n": n,
        "recall_at_5": round(hits / n, 3) if n else 0.0,
        "mrr": round(rr_sum / n, 3) if n else 0.0,
        "per_item": per_item,
    }


def main() -> None:
    golden = load_golden()
    regs = [g for g in golden if g["source_collection"] == "regulations"]
    print(f"골든셋: 전체 {len(golden)}건 (regulations {len(regs)}건)")
    print(f"리랭커 모델: {os.getenv('BGE_RERANK_MODEL', 'BAAI/bge-reranker-base')}\n", flush=True)

    store, embedder, reranker = RAGStore(), BGEEmbedder(), BGEReranker()
    hybrid = RAGRetriever(store, embedder, reranker, hybrid=True)
    dense = RAGRetriever(store, embedder, reranker, hybrid=False)

    def fused_only(query: str, collection: str, n_candidates: int = 20) -> list[dict]:
        """리랭커를 빼고 RRF 융합 순서를 그대로 반환 (ablation용)."""
        qv = embedder.embed([query])[0]
        candidates = store.query(collection, qv, n_results=n_candidates)
        return hybrid._fuse(query, collection, candidates, n_candidates)

    def dense_only_no_rerank(query: str, collection: str, n_candidates: int = 20) -> list[dict]:
        qv = embedder.embed([query])[0]
        return store.query(collection, qv, n_results=n_candidates)

    rows = []

    # ── A. 리랭커 ablation ────────────────────────────────────────────
    print("[A] 리랭커 ablation — 리랭커를 빼면?", flush=True)
    configs = [
        ("dense + 리랭커 (2026-07 이전 기본)", lambda q, c: dense.retrieve(q, c, top_k=5, n_candidates=20)),
        ("dense, 리랭커 없음", dense_only_no_rerank),
        ("hybrid + 리랭커 (현재 기본)", lambda q, c: hybrid.retrieve(q, c, top_k=5, n_candidates=20)),
        ("hybrid, 리랭커 없음", fused_only),
    ]
    for label, fn in configs:
        total = score(golden, fn)
        reg = score(regs, fn)
        rows.append((label, total, reg))
        print(
            f"  {label:32s} | 전체 R@5={total['recall_at_5']:.3f} MRR={total['mrr']:.3f}"
            f" | reg R@5={reg['recall_at_5']:.3f} MRR={reg['mrr']:.3f}"
            f" | ret_015={reg['per_item'].get('ret_015')}",
            flush=True,
        )

    # ── B. n_candidates 스윕 ──────────────────────────────────────────
    print("\n[B] n_candidates 스윕 (hybrid + 리랭커) — 후보를 늘리면 리랭커가 건져 올리나?", flush=True)
    for n_cand in (10, 20, 30, 50):
        fn = lambda q, c, n=n_cand: hybrid.retrieve(q, c, top_k=5, n_candidates=n)
        total = score(golden, fn)
        reg = score(regs, fn)
        print(
            f"  n_candidates={n_cand:3d} | 전체 R@5={total['recall_at_5']:.3f} MRR={total['mrr']:.3f}"
            f" | reg R@5={reg['recall_at_5']:.3f} MRR={reg['mrr']:.3f}"
            f" | ret_015={reg['per_item'].get('ret_015')}",
            flush=True,
        )

    print(
        "\n※ ret_015 = 정답 청크의 최종 순위(None이면 top-5 밖). "
        "dense 단독으로는 후보 20개 안에도 못 들어오던 항목 — EVAL.md 10·12절 참고."
    )


if __name__ == "__main__":
    main()
