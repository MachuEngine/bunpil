"""evals/eval_lib.py eval_retrieval()·eval_retrieval_per_query() 집계 일치 단위 테스트.

LLM/API 호출 없음 — 가짜 retriever(고정 결과를 돌려주는 duck-typed 객체)로만 검증한다.

evals/eval_exam.py main()은 2026-10부터 eval_retrieval()을 따로 호출하지 않고
eval_retrieval_per_query()의 질의 단위 hit/rr에서 Recall@5·MRR을 직접 집계한다(같은
질의를 두 번 검색하지 않기 위함) — 이 테스트는 그 집계식이 eval_retrieval()의 결과와
항상 같은 값을 내는지 확인한다.
"""
from evals.eval_lib import eval_retrieval, eval_retrieval_per_query


class _FakeRetriever:
    """golden 항목의 query에 미리 정해둔 결과 목록을 돌려주는 가짜 retriever.
    retrieve() 호출 횟수를 세어 둔다(이 테스트에서는 단언하지 않지만, 두 함수를 각각
    호출하면 호출 수가 배로 늘어난다는 점을 재현할 수 있게 둔다)."""

    def __init__(self, results_by_query: dict[str, list[dict]]):
        self._results_by_query = results_by_query
        self.call_count = 0

    def retrieve(self, query: str, collection_name: str, top_k: int = 5):
        self.call_count += 1
        return self._results_by_query.get(query, [])


def _aggregate_from_per_query(per_query: list[dict]) -> dict:
    """eval_exam.py main()이 쓰는 집계식(eval_retrieval()과 같은 계산) 재현."""
    n = len(per_query)
    return {
        "recall_at_5": sum(r["hit"] for r in per_query) / n,
        "mrr": sum(r["rr"] for r in per_query) / n,
        "n": n,
    }


def _golden():
    return [
        {"id": "q1", "query": "hit-rank-1", "source_collection": "standards", "chunk_preview": "정답 청크"},
        {"id": "q2", "query": "hit-rank-3", "source_collection": "standards", "chunk_preview": "정답 청크"},
        {"id": "q3", "query": "miss", "source_collection": "standards", "chunk_preview": "정답 청크"},
    ]


def _results_by_query():
    return {
        "hit-rank-1": [{"text": "정답 청크가 포함된 문서"}, {"text": "무관한 문서"}],
        "hit-rank-3": [{"text": "무관한 문서1"}, {"text": "무관한 문서2"}, {"text": "정답 청크가 포함된 문서"}],
        "miss": [{"text": "전혀 무관한 문서"}],
    }


def test_eval_retrieval_and_per_query_aggregation_match():
    golden = _golden()

    agg_retriever = _FakeRetriever(_results_by_query())
    direct = eval_retrieval(agg_retriever, golden)

    per_query_retriever = _FakeRetriever(_results_by_query())
    per_query = eval_retrieval_per_query(per_query_retriever, golden)
    derived = _aggregate_from_per_query(per_query)

    assert derived == direct
    assert direct["recall_at_5"] == 2 / 3
    assert direct["mrr"] == (1.0 + 1 / 3 + 0.0) / 3
    assert direct["n"] == 3
