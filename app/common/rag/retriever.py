import os
from collections import defaultdict

from .embedder import BGEEmbedder
from .lexical import BM25Index
from .reranker import BGEReranker
from .store import RAGStore

# RRF 상수. 순위 1등과 2등의 점수 차이를 얼마나 완만하게 볼 것인가를 정한다.
# 60은 RRF 원논문(Cormack et al., 2009) 이후 관례적으로 쓰는 기본값 —
# 상위권 순위 차이를 과하게 벌리지 않아 두 검색기의 합의를 잘 반영한다.
_RRF_K = 60


def _rrf(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion — 여러 검색 결과의 "등수"만 보고 합친다.

    dense 점수(코사인 거리)와 BM25 점수는 스케일이 전혀 달라서 그냥 더할 수 없다.
    RRF는 점수를 버리고 **등수의 역수**만 더하기 때문에 스케일 정규화가 필요 없다 —
    3등이면 1/(60+3)을 더하는 식. 두 검색기 모두에서 상위에 오른 문서가 자연히
    가장 높은 합산 점수를 받는다.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return scores


class RAGRetriever:
    """2단계 검색: (dense [+ BM25 융합]) 후보 추출 → reranker 재정렬.

    hybrid=None이면 환경변수 `RAG_HYBRID`를 따른다(기본 true — `false`로 두면
    이전의 dense 단독 검색으로 되돌아간다).
    eval 스크립트가 같은 프로세스에서 on/off를 바꿔가며 A/B 비교할 수 있도록
    생성자 인자로도 직접 지정할 수 있게 열어뒀다.
    """

    def __init__(
        self,
        store: RAGStore,
        embedder: BGEEmbedder,
        reranker: BGEReranker,
        hybrid: bool | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        # 기본값 true — 2026-08-03 A/B 측정에서 회귀 없이 개선만 확인돼 채택
        # (전체 MRR 0.789→0.814, standards MRR 0.892→0.938, 후보 포함률 9/10→10/10).
        # 되돌리려면 RAG_HYBRID=false. 근거는 MODEL_SELECTION.md 5절.
        self.hybrid = (
            os.getenv("RAG_HYBRID", "true").lower() == "true" if hybrid is None else hybrid
        )
        # 컬렉션별 BM25 인덱스 캐시. 값은 (청크 수, 인덱스) — 재인덱싱으로 청크 수가
        # 바뀌면 낡은 인덱스를 버리고 다시 만든다(가벼운 무효화 장치).
        self._bm25: dict[str, tuple[int, BM25Index]] = {}

    def _get_bm25(self, collection_name: str) -> BM25Index | None:
        count = self.store.count(collection_name)
        if count == 0:
            return None
        cached = self._bm25.get(collection_name)
        if cached and cached[0] == count:
            return cached[1]
        docs = self.store.all_documents(collection_name)
        if not docs:
            return None
        index = BM25Index(docs, self.embedder.tokenize)
        self._bm25[collection_name] = (count, index)
        return index

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        # 20 → 10 (2026-08-03 실측 근거). 후보를 늘리면 리랭커가 오히려 나빠진다:
        # 골든 22건 전수 비교에서 10이 20보다 3건 개선·0건 악화(Recall@5 0.955→1.000,
        # regulations MRR 0.667→0.753)였고, 리랭커가 채점할 쌍이 절반이라 약 2배 빠르다.
        # bge-reranker-base가 후보가 많아질수록 오답을 상위로 잘못 올리는 것으로 보인다
        # (EVAL.md 13절). 생기부(chain.py)는 원래부터 10을 명시해 쓰고 있었다.
        n_candidates: int = 10,
    ) -> list[dict]:
        query_vec = self.embedder.embed([query])[0]
        candidates = self.store.query(collection_name, query_vec, n_results=n_candidates)

        if self.hybrid:
            candidates = self._fuse(query, collection_name, candidates, n_candidates)

        if not candidates:
            return []
        passages = [c["text"] for c in candidates]
        ranked = self.reranker.rerank(query, passages, top_k=top_k)
        return [
            {
                "text": candidates[r["index"]]["text"],
                "metadata": candidates[r["index"]]["metadata"],
                "score": r["score"],
            }
            for r in ranked
        ]

        """
        candidates = [
            {"text": "사회계약론은 홉스, 로크, 루소가...", "metadata": {"source": "정치.pdf"}, "distance": 0.15},  # index 0
            {"text": "삼권분립은 입법·행정·사법을...", "metadata": {"source": "정치.pdf"}, "distance": 0.22},      # index 1
            {"text": "기본권은 자유권, 평등권...",      "metadata": {"source": "헌법.pdf"}, "distance": 0.31},      # index 2
        ]

        passages = [
            "사회계약론은 홉스, 로크, 루소가...",   # index 0
            "삼권분립은 입법·행정·사법을...",      # index 1
            "기본권은 자유권, 평등권...",         # index 2
        ]

        ranked = [
            {"index": 0, "score": 0.95},   # candidates[0]이 가장 관련 높다고 재평가됨
            {"index": 2, "score": 0.60},   # candidates[2]가 두 번째
        ]

        return = [
            {
                candidates[0]["text"]      # "사회계약론은 홉스, 로크, 루소가..."
                candidates[0]["metadata"]  # {"source": "정치.pdf"}
                r["score"]                 # 0.95
            },
            {
                candidates[2]["text"]      # "기본권은 자유권, 평등권..."
                candidates[2]["metadata"]  # {"source": "헌법.pdf"}
                r["score"]                 # 0.60
            }
        ]

        """

    def _fuse(
        self,
        query: str,
        collection_name: str,
        dense_hits: list[dict],
        n_candidates: int,
    ) -> list[dict]:
        """dense 후보와 BM25 후보를 RRF로 합쳐 상위 n_candidates개를 돌려준다.

        BM25는 dense 후보를 재정렬하는 게 아니라 **컬렉션 전체를 독립적으로 훑는다** —
        해결하려는 문제가 "정답 청크가 dense 후보에 아예 안 들어오는 것"이라,
        dense 결과 안에서만 순위를 바꿔서는 의미가 없기 때문이다.
        """
        index = self._get_bm25(collection_name)
        if index is None:
            return dense_hits

        lexical_ids = index.top_ids(self.embedder.tokenize(query), n_candidates)
        if not lexical_ids:
            return dense_hits

        dense_ids = [c["id"] for c in dense_hits]
        fused = _rrf([dense_ids, lexical_ids])

        by_id = {c["id"]: c for c in dense_hits}
        merged: list[dict] = []
        for doc_id, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            hit = by_id.get(doc_id)
            if hit is None:
                # BM25만 찾아낸 문서 — dense 후보엔 없으므로 인덱스에서 본문을 가져온다.
                # distance는 dense 검색을 안 거쳤다는 뜻으로 None을 둔다(리랭커는
                # 본문만 쓰므로 이후 단계에 영향 없음).
                doc = index.get(doc_id)
                if doc is None:
                    continue
                hit = {"id": doc_id, "text": doc["text"], "metadata": doc["metadata"], "distance": None}
            merged.append(hit)
            if len(merged) >= n_candidates:
                break
        return merged
