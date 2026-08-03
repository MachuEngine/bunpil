"""BM25 어휘 검색 — dense 검색이 놓치는 정확한 용어 매칭을 보완한다.

dense 검색(BGE-M3)은 "의미가 비슷한가"를 보기 때문에 동의어에는 강하지만,
법령 조항번호·고유명사처럼 **글자가 정확히 겹쳐야 하는 경우**엔 약하다. 실제로
생기부 규정 위반 탐지에서 "가정환경·종교 규정 청크가 검색 상위에 안 잡히는"
문제가 열린 이슈로 남아 있었고, 그동안은 검색을 고치는 대신 `chain.py`의
키워드 규칙으로 우회해왔다(EVAL.md 5절 "알려진 리스크").

BM25는 학습 모델이 아니라 공식이다 — 단어 빈도(tf), 희귀도(idf), 문서 길이
보정 세 가지만 계산한다. 그래서 **인덱싱 시점에 저장해둘 것이 전혀 없고**,
ChromaDB에 이미 들어있는 청크 텍스트만 읽어 메모리에 인덱스를 세운다
(재인덱싱·스키마 변경 불필요). BGE-M3의 sparse_vecs를 쓰는 대안도 검토했으나
재인덱싱 + 메타데이터 스키마 변경 + 모델 버전 결합이 필요해 채택하지 않았다.

토큰화는 **BGE-M3에 딸린 토크나이저**(XLM-R sentencepiece)를 그대로 빌려 쓴다.
한국어를 공백으로 자르면 조사가 붙어버려("가정환경이"와 "가정환경"이 다른 토큰)
매칭이 무너지는데, 이 토크나이저는 조사를 별도 토큰으로 분리한다:

    '가정환경'         -> ['▁가정', '환경']
    '가정환경이'       -> ['▁가정', '환경', '이']
    '가정형편이 어려운' -> ['▁가정', '형', '편', '이', '▁어려운']

덕분에 kiwipiepy·mecab 같은 형태소 분석기 의존성을 추가하지 않아도 되고,
"가정형편"과 "가정환경"이 `▁가정` 토큰을 공유해 서로 검색된다.
"""
import math
from collections import Counter, defaultdict

# BM25 표준 튜닝 상수. k1은 "같은 단어가 여러 번 나올 때 점수가 얼마나 빨리
# 포화되는가"(클수록 천천히 포화), b는 "긴 문서에 얼마나 페널티를 줄
# 것인가"(0이면 길이 무시, 1이면 최대 보정). 문헌 표준값을 그대로 쓴다 —
# 이 코퍼스에 맞춘 튜닝은 골든셋이 n=22로 작아 과적합 위험이 크다.
_K1 = 1.5
_B = 0.75


class BM25Index:
    """청크 텍스트로 만든 BM25 역색인. 컬렉션 하나당 인스턴스 하나.

    docs: [{"id": str, "text": str, "metadata": dict}, ...]
    tokenize: 텍스트를 토큰 리스트로 바꾸는 함수 (BGEEmbedder.tokenize)
    """

    def __init__(self, docs: list[dict], tokenize, k1: float = _K1, b: float = _B):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._by_id = {d["id"]: d for d in docs}

        # 역색인: 단어 -> [(문서 번호, 그 문서에서의 등장 횟수), ...]
        # 질문에 나온 단어만 훑으면 되므로 전체 문서를 매번 스캔할 필요가 없다.
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []

        for idx, doc in enumerate(docs):
            tokens = tokenize(doc["text"])
            self._doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((idx, tf))

        self.n = len(docs)
        self.avgdl = (sum(self._doc_len) / self.n) if self.n else 0.0

        # idf: 그 단어를 포함한 문서가 적을수록 높다 — "학생"처럼 모든 규정에
        # 나오는 단어는 변별력이 없으니 낮게, "가정환경"처럼 몇 개 청크에만
        # 나오는 단어는 겹쳤을 때 진짜 신호이므로 높게 친다.
        # log(1 + ...) 형태를 쓰는 이유: 절반 이상의 문서에 나오는 흔한 단어에서
        # idf가 음수가 되어 점수를 깎아버리는 것을 막기 위함(BM25+ 관례).
        self._idf = {
            term: math.log(1 + (self.n - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def get(self, doc_id: str) -> dict | None:
        return self._by_id.get(doc_id)

    def top_ids(self, query_tokens: list[str], n: int) -> list[str]:
        """질문 토큰과 겹치는 문서를 BM25 점수 내림차순으로 최대 n개 반환."""
        if not self.n or not self.avgdl:
            return []

        scores: dict[int, float] = defaultdict(float)
        # set()으로 중복 제거 — 질문에 같은 단어가 두 번 나와도 한 번만 센다
        # (질문 쪽 tf는 BM25 기본형에서 쓰지 않음).
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for idx, tf in postings:
                # 길이 보정: 평균보다 긴 문서일수록 분모가 커져 점수가 깎인다.
                norm = 1 - self.b + self.b * (self._doc_len[idx] / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [self.docs[idx]["id"] for idx, _ in ranked]
