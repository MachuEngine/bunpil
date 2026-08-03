"""app/common/rag/lexical.py — BM25 순수 로직 유닛테스트.

모델·ChromaDB 호출 없음. 토크나이저는 BGE-M3의 실제 동작을 흉내낸 가짜 함수를
쓴다(공백 분리 + 조사 분리) — 실제 BGE-M3 토크나이저를 로드하면 CI에서 모델
다운로드가 필요해지고, BM25 공식 자체의 정확성 검증엔 불필요하기 때문.
"""
import pytest

from app.common.rag.lexical import BM25Index


def fake_tokenize(text: str) -> list[str]:
    """공백으로 자른 뒤 흔한 조사를 떼어낸다 — BGE-M3 토크나이저가 하는 일의 축소판."""
    tokens = []
    for word in text.split():
        for josa in ("이란", "은", "는", "이", "가", "을", "를", "의", "에"):
            if len(word) > len(josa) and word.endswith(josa):
                tokens.append(word[: -len(josa)])
                tokens.append(josa)
                break
        else:
            tokens.append(word)
    return tokens


DOCS = [
    {"id": "d1", "text": "가정환경 경제적 지위는 기재하지 않는다", "metadata": {"source": "규정.pdf"}},
    {"id": "d2", "text": "종교 정치성향 관련 내용은 기재하지 않는다", "metadata": {"source": "규정.pdf"}},
    {"id": "d3", "text": "봉사활동 실적은 일자와 장소를 기재한다", "metadata": {"source": "규정.pdf"}},
    {"id": "d4", "text": "교과 학습 발달상황을 기재한다", "metadata": {"source": "규정.pdf"}},
]


@pytest.fixture
def index():
    return BM25Index(DOCS, fake_tokenize)


def test_finds_document_by_shared_term(index):
    """질문 단어를 포함한 문서가 1순위로 나온다."""
    assert index.top_ids(fake_tokenize("가정환경"), 5)[0] == "d1"
    assert index.top_ids(fake_tokenize("종교"), 5)[0] == "d2"


def test_josa_does_not_break_matching(index):
    """'가정환경이'처럼 조사가 붙어도 어간이 분리되어 매칭된다 — 한국어 검색의 핵심 요구사항."""
    assert index.top_ids(fake_tokenize("가정환경이"), 5)[0] == "d1"


def test_rare_term_outranks_common_term(index):
    """희귀 단어(idf 높음)가 흔한 단어보다 순위 결정에 크게 기여한다.

    '기재하지'는 4개 중 3개 문서에 나오는 흔한 단어라 변별력이 없고,
    '종교'는 1개 문서에만 나온다 — 둘 다 포함한 질문에서 d2가 1등이어야 한다.
    """
    ranked = index.top_ids(fake_tokenize("종교 기재하지"), 5)
    assert ranked[0] == "d2"


def test_no_match_returns_empty(index):
    assert index.top_ids(fake_tokenize("존재하지않는단어xyz"), 5) == []


def test_respects_n_limit(index):
    """겹치는 문서가 여러 개여도 요청한 개수까지만 반환한다."""
    ranked = index.top_ids(fake_tokenize("기재하지 기재한다"), 2)
    assert len(ranked) <= 2


def test_empty_corpus_is_safe():
    """빈 컬렉션에서 0으로 나누지 않는다."""
    empty = BM25Index([], fake_tokenize)
    assert empty.top_ids(fake_tokenize("아무거나"), 5) == []


def test_get_returns_document(index):
    assert index.get("d1")["text"].startswith("가정환경")
    assert index.get("없는id") is None


def test_idf_is_never_negative(index):
    """모든 문서에 나오는 단어여도 idf가 음수가 되어 점수를 깎으면 안 된다.

    표준 BM25 idf는 df > N/2에서 음수가 되는데, 이 구현은 log(1 + ...) 형태라
    항상 양수다. 음수가 되면 '흔한 단어를 포함한 문서일수록 순위가 내려가는'
    직관에 반하는 동작이 생긴다.
    """
    all_docs_term = BM25Index(
        [
            {"id": "a", "text": "공통 단어 하나", "metadata": {}},
            {"id": "b", "text": "공통 단어 둘", "metadata": {}},
        ],
        fake_tokenize,
    )
    assert all(v > 0 for v in all_docs_term._idf.values())
