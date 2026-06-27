import contextvars
import logging
import threading
import uuid

logger = logging.getLogger(__name__)

from langchain_core.tools import tool

from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

# ── 세션 컨텍스트 ──
# _request_ctx: 요청별 독립 dict. asyncio.to_thread + contextvars.copy_context()로
# 요청 간 격리 보장. 같은 요청의 worker 스레드들은 동일 dict 객체를 공유하므로
# intra-request 가시성 유지 (GIL로 단순 list/dict 연산은 안전).
# last_id: 스레드별 분리 — 병렬 생성 시 레이스 컨디션 방지
_request_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("_request_ctx")
_thread_local = threading.local()


def _get_ctx() -> dict:
    return _request_ctx.get()


def init_session(collection: str) -> None:
    _request_ctx.set({
        "collection": collection,
        "items": [],
        "scores": {},
        "duplicates": {},
    })


def get_draft_items() -> list:
    ctx = _get_ctx()
    result = []
    for item in ctx["items"]:
        iid = item.get("item_id", "")
        score = ctx["scores"].get(iid, 0.0)
        dup = ctx["duplicates"].get(iid, False)
        result.append(
            {
                **item,
                "judge_score": score,
                "is_duplicate": dup,
                "status": "approved" if score >= 3 and not dup else "rejected",
            }
        )
    return result


# ── 싱글턴 인프라 ──
_store: RAGStore = None
_embedder: BGEEmbedder = None
_reranker: BGEReranker = None


def _get_store() -> RAGStore:
    global _store
    if _store is None:
        _store = RAGStore()
    return _store


def _get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder


def _get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker


# ── 도구 정의 ──
# 모든 도구는 LLM 호출 없이 순수 계산/검색/저장만 수행한다.
# 추론과 생성은 에이전트(LLM)가 직접 담당한다.

@tool
def search_passages(query: str) -> str:
    """성취기준 관련 내용을 검색합니다. query: 검색 키워드"""
    retriever = RAGRetriever(_get_store(), _get_embedder(), _get_reranker())
    results = []

    col = _get_ctx()["collection"]
    if col:
        results = retriever.retrieve(query, col, top_k=3)

    if _get_store().count("standards") > 0:
        std_results = retriever.retrieve(query, "standards", top_k=3)
        if std_results:
            all_texts = [r["text"] for r in results + std_results]
            ranked = _get_reranker().rerank(query, all_texts, top_k=3)
            results = [{"text": all_texts[r["index"]], "score": r["score"]} for r in ranked]

    if not results:
        return "관련 성취기준 없음"
    return "\n\n".join(f"[{i+1}] {r['text'][:400]}" for i, r in enumerate(results))


@tool
def search_regulations(query: str) -> str:
    """교육과정 법령·지침에서 관련 내용을 검색합니다. query: 검색 키워드"""
    count = _get_store().count("regulations")
    if count == 0:
        logger.warning("regulations 컬렉션이 비어있습니다.")
        return "교육과정 자료 없음"
    retriever = RAGRetriever(_get_store(), _get_embedder(), _get_reranker())
    results = retriever.retrieve(query, "regulations", top_k=3)
    if not results:
        return "관련 규정 없음"
    return "\n\n".join(f"[{i+1}] {r['text'][:300]}" for i, r in enumerate(results))


@tool
def get_past_item_examples(concept: str) -> str:
    """기출문제에서 유사 문항을 참조합니다. 출제 스타일 벤치마킹 및 차별화에 활용하세요."""
    count = _get_store().count("past_exams")
    if count == 0:
        logger.warning("past_exams 컬렉션이 비어있습니다.")
        return "기출문제 데이터 없음"
    retriever = RAGRetriever(_get_store(), _get_embedder(), _get_reranker())
    results = retriever.retrieve(concept, "past_exams", top_k=2)
    if not results:
        return "관련 기출문제 없음"
    return "\n\n".join(f"[기출 {i+1}] {r['text'][:400]}" for i, r in enumerate(results))


@tool
def validate_item_format(question: str, options: list, answer: str, item_type: str) -> str:
    """문항 형식을 검증합니다. 오류가 있으면 구체적인 수정 지침을 반환합니다.
    question: 문제 질문
    options: 선지 목록 (객관식: ["①...", "②...", "③...", "④..."], 서술형: [])
    answer: 정답 (객관식: "①"~"④", 서술형: "")
    item_type: 객관식|서술형
    """
    errors = []
    if not question or len(question.strip()) < 10:
        errors.append("질문이 너무 짧습니다 (10자 이상 필요)")
    if item_type == "객관식":
        if len(options) != 4:
            errors.append(f"선지는 4개여야 합니다 (현재 {len(options)}개)")
        marks = ["①", "②", "③", "④"]
        if answer not in marks:
            errors.append(f"정답은 ①②③④ 중 하나여야 합니다 (현재: '{answer}')")
        for i, opt in enumerate(options[:4]):
            if not str(opt).startswith(marks[i]):
                errors.append(f"선지 {i+1}번이 '{marks[i]}'로 시작해야 합니다")
                break
    if errors:
        return "형식 오류 — 수정 필요: " + " / ".join(errors)
    return "형식 검증 통과"


@tool
def save_item(question: str, options: list, answer: str, item_type: str, difficulty: str, standard: str = "") -> str:
    """검증된 문항을 저장합니다. 에이전트가 직접 작성한 내용을 저장합니다.
    question: 문제 질문
    options: 선지 목록 (객관식: ["①...", "②...", "③...", "④..."], 서술형: [])
    answer: 정답 (객관식: "①"~"④", 서술형: "")
    item_type: 객관식|서술형
    difficulty: 상|중|하
    standard: 성취기준명 (선택)
    """
    item_id = uuid.uuid4().hex[:8]
    item = {
        "item_id": item_id,
        "question": question,
        "options": options,
        "answer": answer,
        "item_type": item_type,
        "difficulty": difficulty,
        "standard": standard,
    }
    _thread_local.last_id = item_id
    _get_ctx()["items"].append(item)
    return f"저장 완료 (item_id={item_id})"


@tool
def record_score(score: int) -> str:
    """문항 품질 점수를 기록합니다. 에이전트가 직접 평가한 점수를 입력합니다.
    score: 0~5 (5=매우 우수, 4=우수, 3=보통, 2=미흡, 1=불량, 0=생성 실패)
    """
    item_id = getattr(_thread_local, "last_id", "")
    if item_id:
        _get_ctx()["scores"][item_id] = float(max(0, min(5, int(score))))
    return f"품질 점수 {score}/5 기록됨"


@tool
def check_duplicate(question: str) -> str:
    """기출 문제와 중복 여부를 확인합니다. 중복이면 True, 아니면 False 반환."""
    try:
        count = _get_store().count("past_exams")
        item_id = getattr(_thread_local, "last_id", "")
        if count == 0:
            logger.warning(
                "past_exams 컬렉션이 비어있습니다. "
                "scripts/index_past_exams.py를 실행한 뒤 다시 시도하세요."
            )
            if item_id:
                _get_ctx()["duplicates"][item_id] = False
            return "False"
        q_vec = _get_embedder().embed([question])[0]
        results = _get_store().query("past_exams", q_vec, n_results=min(3, count))
        if not results:
            if item_id:
                _get_ctx()["duplicates"][item_id] = False
            return "False"
        passages = [r["text"] for r in results]
        ranked = _get_reranker().rerank(question, passages, top_k=1)
        is_dup = ranked[0]["score"] > 0.8
        if item_id:
            _get_ctx()["duplicates"][item_id] = is_dup
        return str(is_dup)
    except Exception:
        logger.warning("check_duplicate 예외 발생", exc_info=True)
        item_id = getattr(_thread_local, "last_id", "")
        if item_id:
            _get_ctx()["duplicates"][item_id] = False
        return "False"


TOOLS = [
    search_passages,
    search_regulations,
    get_past_item_examples,
    validate_item_format,
    save_item,
    record_score,
    check_duplicate,
]
