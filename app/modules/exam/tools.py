import asyncio
import concurrent.futures
import json
import logging
import uuid

from typing import Any

logger = logging.getLogger(__name__)

from langchain_core.tools import tool

from app.common.llm import PromptTemplate, get_llm_backend
from app.common.rag import BGEEmbedder, BGEReranker, RAGRetriever, RAGStore

# ── 세션 컨텍스트 (단일 사용자 시스템이므로 module-level dict 사용) ──
_ctx: dict = {
    "collection": "",
    "items": [],       # list[dict] — generate_item이 추가
    "scores": {},      # item_id -> float
    "duplicates": {},  # item_id -> bool
    "last_id": "",
}


def init_session(collection: str) -> None:
    _ctx["collection"] = collection
    _ctx["items"] = []
    _ctx["scores"] = {}
    _ctx["duplicates"] = {}
    _ctx["last_id"] = ""
    _ctx["last_passage"] = ""


def get_draft_items() -> list:
    result = []
    for item in _ctx["items"]:
        iid = item.get("item_id", "")
        score = _ctx["scores"].get(iid, 0.0)
        dup = _ctx["duplicates"].get(iid, False)
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


def _run_async(coro):
    """async coroutine을 동기 컨텍스트에서 안전하게 실행한다."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=300)


# ── 프롬프트 템플릿 ──
_GENERATE_TPL = PromptTemplate(
    system=(
        "한국 고등학교 사회 문항 출제 전문가입니다. "
        "JSON 형식으로만 응답하세요.\n"
        "형식: {\"question\":\"...\",\"options\":[\"①...\",\"②...\",\"③...\",\"④...\"],\"answer\":\"①\","
        "\"item_type\":\"객관식\",\"difficulty\":\"중\",\"standard\":\"...\"}\n"
        "서술형은 options를 []로 설정하세요."
    ),
    few_shots=[
        {
            "user": "유형:객관식 난이도:중 성취기준:민주주의이해 지문:민주주의는 국민이 주권을 갖는다.",
            "assistant": (
                '{"question":"민주주의의 핵심 원리는?","options":["①국민주권","②국가주권","③왕정","④군주제"],'
                '"answer":"①","item_type":"객관식","difficulty":"중","standard":"민주주의이해"}'
            ),
        }
    ],
)

_JUDGE_TPL = PromptTemplate(
    system="문항 품질을 0~5점으로 평가하세요. 숫자 하나만 응답하세요.",
    few_shots=[
        {
            "user": '{"question":"민주주의 핵심은?","options":["①국민주권","②왕정","③독재","④귀족"],"answer":"①"}',
            "assistant": "4",
        }
    ],
)


# ── 도구 정의 ──

@tool
def search_passages(query: str) -> str:
    """지문에서 관련 내용을 검색합니다. query: 검색 키워드"""
    col = _ctx["collection"]
    if not col:
        return "컬렉션이 설정되지 않았습니다."
    retriever = RAGRetriever(_get_store(), _get_embedder(), _get_reranker())

    # 업로드 임시 컬렉션 검색
    results = retriever.retrieve(query, col, top_k=3)

    # standards 영구 컬렉션이 있으면 함께 검색하고 재랭킹
    if _get_store().count("standards") > 0:
        std_results = retriever.retrieve(query, "standards", top_k=3)
        if std_results:
            all_results = results + std_results
            all_texts = [r["text"] for r in all_results]
            ranked = _get_reranker().rerank(query, all_texts, top_k=3)
            results = [{"text": all_texts[r["index"]], "score": r["score"]} for r in ranked]

    if not results:
        return "관련 지문 없음"
    combined = "\n\n".join(f"[{i+1}] {r['text'][:400]}" for i, r in enumerate(results))
    _ctx["last_passage"] = results[0]["text"]  # generate_item fallback용
    return combined


@tool
def generate_item(item_type: str, difficulty: str, standard: str = "", passage: str = "") -> str:
    """문항을 생성합니다.
    item_type: 객관식|서술형
    difficulty: 상|중|하
    standard: 성취기준명 (선택)
    passage: 참조 지문 (선택)
    """
    if not passage:
        passage = _ctx.get("last_passage", "")
    prompt = f"유형:{item_type} 난이도:{difficulty} 성취기준:{standard} 지문:{passage[:300]}"
    messages = _GENERATE_TPL.build(prompt)
    raw = _run_async(get_llm_backend().generate(messages, max_tokens=400))

    item = {}
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 and e > s:
            item = json.loads(raw[s:e])
    except Exception:
        pass

    item_id = uuid.uuid4().hex[:8]
    item["item_id"] = item_id
    item.setdefault("item_type", item_type)
    item.setdefault("difficulty", difficulty)
    item.setdefault("standard", standard)
    item.setdefault("question", raw[:200])
    item.setdefault("options", [])
    item.setdefault("answer", "")

    _ctx["last_id"] = item_id
    _ctx["items"].append(item)
    return json.dumps(item, ensure_ascii=False)


@tool
def judge_item(question_json: Any) -> str:
    """문항 품질을 0~5점으로 평가합니다. generate_item의 반환값(JSON 문자열 또는 dict)을 입력으로 주세요."""
    if isinstance(question_json, dict):
        question_json = json.dumps(question_json, ensure_ascii=False)
    elif not isinstance(question_json, str):
        question_json = str(question_json)
    messages = _JUDGE_TPL.build(question_json)
    raw = _run_async(get_llm_backend().generate(messages, max_tokens=10))
    score = 3.0
    for ch in raw.strip():
        if ch in "012345":  # ASCII 숫자 0-5만 허용
            score = float(ch)
            break
    if _ctx["last_id"]:
        _ctx["scores"][_ctx["last_id"]] = score
    return f"점수: {int(score)}/5"


@tool
def check_duplicate(question: str) -> str:
    """기출 문제와 중복 여부를 확인합니다. 중복이면 True, 아니면 False 반환."""
    try:
        count = _get_store().count("past_exams")
        if count == 0:
            logger.warning(
                "past_exams 컬렉션이 비어있습니다. "
                "scripts/index_past_exams.py를 실행한 뒤 다시 시도하세요."
            )
            if _ctx["last_id"]:
                _ctx["duplicates"][_ctx["last_id"]] = False
            return "False"
        q_vec = _get_embedder().embed([question])[0]
        results = _get_store().query("past_exams", q_vec, n_results=min(3, count))
        if not results:
            if _ctx["last_id"]:
                _ctx["duplicates"][_ctx["last_id"]] = False
            return "False"
        passages = [r["text"] for r in results]
        ranked = _get_reranker().rerank(question, passages, top_k=1)
        is_dup = ranked[0]["score"] > 0.8
        if _ctx["last_id"]:
            _ctx["duplicates"][_ctx["last_id"]] = is_dup
        return str(is_dup)
    except Exception:
        if _ctx["last_id"]:
            _ctx["duplicates"][_ctx["last_id"]] = False
        return "False"


TOOLS = [search_passages, generate_item, judge_item, check_duplicate]
