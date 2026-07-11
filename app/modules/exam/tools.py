import contextvars
import logging
import re
import threading
import uuid

logger = logging.getLogger(__name__)

from langchain_core.tools import tool

from app.common.rag import get_retriever, get_store

# ── 세션 컨텍스트 ──
# _request_ctx: 요청별 독립 dict. asyncio.to_thread + contextvars.copy_context()로
# 요청 간 격리 보장. 같은 요청의 worker 스레드들은 동일 dict 객체를 공유하므로
# intra-request 가시성 유지 (GIL로 단순 list/dict 연산은 안전).
# last_id: 스레드별 분리 — 병렬 생성 시 레이스 컨디션 방지
_request_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("_request_ctx")
_thread_local = threading.local()

_HANGUL_RE = re.compile(r"[가-힣]")
_HAN_RE = re.compile(r"[一-鿿]")  # CJK 한자 (중국어 오염 검출용)


def _get_ctx() -> dict:
    return _request_ctx.get()


def init_session(passage_text: str = "") -> None:
    # LangGraph는 각 노드를 context.run()으로 격리 실행하므로
    # plan_node 내에서 set()한 새 dict가 agent_node에 전파되지 않는다.
    # 해결: asyncio.to_thread 호출 전 main.py에서 먼저 set()으로 dict를 생성하고,
    # 이후 호출(plan_node)에서는 같은 dict를 in-place로 초기화해 모든 노드가 공유한다.
    # passage_text: save_item의 원문 복사 게이트가 참조 (plan_node가 spec에서 전달)
    try:
        ctx = _request_ctx.get()
        ctx["items"] = []
        ctx["scores"] = {}
        ctx["judge_result"] = {}
        ctx["passage_text"] = passage_text
    except LookupError:
        _request_ctx.set({
            "items": [],
            "scores": {},
            "judge_result": {},
            "passage_text": passage_text,
        })


def get_draft_items() -> list:
    ctx = _get_ctx()
    result = []
    for item in ctx["items"]:
        iid = item.get("item_id", "")
        score = ctx["scores"].get(iid, 0.0)
        result.append(
            {
                **item,
                "judge_score": score,
                "status": "approved" if score >= 3 else "rejected",
            }
        )
    return result


def get_judge_result() -> dict:
    return _get_ctx().get("judge_result", {})


def reset_judge() -> None:
    """재시도 시 문항(items/scores)은 유지하고 similarity_judge 결과만 초기화한다.
    이전 시도(더 적은 문항 기준)의 판정이 이번 재시도(누적된 문항 기준) 결과에
    잘못 남아있지 않도록 한다. 문항 자체를 지우지 않는 것이 init_session()과의 차이."""
    _get_ctx()["judge_result"] = {}


# ── 도구 정의 ──
# 모든 도구는 LLM 호출 없이 순수 계산/검색/저장만 수행한다.
# 추론과 생성은 에이전트(LLM)가 직접 담당한다.

@tool
def search_regulations(query: str) -> str:
    """교육과정 법령·지침에서 관련 내용을 검색합니다. query: 검색 키워드"""
    count = get_store().count("regulations")
    if count == 0:
        logger.warning("regulations 컬렉션이 비어있습니다.")
        return "교육과정 자료 없음"
    results = get_retriever().retrieve(query, "regulations", top_k=3)
    if not results:
        return "관련 규정 없음"
    return "\n\n".join(f"[{i+1}] {r['text'][:300]}" for i, r in enumerate(results))


@tool
def search_standards(query: str) -> str:
    """성취기준 관련 내용을 사회과 교육과정(2022 개정) standards 컬렉션에서 검색합니다.
    query: 검색 키워드 (예: 성취기준명)"""
    count = get_store().count("standards")
    if count == 0:
        logger.warning("standards 컬렉션이 비어있습니다.")
        return "교육과정 성취기준 자료 없음"
    results = get_retriever().retrieve(query, "standards", top_k=3)
    if not results:
        return "관련 성취기준 없음"
    return "\n\n".join(f"[{i+1}] {r['text'][:400]}" for i, r in enumerate(results))


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


def _check_korean(question: str, options: list, answer: str) -> str | None:
    """문항 텍스트가 한국어인지 결정론적으로 검사한다. 통과하면 None, 아니면 거부 사유 반환.

    qwen2.5:7b가 컨텍스트가 길어지면 확률적으로 중국어 문항을 생성하는 문제가 있어
    (2026-07-11 발견, TROUBLESHOOTING.md 참고) 저장 전에 코드가 차단한다.
    한자 비율 5% 미만은 허용 — 정당한 괄호 병기(예: 사법(私法))까지 막지 않기 위함."""
    text = " ".join([str(question), str(answer), *[str(o) for o in options]])
    hangul = len(_HANGUL_RE.findall(text))
    han = len(_HAN_RE.findall(text))
    if hangul == 0:
        return "저장 거부 — 문항에 한국어가 없습니다. 모든 내용을 한국어로 작성한 뒤 다시 저장하세요."
    if han and han / (han + hangul) >= 0.05:
        return "저장 거부 — 문항에 중국어가 포함되어 있습니다. question·options·answer 전체를 한국어로 다시 작성한 뒤 저장하세요."
    return None


# ── 유사도 게이트 ──
# qwen2.5:7b가 예시 문제를 그대로 복사하거나 세트 안에 같은 문항을 반복 생성하는 문제
# (2026-07-11, 사람 라벨 20건 중 최다 감점 사유)의 결정론적 차단. 7B Judge는 rubric을
# 줘도 텍스트 동일성 대조를 못 해내서(EVAL.md 5절) 코드로 이관함.
# 임계값 근거: 라벨링된 골든셋 실측 분포 — 완전 복사는 containment 1.00, 정상적인 주제
# 유사 변형은 ~0.73 이하 / 진짜 중복은 jaccard 0.86~1.00, 정상 변형은 ~0.67 이하.
_SIMILARITY_STRIP_RE = re.compile(r"[\s\d①②③④.,?!()\[\]·:;'\"—\-~%]")
_PASSAGE_COPY_THRESHOLD = 0.90
_DUPLICATE_JACCARD_THRESHOLD = 0.80


def _bigrams(text: str) -> set:
    t = _SIMILARITY_STRIP_RE.sub("", str(text))
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _check_similarity(question: str) -> str | None:
    """예시 문제 원문 복사·세트 내 중복을 검사한다. 통과하면 None, 아니면 거부 사유 반환."""
    qb = _bigrams(question)
    if len(qb) < 8:  # 극단적으로 짧은 질문은 판정 불가 — 길이 검증은 validate_item_format 몫
        return None
    ctx = _get_ctx()
    passage = ctx.get("passage_text", "")
    if passage:
        pb = _bigrams(passage)
        if pb and len(qb & pb) / len(qb) >= _PASSAGE_COPY_THRESHOLD:
            return ("저장 거부 — 이 문항은 예시 문제를 거의 그대로 복사한 것입니다. "
                    "예시는 참고만 하고, 같은 주제라도 질문·선지를 새로 구성해 다시 저장하세요.")
    for existing in ctx["items"]:
        eb = _bigrams(existing.get("question", ""))
        union = qb | eb
        if union and len(qb & eb) / len(union) >= _DUPLICATE_JACCARD_THRESHOLD:
            return ("저장 거부 — 이미 저장된 문항과 사실상 동일합니다. "
                    "다른 개념이나 다른 관점을 묻는 새 문항을 작성해 저장하세요.")
    return None


@tool
def save_item(question: str, options: list, answer: str, item_type: str, difficulty: str = "중", standard: str = "") -> str:
    """검증된 문항을 저장합니다. 에이전트가 직접 작성한 내용을 저장합니다.
    (다음 문항은 저장이 거부됩니다 — 한국어가 아닌 문항, 예시 문제를 그대로 복사한 문항,
    이미 저장된 문항과 동일한 문항. 거부 시 안내에 따라 새로 작성해 재시도하세요.)
    question: 문제 질문
    options: 선지 목록 (객관식: ["①...", "②...", "③...", "④..."], 서술형: [])
    answer: 정답 (객관식: "①"~"④", 서술형: "")
    item_type: 객관식|서술형
    difficulty: 상|중|하
    standard: 성취기준명 (선택)
    """
    rejection = _check_korean(question, options, answer) or _check_similarity(question)
    if rejection:
        return rejection
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
def similarity_judge(
    type_ratio_score: float,
    difficulty_match: bool,
    overall_score: int,
) -> str:
    """예시 문제와 방금 작성한 문항 세트의 구조적 유사도를 기록합니다.
    문항 세트 작성을 모두 마친 뒤, 스스로 판단한 평가 결과를 인자로 전달해 호출하세요.
    (통과/재시도 여부는 이 도구가 아니라 이후 로직이 threshold로 결정합니다.
    문항 개수 일치 여부는 이 도구가 아니라 코드가 자동으로 검증합니다.)
    type_ratio_score: 유형(객관식/서술형) 구성 비율의 유사도 (0.0~1.0)
    difficulty_match: 난이도 수준 구성이 예시 문제와 부합하는가
    overall_score: 종합 평가 점수 (0~5, 5=매우 유사)
    """
    result = {
        "type_ratio_score": float(max(0.0, min(1.0, type_ratio_score))),
        "difficulty_match": bool(difficulty_match),
        "overall_score": int(max(0, min(5, overall_score))),
    }
    _get_ctx()["judge_result"] = result
    return f"구조 유사도 평가 기록됨: {result}"


TOOLS = [
    search_regulations,
    search_standards,
    validate_item_format,
    save_item,
    record_score,
    similarity_judge,
]
