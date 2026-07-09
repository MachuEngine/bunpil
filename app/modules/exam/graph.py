import logging
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .llm import get_langchain_model
from .state import ExamState
from .tools import TOOLS, get_draft_items, get_judge_result, init_session, reset_judge

logger = logging.getLogger(__name__)


def _invoke_with_retry(llm, messages, retries: int = 2, delay: float = 2.0):
    """장시간 세션에서 간헐적으로 발생하는 Ollama 스트림 오류
    ("No data received from Ollama stream")를 흡수한다. 2026-07-10 발견 —
    TROUBLESHOOTING.md 참고. 프롬프트/모델 문제가 아니라 연결 자체의 일시적 오류라
    같은 요청을 그대로 재시도하면 대부분 해결됨."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            logger.warning("LLM invoke 실패(%d/%d), 재시도: %s", attempt + 1, retries + 1, e)
            if attempt < retries:
                time.sleep(delay)
    raise last_err


def plan_node(state: ExamState) -> dict:
    """세션을 초기화한다. 요청 전체에서 단 한 번만 호출됨 — 재시도 시에는 agent_node가
    문항을 유지한 채 reset_judge()만 호출한다(부분 진행을 재시도마다 버리지 않기 위함,
    2026-07-10 개선. 이전에는 agent_node가 매 재시도마다 init_session()으로 전체
    초기화를 했었음)."""
    init_session()
    return {
        "validation_passed": False,
        "similarity_judge_result": {},
        "error": "",
    }


def _build_system_prompt(passage_text: str, num_items: int, existing_items: list) -> str:
    no_text_rule = (
        "**매우 중요한 규칙**: 이 대화 내내 도구 호출(tool call) 외에는 어떤 텍스트도 "
        "출력하지 마세요. 인사, 생각 과정 설명, 진행 상황 서술, 문항 초안을 텍스트로 "
        "먼저 보여주는 것 모두 금지입니다. 매 턴 오직 도구 호출만 하세요."
    )
    remaining = max(0, num_items - len(existing_items))

    def _summary(items):
        return "\n".join(
            f"  {i+1}. [{it.get('item_type','?')}/{it.get('difficulty','?')}] {str(it.get('question',''))[:40]}"
            for i, it in enumerate(items)
        )

    if not existing_items:
        progress_note = ""
        count_instruction = (
            "예시 문제는 스타일·주제·난이도 참고용입니다. 문항 개수는 예시 개수와 무관하게 "
            f"지정된 개수({num_items}개)에 맞춰 작성하세요. 유형(객관식/서술형) 구성과 난이도 수준은 "
            "예시를 참고해 구성하되, 개수만은 반드시 지정된 개수를 따르세요."
        )
        action_instruction = "문항마다 다음 순서로 도구를 호출하세요:"
    elif remaining > 0:
        progress_note = (
            f"\n\n이전 시도에서 이미 다음 {len(existing_items)}개 문항을 작성해 저장했습니다"
            f"(다시 만들지 마세요, 그대로 유지됩니다):\n{_summary(existing_items)}\n"
        )
        count_instruction = (
            f"목표 개수는 {num_items}개이고 이미 {len(existing_items)}개가 있으므로, "
            f"나머지 {remaining}개만 새로 작성하세요. 기존 문항과 겹치지 않는 내용으로, "
            "예시 문제 스타일·난이도를 참고해 구성하세요."
        )
        action_instruction = f"나머지 {remaining}개 문항마다 다음 순서로 도구를 호출하세요:"
    else:
        # 개수는 이미 채워졌지만(remaining<=0) 구조 유사도 등 다른 기준으로 재시도된 경우.
        # 이미 만든 문항을 유지한 채 similarity_judge만 다시 호출하도록 안내한다
        # (개별 문항 수정/교체 기능은 아직 없음 — 알려진 한계, TROUBLESHOOTING.md 참고).
        progress_note = f"\n\n목표 개수({num_items}개)는 이미 채워져 있습니다:\n{_summary(existing_items)}\n"
        count_instruction = (
            "새 문항을 추가로 작성하지 마세요. 위 문항 세트를 검토한 뒤 "
            "similarity_judge 도구만 다시 호출해 구조적 유사도를 재평가하세요."
        )
        action_instruction = "바로 similarity_judge 도구를 호출하세요:"

    return (
        "당신은 한국 고등학교 사회 문항 출제 전문가 에이전트입니다. 한국어로만 응답하세요.\n\n"
        f"{no_text_rule}\n\n"
        "다음은 교사가 참고용으로 제시한 예시 문제입니다.\n\n"
        f"[예시 문제]\n{passage_text}\n"
        f"{progress_note}\n"
        f"{count_instruction}\n\n"
        f"{action_instruction}\n"
        "1. [선택] search_standards — 참고 성취기준 원문 확인\n"
        "2. [선택] search_regulations — 교육과정 준수 사항 확인\n"
        "3. validate_item_format — 직접 구성한 문항의 형식 검증\n"
        "   (오류가 있으면 수정 후 재검증, 통과할 때까지 반복)\n"
        "4. save_item — 검증 통과한 문항 저장\n"
        "5. record_score — 품질 자체 평가 (0~5점)\n\n"
        "문항 세트 작성이 모두 끝나면 similarity_judge 도구를 호출해 "
        "예시 문제와의 구조적 유사도(유형 비율·난이도 구성)를 스스로 평가하세요. "
        "(문항 개수 일치 여부는 이 도구가 아니라 시스템이 자동으로 검증합니다.)\n\n"
        "문항은 당신이 직접 작성합니다. "
        "객관식 선지는 반드시 ①②③④ 형식으로 4개 작성하세요.\n\n"
        "오답(정답이 아닌 선지)은 명백히 틀리거나 문제와 무관한 내용이 아니라, "
        "같은 개념 범주 안에서 학생이 실제로 헷갈릴 수 있는 그럴듯한 오답으로 구성하세요. "
        "예를 들어 정답이 '비례대표제'라면 오답은 '외계인 침공'처럼 무관한 선지가 아니라 "
        "'소선거구제', '직접 선거제'처럼 같은 주제의 인접 개념이어야 합니다.\n\n"
        f"{no_text_rule}"
    )


def agent_node(state: ExamState) -> dict:
    """ReAct 에이전트가 예시 문제를 분석해 문항 세트를 생성한다.

    2026-07-10 개선: 재시도마다 전체를 초기화하지 않는다. 이미 저장된 문항
    (get_draft_items())은 유지하고, 부족한 개수만 이어서 작성하도록 프롬프트를
    동적으로 구성한다(reset_judge()로 판정 결과만 초기화 — 누적된 문항 기준으로
    다시 판단해야 하므로).
    """
    reset_judge()

    spec = state["spec"]
    passage_text = spec.get("passage_text", "")
    standards = spec.get("standards") or []
    num_items = spec.get("num_items", 5)
    existing_items = get_draft_items()

    system_prompt = _build_system_prompt(passage_text, num_items, existing_items)
    user_content = "위 지침에 따라 문항을 작성하세요."
    if standards:
        user_content += f"\n\n참고 성취기준: {', '.join(standards)}"

    tool_map = {t.name: t for t in TOOLS}
    llm = get_langchain_model().bind_tools(TOOLS)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

    for _ in range(14):
        response = _invoke_with_retry(llm, messages)
        messages.append(response)

        if not getattr(response, "tool_calls", []):
            break

        judged = False
        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            if not fn:
                result_content = f"Unknown tool: {tc['name']}"
            else:
                try:
                    result_content = str(fn.invoke(tc["args"]))
                except Exception as e:
                    # 소형 LLM이 인자 타입을 틀리는 경우가 있어(예: 리스트 대신 문자열 필드에
                    # 리스트를 채움), 예외로 전체 루프를 죽이지 않고 에이전트가 스스로
                    # 고칠 수 있도록 오류를 도구 응답 형태로 되돌려준다.
                    result_content = f"도구 호출 오류 — 인자 형식을 확인하고 다시 호출하세요: {e}"
            messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))
            if tc["name"] == "similarity_judge":
                judged = True
        if judged:
            break

    return {
        "agent_messages": messages,
        "similarity_judge_result": get_judge_result(),
        "budget": state["budget"] - 1,
    }


def validate_node(state: ExamState) -> dict:
    """similarity_judge 결과를 threshold로 판정한다.
    count_match는 LLM 판단이 아니라 spec["num_items"] 기준으로 코드가 직접 계산한다
    (문항 개수는 예시 문제 개수와 무관하게 지정된 값을 따라야 하므로)."""
    judge = state.get("similarity_judge_result", {})
    draft_items = get_draft_items()
    count_match = len(draft_items) == state["spec"].get("num_items", 5)
    passed = (
        count_match
        and judge.get("type_ratio_score", 0) >= 0.7
        and judge.get("difficulty_match", False)
        and judge.get("overall_score", 0) >= 4
    )
    return {
        "draft_items": draft_items,
        "validation_passed": passed,
    }


def should_retry(state: ExamState) -> Literal["agent", "end"]:
    if state.get("validation_passed"):
        return "end"
    if state.get("budget", 0) > 0:
        return "agent"
    return "end"


def build_exam_graph():
    g = StateGraph(ExamState)
    g.add_node("plan", plan_node)
    g.add_node("agent", agent_node)
    g.add_node("validate", validate_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "agent")
    g.add_edge("agent", "validate")
    g.add_conditional_edges("validate", should_retry, {"agent": "agent", "end": END})

    return g.compile()


_exam_graph = None


def get_exam_graph():
    global _exam_graph
    if _exam_graph is None:
        _exam_graph = build_exam_graph()
    return _exam_graph
