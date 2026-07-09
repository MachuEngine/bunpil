from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .llm import get_langchain_model
from .state import ExamState
from .tools import TOOLS, get_draft_items, get_judge_result, init_session


def plan_node(state: ExamState) -> dict:
    """초기 상태를 리셋한다. 세션 초기화는 재시도마다 반복되어야 하므로 agent_node가 담당한다."""
    return {
        "validation_passed": False,
        "similarity_judge_result": {},
        "error": "",
    }


def agent_node(state: ExamState) -> dict:
    """ReAct 에이전트가 예시 문제를 분석해 문항 세트 전체를 한 번에 생성한다.

    세트 전체 단위로 재시도하므로(should_retry), 매 호출마다 세션을 초기화해
    이전 시도의 문항이 이번 결과에 섞이지 않도록 한다.
    """
    init_session()

    spec = state["spec"]
    passage_text = spec.get("passage_text", "")
    standards = spec.get("standards") or []
    num_items = spec.get("num_items", 5)

    system_prompt = (
        "당신은 한국 고등학교 사회 문항 출제 전문가 에이전트입니다. 한국어로만 응답하세요.\n\n"
        "다음은 교사가 참고용으로 제시한 예시 문제입니다.\n\n"
        f"[예시 문제]\n{passage_text}\n\n"
        "예시 문제는 스타일·주제·난이도 참고용입니다. 문항 개수는 예시 개수와 무관하게 "
        f"지정된 개수({num_items}개)에 맞춰 작성하세요. 유형(객관식/서술형) 구성과 난이도 수준은 "
        "예시를 참고해 구성하되, 개수만은 반드시 지정된 개수를 따르세요.\n\n"
        "문항마다 다음 순서로 도구를 호출하세요:\n"
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
        "도구 호출 외의 설명 텍스트는 쓰지 마세요. 생각 과정이나 진행 상황을 문장으로 "
        "서술하지 말고, 곧바로 다음 도구를 호출하세요."
    )
    user_content = "위 지침에 따라 예시 문제와 동일한 구성의 문항 세트를 작성하세요."
    if standards:
        user_content += f"\n\n참고 성취기준: {', '.join(standards)}"

    tool_map = {t.name: t for t in TOOLS}
    llm = get_langchain_model().bind_tools(TOOLS)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

    for _ in range(14):
        response = llm.invoke(messages)
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
