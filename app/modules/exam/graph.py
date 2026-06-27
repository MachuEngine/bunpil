from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .llm import get_langchain_model
from .state import ExamState
from .tools import TOOLS, get_draft_items, init_session


def _build_target_pairs(spec: dict) -> list[tuple]:
    """(item_type, difficulty) 쌍의 목표 리스트를 생성한다.
    type_dist와 difficulty_dist를 순서대로 짝지으며, 길이가 다르면 짧은 쪽을 순환한다."""
    type_items: list[str] = []
    for itype, cnt in spec["type_dist"].items():
        type_items.extend([itype] * cnt)
    diff_items: list[str] = []
    for diff, cnt in spec["difficulty_dist"].items():
        diff_items.extend([diff] * cnt)
    n = max(len(type_items), len(diff_items))
    return [
        (type_items[i % len(type_items)], diff_items[i % len(diff_items)])
        for i in range(n)
    ]


def plan_node(state: ExamState) -> dict:
    """spec 분석, 세션 초기화, coverage_map 설정."""
    spec = state["spec"]
    standards = spec.get("standards") or [f"{spec['unit']} 핵심 개념 이해"]
    init_session(state["source_collection"])
    return {
        "coverage_map": {s: 0 for s in standards},
        "draft_items": [],
        "validation_passed": False,
        "error": "",
    }


def agent_node(state: ExamState) -> dict:
    """ReAct 에이전트로 문항을 병렬 생성한다.

    remaining 목록의 각 (type, difficulty) 쌍을 ThreadPoolExecutor로 동시에 처리한다.
    last_id / last_passage는 threading.local()로 스레드별 분리된다.
    """
    spec = state["spec"]
    standards = spec.get("standards") or [f"{spec['unit']} 핵심 개념 이해"]

    target_pairs = _build_target_pairs(spec)
    remaining = list(target_pairs)
    for it in get_draft_items():
        if it.get("status") == "approved":
            pair = (it.get("item_type", ""), it.get("difficulty", ""))
            if pair in remaining:
                remaining.remove(pair)

    if not remaining:
        return {"agent_messages": [], "budget": state["budget"] - 1}

    tool_map = {t.name: t for t in TOOLS}
    llm = get_langchain_model().bind_tools(TOOLS)

    def _run_item(itype: str, diff: str, std: str) -> list:
        passage_text = spec.get("passage_text", "")
        system_prompt = (
            "당신은 한국 고등학교 사회 문항 출제 전문가 에이전트입니다. 한국어로만 응답하세요.\n\n"
            "다음 순서대로 도구를 호출해 문항을 출제하세요:\n"
            "1. search_passages — 성취기준 관련 내용 검색\n"
            "2. [선택] get_past_item_examples — 기출 스타일 참조\n"
            "3. [선택] search_regulations — 교육과정 준수 사항 확인\n"
            "4. validate_item_format — 직접 구성한 문항의 형식 검증\n"
            "   (오류가 있으면 수정 후 재검증, 통과할 때까지 반복)\n"
            "5. save_item — 검증 통과한 문항 저장\n"
            "6. record_score — 품질 자체 평가 (0~5점)\n"
            "7. check_duplicate — 기출 중복 확인\n\n"
            "문항은 당신이 직접 작성합니다. "
            "객관식 선지는 반드시 ①②③④ 형식으로 4개 작성하세요."
        )
        user_content = (
            f"단원 '{spec['unit']}'에서 문항을 출제하세요.\n"
            f"유형: {itype}, 난이도: {diff}, 성취기준: {std}"
            + (f"\n\n[지문]\n{passage_text[:3000]}" if passage_text else "")
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        item_messages = []

        for _ in range(14):
            response = llm.invoke(messages)
            messages.append(response)
            item_messages.append(response)

            if not getattr(response, "tool_calls", []):
                break

            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                result_content = str(fn.invoke(tc["args"])) if fn else f"Unknown tool: {tc['name']}"
                tm = ToolMessage(content=result_content, tool_call_id=tc["id"])
                messages.append(tm)
                item_messages.append(tm)

        return item_messages

    all_messages = []
    with ThreadPoolExecutor(max_workers=len(remaining)) as executor:
        futures = {
            executor.submit(_run_item, itype, diff, standards[idx % len(standards)]): idx
            for idx, (itype, diff) in enumerate(remaining)
        }
        for future in as_completed(futures):
            all_messages.extend(future.result())

    return {
        "agent_messages": all_messages,
        "budget": state["budget"] - 1,
    }


def validate_node(state: ExamState) -> dict:
    """생성된 문항이 spec 제약을 만족하는지 검증한다."""
    spec = state["spec"]
    items = get_draft_items()
    approved = [it for it in items if it.get("status") == "approved"]

    # 유형 분포 충족 여부
    type_counts: dict = {}
    for it in approved:
        t = it.get("item_type", "")
        type_counts[t] = type_counts.get(t, 0) + 1
    type_ok = all(type_counts.get(k, 0) >= v for k, v in spec["type_dist"].items())

    # 난이도 분포 충족 여부
    diff_counts: dict = {}
    for it in approved:
        d = it.get("difficulty", "")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    diff_ok = all(diff_counts.get(k, 0) >= v for k, v in spec["difficulty_dist"].items())

    # 성취기준 커버리지
    standards = spec.get("standards") or []
    coverage_map = {s: 0 for s in standards}
    for it in approved:
        s = it.get("standard", "")
        if s in coverage_map:
            coverage_map[s] += 1
    coverage_ok = all(v > 0 for v in coverage_map.values()) if coverage_map else True

    passed = (
        len(approved) >= spec["num_items"]
        and type_ok
        and diff_ok
        and coverage_ok
    )

    return {
        "draft_items": items,
        "coverage_map": coverage_map,
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
