#!/usr/bin/env python
"""재시도 시 부분 진행 보존(2026-07-10 개선) 전/후 비교.

OLD: 재시도마다 init_session()으로 전체 초기화(이전 시도 문항 폐기), 매번 동일한
"목표 num_items개를 처음부터 작성하라" 프롬프트.
NEW: app/modules/exam/graph.py의 실제 agent_node — 재시도 시 기존 문항 유지,
"나머지 N개만" 프롬프트로 이어서 작성.

num_items가 클수록(5~10개) 한 번에 다 맞추기 어려워 재시도 이득이 클 것이라는
가설로, num_items>=5인 샘플만 사용한다. 지표: exact_match_rate
(len(draft_items)==num_items), budget 소진까지 걸린 시도 횟수.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.exam.graph import _invoke_with_retry
from app.modules.exam.llm import get_langchain_model
from app.modules.exam.tools import TOOLS, get_draft_items, init_session

from gen_structure_golden import PASSAGE_SAMPLES

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_retry_preservation_compare.json")

BUDGET = 3
TEST_SAMPLES = [s for s in PASSAGE_SAMPLES if s["num_items"] >= 5]


def _old_system_prompt(passage_text: str, num_items: int) -> str:
    """2026-07-10 이전 방식 재현 — 재시도 여부와 무관하게 매번 동일, 기존 문항 인지 없음."""
    return (
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
        "예시 문제와의 구조적 유사도(유형 비율·난이도 구성)를 스스로 평가하세요.\n\n"
        "문항은 당신이 직접 작성합니다. "
        "객관식 선지는 반드시 ①②③④ 형식으로 4개 작성하세요.\n\n"
        "**매우 중요한 규칙**: 이 대화 내내 도구 호출(tool call) 외에는 어떤 텍스트도 "
        "출력하지 마세요. 매 턴 오직 도구 호출만 하세요."
    )


def run_old(sample: dict, budget: int) -> dict:
    """매 시도마다 전체 초기화 — 마지막 시도의 결과만 최종값으로 남는다(2026-07-09 이전 동작)."""
    tool_map = {t.name: t for t in TOOLS}
    final_items = []
    for attempt in range(budget):
        init_session()  # 전체 초기화 — 이전 시도 문항 폐기
        llm = get_langchain_model().bind_tools(TOOLS)
        system_prompt = _old_system_prompt(sample["passage_text"], sample["num_items"])
        user_content = "위 지침에 따라 예시 문제와 동일한 구성의 문항 세트를 작성하세요."
        if sample.get("standards"):
            user_content += f"\n\n참고 성취기준: {', '.join(sample['standards'])}"
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
                        result_content = f"도구 호출 오류: {e}"
                messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))
                if tc["name"] == "similarity_judge":
                    judged = True
            if judged:
                break
        final_items = get_draft_items()
        if len(final_items) == sample["num_items"]:
            break
    return {"id": sample["id"], "num_items": sample["num_items"], "generated": len(final_items),
            "exact_match": len(final_items) == sample["num_items"]}


def run_new(sample: dict, budget: int) -> dict:
    """graph.py 실제 구현 사용(부분 진행 보존)."""
    spec: ExamSpec = {
        "passage_text": sample["passage_text"],
        "standards": sample.get("standards", []),
        "num_items": sample["num_items"],
    }
    init_session()
    graph = get_exam_graph()
    state = graph.invoke(
        {
            "spec": spec, "budget": budget, "draft_items": [], "agent_messages": [],
            "validation_passed": False, "similarity_judge_result": {}, "error": "",
        }
    )
    items = state.get("draft_items", [])
    return {"id": sample["id"], "num_items": sample["num_items"], "generated": len(items),
            "exact_match": len(items) == sample["num_items"]}


def summarize(label: str, rs: list) -> None:
    n = len(rs)
    exact = sum(1 for r in rs if r["exact_match"])
    print(f"{label}: 정확히 일치 {exact}/{n} ({exact/n*100:.0f}%)")


def main() -> None:
    results = {"old": [], "new": []}
    for label, fn in [("old", run_old), ("new", run_new)]:
        print(f"\n=== {label} (budget={BUDGET}, n={len(TEST_SAMPLES)}) ===")
        for i, s in enumerate(TEST_SAMPLES, 1):
            print(f"[{i}/{len(TEST_SAMPLES)}] {s['id']} (num_items={s['num_items']}) 생성 중...")
            r = fn(s, BUDGET)
            print(f"  생성 {r['generated']}개 (목표 {r['num_items']})")
            results[label].append(r)

    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 결과 요약 ===")
    summarize("old (전체 재시도)", results["old"])
    summarize("new (부분 진행 보존)", results["new"])


if __name__ == "__main__":
    main()
