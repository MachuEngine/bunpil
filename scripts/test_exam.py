#!/usr/bin/env python
"""출제 모듈 통합 테스트 (passage_text 붙여넣기 리디자인 반영).
예시 문제 텍스트를 입력으로 ReAct 에이전트가 문항 세트를 생성하고,
별도 judge 노드(get_judge_backend())가 채점한 결과 기반으로 재시도
여부를 판단하는 흐름을 확인한다(2026-07-23부터 자기채점 아님).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2026-08-04 추가: 이 스크립트는 .env를 전혀 읽지 않아 LANGCHAIN_API_KEY가 세팅되지
# 않았고, LANGCHAIN_TRACING_V2=true로 실행하면 트레이스 전송이 401로 거부됐다.
# 또 init_langsmith_project()를 호출하지 않아 프로젝트 자동 분기(-dev/-prod)도
# 안 걸려, 트레이스가 bunpil-dev가 아니라 맨 'bunpil'로 샐 수 있었다.
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:14b")

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.exam.graph import _MIN_OVERALL_SCORE, _MIN_TYPE_RATIO_SCORE
from app.modules.exam.tools import init_session

# 2026-08-07 변경: 예시 문항 2개를 **둘 다 객관식(유형 균일)** 으로 바꿨다.
#
# 이전엔 객관식1 + 서술형1이었는데, num_items=1을 요청하면 1문항 세트가 그 2가지 유형
# 구성을 재현할 수 없어 `type_ratio_score`가 구조적으로 0.5에서 막혔다. 게다가
# `STRUCTURE_JUDGE_TPL`의 few-shot 3번이 정확히 이 상황("객관식2+서술형1" 예시에
# 객관식 1개 생성)을 `overall_score=2`로 채점하라고 가르치고 있어, **어떤 임계값을 써도
# validate가 통과할 수 없는 조건**이었다 — 즉 이 스모크 테스트는 회귀 감지 도구로
# 기능하지 못하고 있었다(EVAL.md 18절, TROUBLESHOOTING 14번).
#
# 유형만 균일하게 맞추면 개수 디커플링(1 ≠ 2) 검증은 그대로 유지된다. 실측 확인:
# 예시=객관식2 / 생성=객관식1 → type_ratio 1.0, overall 4~5 (3회 시행 전부 통과).
# Judge가 개수를 감점하지 않는다는 프롬프트 명세도 이 실측으로 함께 확인됐다.
#
# 트레이드오프: 서술형 생성 경로는 이 스모크에서 더 이상 타지 않는다. 형식 검증
# 자체는 `tests/test_exam_tool_gates.py`가 LLM 없이 결정론적으로 커버한다.
PASSAGE_TEXT = """\
[예시 문제]
1. 다음 중 대한민국 헌법이 규정한 민주주의 원리로 옳지 않은 것은?
① 국민 주권 ② 권력 분립 ③ 기본권 보장 ④ 계획 경제

2. 다음 중 시장 실패의 사례로 가장 적절한 것은?
① 완전 경쟁 시장의 가격 결정 ② 공공재의 무임승차 ③ 자유로운 시장 진입 ④ 신축적인 가격 조정
"""


def main() -> None:
    # num_items를 예시 문제 자체의 문항 수(2개)와 다르게 줘서, 생성 개수가
    # 예시 개수가 아니라 num_items를 따르는지(count_match 디커플링) 검증한다.
    # 라이브 모델 smoke test는 tool-call 경로 자체를 안정적으로 확인하도록 1개만 생성한다.
    # 다문항 개수·교체 게이트는 tests/test_exam_*.py의 결정론적 테스트가 담당한다.
    num_items = 1
    spec: ExamSpec = {
        "passage_text": PASSAGE_TEXT,
        "num_items": num_items,
    }

    print("=== 출제 모듈 통합 테스트 (passage_text 리디자인) ===\n")
    print(f"입력 지문 길이: {len(PASSAGE_TEXT)}자 (예시 문항 2개, 요청 num_items={num_items})")
    print("\nReAct 에이전트 출제 시작...")

    init_session()
    graph = get_exam_graph()
    state = graph.invoke(
        {
            "spec": spec,
            "budget": 2,
            "draft_items": [],
            "agent_messages": [],
            "validation_passed": False,
            "similarity_judge_result": {},
        }
    )

    print("\n결과 확인")
    items = state.get("draft_items", [])

    print(f"  생성 문항: {len(items)}개 (목표 {num_items}개)")
    print(f"  검증 통과: {state.get('validation_passed', False)}")
    print(f"  similarity_judge_result: {state.get('similarity_judge_result')}")

    for i, it in enumerate(items, 1):
        print(
            f"\n  [{i}] {it.get('item_type','?')} | 난이도:{it.get('difficulty','?')}"
        )
        print(f"       Q: {str(it.get('question',''))[:80]}")

    # 2026-08-07: 실패를 한 줄로 뭉뚱그리지 않고 **어느 단계가 깨졌는지** 구분한다.
    # 이전엔 "[실패] 목표 문항 수·구조 검증 미충족" 한 줄뿐이라, 배선이 끊긴 것인지
    # 모델 품질이 모자란 것인지 알 수 없었다 — 실제로 이 모호함 때문에 게이트 재보정
    # 효과를 잘못 읽을 뻔했다(EVAL.md 18절).
    judge = state.get("similarity_judge_result") or {}
    if not items:
        print(
            "\n[실패] 문항이 하나도 저장되지 않았습니다 — tool-calling 배선/안정성 문제일 수 있습니다.\n"
            "        LangSmith 트레이스에서 save_item 호출 여부와 malformed tool-call을 확인하세요."
        )
        raise SystemExit(1)
    if len(items) != num_items:
        print(
            f"\n[실패] 개수 불일치: {len(items)}개 생성 / 목표 {num_items}개 — "
            "생성은 되지만 목표 개수를 못 채웠습니다(턴 예산·재시도 확인)."
        )
        raise SystemExit(1)
    if not state.get("validation_passed", False):
        print(
            f"\n[실패] 개수는 맞으나 Judge 게이트 미달: {judge}\n"
            f"        기준 — type_ratio>={_MIN_TYPE_RATIO_SCORE}, difficulty_match=True, "
            f"overall>={_MIN_OVERALL_SCORE}.\n"
            "        생성 품질 문제일 수도, 게이트가 다시 과하게 조여진 것일 수도 있습니다."
        )
        raise SystemExit(1)
    print("\n[완료] 출제 모듈 통합 테스트 통과")


if __name__ == "__main__":
    main()
