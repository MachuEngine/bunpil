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
from app.modules.exam.tools import init_session

PASSAGE_TEXT = """\
[예시 문제]
1. 다음 중 대한민국 헌법이 규정한 민주주의 원리로 옳지 않은 것은?
① 국민 주권 ② 권력 분립 ③ 기본권 보장 ④ 계획 경제

2. 시장 실패가 발생하는 원인을 두 가지 이상 서술하고, 정부의 대응 방안을 설명하시오.
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

    passed = len(items) == num_items and state.get("validation_passed", False)
    if not passed:
        print("\n[실패] 목표 문항 수·구조 검증을 충족하지 못했습니다.")
        raise SystemExit(1)
    print("\n[완료] 출제 모듈 통합 테스트 통과")


if __name__ == "__main__":
    main()
