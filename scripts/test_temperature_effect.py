#!/usr/bin/env python
"""temperature 0.7 vs 0.2가 tool-calling 실패율에 영향을 주는지 A/B 비교.

TROUBLESHOOTING.md 배경: num_ctx 수정 후에도 qwen2.5:7b가 도구 호출 대신
일반 텍스트로 응답하며 문항 0개로 끝나는 잔여 실패율(~35~40%)이 남아있음.
temperature를 낮추면 구조화 출력(tool calling)이 더 안정적이라는 가설을 검증한다.

같은 passage 세트를 budget=1(재시도 없음, 순수 1회 성공률 측정)로 두 temperature에서
각각 돌려 "문항 0개로 끝난 비율"을 비교한다. agent_node의 신규 "설명 텍스트 금지"
지시는 두 조건 모두에 이미 반영된 현재 graph.py를 그대로 사용(공통 조건으로 고정,
temperature만 변수).
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

from app.modules.exam.llm import get_langchain_model
from app.modules.exam.tools import TOOLS, get_draft_items, init_session

# gen_structure_golden.py의 기존 정의를 재사용(중복 방지)
from gen_structure_golden import PASSAGE_SAMPLES

_TEST_IDS = {
    "str_002", "str_005", "str_006", "str_008", "str_009",
    "str_013", "str_015", "str_016", "str_018", "str_020",
}
TEST_SAMPLES = [s for s in PASSAGE_SAMPLES if s["id"] in _TEST_IDS]

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_temperature_ab_compare.json")


def build_system_prompt(passage_text: str, num_items: int) -> str:
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


def run_once(sample: dict, temperature: float) -> dict:
    system_prompt = build_system_prompt(sample["passage_text"], sample["num_items"])
    user_content = "위 지침에 따라 예시 문제와 동일한 구성의 문항 세트를 작성하세요."
    if sample.get("standards"):
        user_content += f"\n\n참고 성취기준: {', '.join(sample['standards'])}"

    tool_map = {t.name: t for t in TOOLS}
    init_session()
    llm = get_langchain_model(temperature=temperature).bind_tools(TOOLS)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

    extra_text_turns = 0  # 도구 호출과 별개로 content에 텍스트를 쓴 턴 수(응답 품질 프록시)
    for _ in range(14):
        response = llm.invoke(messages)
        messages.append(response)
        if getattr(response, "content", "") and getattr(response, "tool_calls", []):
            extra_text_turns += 1
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
                    result_content = f"도구 호출 오류 — 인자 형식을 확인하고 다시 호출하세요: {e}"
            messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))
            if tc["name"] == "similarity_judge":
                judged = True
        if judged:
            break

    items = get_draft_items()
    return {
        "id": sample["id"],
        "num_items": sample["num_items"],
        "generated": len(items),
        "extra_text_turns": extra_text_turns,
    }


def main() -> None:
    results = {"temperature_0.7": [], "temperature_0.2": []}
    for label, temp in [("temperature_0.7", 0.7), ("temperature_0.2", 0.2)]:
        print(f"\n=== {label} ===")
        for i, sample in enumerate(TEST_SAMPLES, 1):
            print(f"[{i}/{len(TEST_SAMPLES)}] {sample['id']} 생성 중...")
            r = run_once(sample, temp)
            print(f"  생성 {r['generated']}개 (목표 {r['num_items']}), 설명텍스트 있던 턴 {r['extra_text_turns']}개")
            results[label].append(r)

    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 결과 요약 ===")
    for label in results:
        rs = results[label]
        fail = sum(1 for r in rs if r["generated"] == 0)
        avg_extra = sum(r["extra_text_turns"] for r in rs) / len(rs)
        print(f"{label}: 0개 실패 {fail}/{len(rs)} ({fail/len(rs)*100:.0f}%), 턴당 평균 설명텍스트 {avg_extra:.2f}")


if __name__ == "__main__":
    main()
