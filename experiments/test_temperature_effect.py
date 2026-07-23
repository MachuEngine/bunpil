#!/usr/bin/env python
"""temperature 0.7 vs 0.2가 tool-calling 성공률에 영향을 주는지 A/B 비교.

TROUBLESHOOTING.md 배경: num_ctx 수정 후에도 qwen2.5:7b가 도구 호출 대신
일반 텍스트로 응답하거나 목표 개수를 못 맞추는 잔여 실패율이 남아있음.
temperature를 낮추면 구조화 출력(tool calling)이 더 안정적이라는 가설을 검증한다.

2026-07-10 1차 검증(n=10, "0개 실패율" 기준)에서 방향성(40%→20%)은 봤으나 표본이
작고, "생성 개수==num_items 정확히 일치"가 아니라 "0개가 아님"만 봐서 목표 초과
생성(예: 3개 요청→5~6개) 같은 새로운 실패 유형을 놓쳤을 수 있다는 지적을 받아
아래처럼 재설계함:

- 지표: exact_match_rate = (생성 개수 == num_items) 비율 (0개/부족/초과 전부 실패로 집계)
- 표본: 조건당 --n개(기본 28, gen_structure_golden.py의 34개 샘플 중 앞에서부터 사용)
- "도구 호출 외 설명 텍스트 금지" 지시가 temperature 0.2에서도 전혀 안 먹혔던 것
  (턴당 0.70으로 baseline과 동일)에 대해 프롬프트 위치/강조를 바꾼 "strong" 변형을
  추가해 --prompt-variant로 선택 가능하게 함.

사용법:
  python experiments/test_temperature_effect.py --check-variant   # 지시 위치/강조 변경 효과만 소표본으로 빠르게 확인
  python experiments/test_temperature_effect.py --full --n 28 --prompt-variant strong  # 본 실험
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden_gen"))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.modules.exam.graph import _invoke_with_retry
from app.modules.exam.llm import get_langchain_model
from app.modules.exam.tools import TOOLS, get_draft_items, init_session

# gen_structure_golden.py의 기존 정의를 재사용(중복 방지) — 34개, 다양한 주제/num_items
from gen_structure_golden import PASSAGE_SAMPLES

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_temperature_ab_compare.json")

_NO_TEXT_BASELINE = (
    "도구 호출 외의 설명 텍스트는 쓰지 마세요. 생각 과정이나 진행 상황을 문장으로 "
    "서술하지 말고, 곧바로 다음 도구를 호출하세요."
)
_NO_TEXT_STRONG = (
    "**매우 중요한 규칙**: 이 대화 내내 도구 호출(tool call) 외에는 어떤 텍스트도 "
    "출력하지 마세요. 인사, 생각 과정 설명, 진행 상황 서술, 문항 초안을 텍스트로 "
    "먼저 보여주는 것 모두 금지입니다. 매 턴 오직 도구 호출만 하세요."
)


def build_system_prompt(passage_text: str, num_items: int, prompt_variant: str) -> str:
    identity = "당신은 한국 고등학교 사회 문항 출제 전문가 에이전트입니다. 한국어로만 응답하세요."
    no_text_rule = _NO_TEXT_STRONG if prompt_variant == "strong" else _NO_TEXT_BASELINE

    body = (
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
        "'소선거구제', '직접 선거제'처럼 같은 주제의 인접 개념이어야 합니다."
    )

    if prompt_variant == "strong":
        # 강조(primacy) — 정체성 선언 바로 다음, 최우선 규칙으로 배치 + 끝에서 한 번 더 반복(recency)
        return f"{identity}\n\n{no_text_rule}\n\n{body}\n\n{no_text_rule}"
    return f"{identity}\n\n{body}\n\n{no_text_rule}"


def run_once(sample: dict, temperature: float, prompt_variant: str) -> dict:
    system_prompt = build_system_prompt(sample["passage_text"], sample["num_items"], prompt_variant)
    user_content = "위 지침에 따라 예시 문제와 동일한 구성의 문항 세트를 작성하세요."
    if sample.get("standards"):
        user_content += f"\n\n참고 성취기준: {', '.join(sample['standards'])}"

    tool_map = {t.name: t for t in TOOLS}
    init_session()
    llm = get_langchain_model(temperature=temperature).bind_tools(TOOLS)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

    extra_text_turns = 0  # 도구 호출과 별개로 content에 텍스트를 쓴 턴 수(지시 준수 프록시)
    for _ in range(14):
        response = _invoke_with_retry(llm, messages)
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
        "exact_match": len(items) == sample["num_items"],
        "extra_text_turns": extra_text_turns,
    }


def summarize(label: str, rs: list) -> None:
    n = len(rs)
    exact = sum(1 for r in rs if r["exact_match"])
    zero = sum(1 for r in rs if r["generated"] == 0)
    over = sum(1 for r in rs if r["generated"] > r["num_items"])
    under = sum(1 for r in rs if 0 < r["generated"] < r["num_items"])
    avg_extra = sum(r["extra_text_turns"] for r in rs) / n
    print(
        f"{label}: 정확히 일치 {exact}/{n} ({exact/n*100:.0f}%) | "
        f"0개 {zero} | 부족 {under} | 초과 {over} | 턴당 평균 설명텍스트 {avg_extra:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-variant", action="store_true", help="지시 위치/강조 변경 효과만 소표본(8개, temp=0.2)으로 빠르게 확인")
    parser.add_argument("--full", action="store_true", help="0.7 vs 0.2 본 실험 실행")
    parser.add_argument("--n", type=int, default=28)
    parser.add_argument("--prompt-variant", choices=["baseline", "strong"], default="baseline")
    args = parser.parse_args()

    if args.check_variant:
        samples = PASSAGE_SAMPLES[:8]
        results = {"baseline": [], "strong": []}
        for variant in ["baseline", "strong"]:
            print(f"\n=== 지시 변형 확인: {variant} (temperature=0.2) ===")
            for i, s in enumerate(samples, 1):
                print(f"[{i}/{len(samples)}] {s['id']} 생성 중...")
                r = run_once(s, 0.2, variant)
                print(f"  생성 {r['generated']}개 (목표 {r['num_items']}), 설명텍스트 있던 턴 {r['extra_text_turns']}개")
                results[variant].append(r)
        print("\n=== 지시 변형 비교 요약 ===")
        summarize("baseline", results["baseline"])
        summarize("strong", results["strong"])
        return

    if args.full:
        samples = PASSAGE_SAMPLES[: args.n]
        results = {"temperature_0.7": [], "temperature_0.2": []}
        for label, temp in [("temperature_0.7", 0.7), ("temperature_0.2", 0.2)]:
            print(f"\n=== {label} (prompt_variant={args.prompt_variant}, n={len(samples)}) ===")
            for i, s in enumerate(samples, 1):
                print(f"[{i}/{len(samples)}] {s['id']} 생성 중...")
                r = run_once(s, temp, args.prompt_variant)
                print(f"  생성 {r['generated']}개 (목표 {r['num_items']}), 설명텍스트 있던 턴 {r['extra_text_turns']}개")
                results[label].append(r)

        with open(_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n=== 결과 요약 ===")
        summarize("temperature_0.7", results["temperature_0.7"])
        summarize("temperature_0.2", results["temperature_0.2"])
        return

    parser.print_help()


if __name__ == "__main__":
    main()
