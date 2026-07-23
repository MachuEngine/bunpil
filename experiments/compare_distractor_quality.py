#!/usr/bin/env python
"""agent_node 시스템 프롬프트에 추가한 오답 매력도 지시의 실제 효과 검증.

eval_exam.py의 eval_item_quality()는 ITEM_GOLDEN(스크립트에 하드코딩된 고정
30개 문항)을 채점한다 — agent_node를 전혀 호출하지 않으므로 생성 프롬프트를
바꿔도 그 결과에 영향을 줄 수 없다(2026-07-09 발견, before/after가 완전히
동일하게 나와서 확인됨).

이 스크립트는 실제로 graph.py의 agent_node 루프를 그대로 복제해 OLD(변경 전)
프롬프트와 NEW(변경 후, 오답 매력도 지시+예시 추가) 프롬프트 각각으로 같은
passage 세트에 대해 문항을 새로 생성하고, 그 결과를 JUDGE_TPL로 채점해
오답매력도 평균을 비교한다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.common.llm import get_judge_backend
from app.modules.exam.llm import get_langchain_model
from app.modules.exam.tools import TOOLS, get_draft_items, init_session

from eval_lib import JUDGE_TPL, _run_async

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_distractor_quality_compare.json")

OLD_SYSTEM_SUFFIX = (
    "문항은 당신이 직접 작성합니다. "
    "객관식 선지는 반드시 ①②③④ 형식으로 4개 작성하세요."
)
NEW_SYSTEM_SUFFIX = (
    "문항은 당신이 직접 작성합니다. "
    "객관식 선지는 반드시 ①②③④ 형식으로 4개 작성하세요.\n\n"
    "오답(정답이 아닌 선지)은 명백히 틀리거나 문제와 무관한 내용이 아니라, "
    "같은 개념 범주 안에서 학생이 실제로 헷갈릴 수 있는 그럴듯한 오답으로 구성하세요. "
    "예를 들어 정답이 '비례대표제'라면 오답은 '외계인 침공'처럼 무관한 선지가 아니라 "
    "'소선거구제', '직접 선거제'처럼 같은 주제의 인접 개념이어야 합니다."
)

# STRUCTURE_GOLDEN과 겹치지 않는 새 객관식 위주 passage (오답매력도는 객관식에만 적용되므로
# 서술형 비중을 줄이고 객관식을 늘려 표본을 최대화한다)
PASSAGES = [
    {
        "passage_text": "1. 국가 예산이 국회에서 확정되는 절차로 옳은 것은?\n① 대통령이 단독으로 확정한다 ② 국회 심의·의결을 거쳐 확정된다 ③ 감사원이 최종 승인한다 ④ 지방자치단체가 결정한다",
        "standards": ["예산 심의와 확정 절차"],
    },
    {
        "passage_text": "1. 인플레이션이 지속될 때 나타나는 현상으로 옳은 것은?\n① 화폐 가치 상승 ② 실질 소득 감소 ③ 저축 유인 증가 ④ 수출 경쟁력 강화\n\n2. 통화량 증가가 물가에 미치는 영향으로 옳은 것은?\n① 물가 하락 ② 물가 상승 ③ 물가 불변 ④ 환율 하락",
        "standards": ["물가와 인플레이션"],
    },
    {
        "passage_text": "1. 소비자기본법에서 보장하는 소비자의 권리로 옳지 않은 것은?\n① 안전할 권리 ② 알 권리 ③ 기업의 영업 비밀을 요구할 권리 ④ 피해 보상을 받을 권리",
        "standards": ["소비자의 권리와 보호"],
    },
    {
        "passage_text": "1. 지방자치단체의 조례 제정권에 대한 설명으로 옳은 것은?\n① 법률의 범위를 벗어나도 유효하다 ② 법령의 범위 안에서 제정할 수 있다 ③ 국회의 사전 승인이 필요하다 ④ 조례는 상위법과 무관하게 독립적으로 효력을 가진다",
        "standards": ["지방자치단체의 자치입법권"],
    },
    {
        "passage_text": "1. 중앙은행의 통화정책 수단으로 옳은 것은?\n① 최저임금 조정 ② 기준금리 조정 ③ 법인세율 조정 ④ 관세율 조정",
        "standards": ["통화정책의 수단과 효과"],
    },
]


def run_agent(passage_text: str, standards: list, system_suffix: str, budget: int = 3):
    system_prompt = (
        "당신은 한국 고등학교 사회 문항 출제 전문가 에이전트입니다. 한국어로만 응답하세요.\n\n"
        "다음은 교사가 참고용으로 제시한 예시 문제입니다.\n\n"
        f"[예시 문제]\n{passage_text}\n\n"
        "위 예시의 문항 수, 유형(객관식/서술형) 구성, 난이도 수준을 그대로 파악하여 "
        "동일한 개수·구성·난이도의 새 문항 세트를 작성하세요.\n\n"
        "문항마다 다음 순서로 도구를 호출하세요:\n"
        "1. [선택] search_standards — 참고 성취기준 원문 확인\n"
        "2. [선택] search_regulations — 교육과정 준수 사항 확인\n"
        "3. validate_item_format — 직접 구성한 문항의 형식 검증\n"
        "   (오류가 있으면 수정 후 재검증, 통과할 때까지 반복)\n"
        "4. save_item — 검증 통과한 문항 저장\n"
        "5. record_score — 품질 자체 평가 (0~5점)\n\n"
        "문항 세트 작성이 모두 끝나면 similarity_judge 도구를 호출해 "
        "예시 문제와의 구조적 유사도(문항 개수·유형 비율·난이도 구성)를 스스로 평가하세요.\n\n"
        f"{system_suffix}"
    )
    user_content = "위 지침에 따라 예시 문제와 동일한 구성의 문항 세트를 작성하세요."
    if standards:
        user_content += f"\n\n참고 성취기준: {', '.join(standards)}"

    tool_map = {t.name: t for t in TOOLS}

    for attempt in range(budget):
        init_session()
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
                        result_content = f"도구 호출 오류 — 인자 형식을 확인하고 다시 호출하세요: {e}"
                messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))
                if tc["name"] == "similarity_judge":
                    judged = True
            if judged:
                break
        items = get_draft_items()
        mc_items = [it for it in items if it.get("item_type") == "객관식" and len(it.get("options", [])) == 4]
        if mc_items:
            return mc_items
    return []


def judge_items(items: list, judge_llm) -> list:
    scores = []
    for item in items:
        item_str = json.dumps(
            {"question": item["question"], "options": item.get("options", []), "answer": item.get("answer", "")},
            ensure_ascii=False,
        )
        messages = JUDGE_TPL.build(item_str)
        raw = _run_async(judge_llm.generate(messages))
        try:
            s, e = raw.find("{"), raw.rfind("}") + 1
            parsed = json.loads(raw[s:e]) if s >= 0 and e > s else {}
        except Exception:
            parsed = {}
        scores.append(int(parsed.get("오답매력도", 3)))
    return scores


def main() -> None:
    judge_llm = get_judge_backend()

    results = {"old": [], "new": []}
    for label, suffix in [("old", OLD_SYSTEM_SUFFIX), ("new", NEW_SYSTEM_SUFFIX)]:
        print(f"\n=== {label.upper()} 프롬프트로 생성 ===")
        all_items = []
        for i, p in enumerate(PASSAGES, 1):
            print(f"[{i}/{len(PASSAGES)}] 생성 중...")
            mc_items = run_agent(p["passage_text"], p.get("standards", []), suffix, budget=3)
            print(f"  객관식 문항 {len(mc_items)}개 확보")
            all_items.extend(mc_items)
        print(f"{label.upper()} 총 객관식 문항: {len(all_items)}개 — 채점 중...")
        scores = judge_items(all_items, judge_llm)
        results[label] = scores
        print(f"{label.upper()} 오답매력도 점수: {scores}")

    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    print("\n=== 결과 ===")
    print(f"OLD: n={len(results['old'])}, 평균 오답매력도={avg(results['old'])}")
    print(f"NEW: n={len(results['new'])}, 평균 오답매력도={avg(results['new'])}")


if __name__ == "__main__":
    main()
