#!/usr/bin/env python
"""정답유일성 지시의 효과를 표본을 늘려 A/B로 재검증 (2026-08-07).

## 배경

23절에서 `agent_node` 프롬프트에 "조건을 만족하는 선지가 정확히 하나인지 확인하라"는
지시를 추가해 정답유일성 3.375(n=8) → 4.125(n=16)를 얻었다. 그러나:

- **대조군이 n=8로 약하다** — 비교의 신뢰도는 약한 쪽 팔이 결정한다.
- 두 팔이 **다른 시점에** 측정됐다(머신 상태·모델 캐시 등 교란 가능).

그래서 양쪽 팔을 모두 늘리고 **교차 실행**(control, treatment, control, ...)해
시간 드리프트를 통제한 뒤 재검증한다.

## 설계

- 대조군: 유일성 지시 **없는** 프롬프트 (변경 전과 동일)
- 처치군: 현재 프로덕션 프롬프트 (유일성 지시 포함)
- 두 팔을 번갈아 실행. 그 외 조건(passage, num_items=1, budget=5, 모델)은 동일.
- 대조군은 **프로덕션 코드를 건드리지 않고** `_build_system_prompt`를 이 스크립트 안에서만
  monkeypatch해 해당 블록을 제거한다(실험이 프로덕션 동작을 바꾸지 않도록).
- Mann-Whitney U(순서형·소표본에 적합)로 유의성을 함께 본다.

실행:
    CHROMA_PERSIST_DIR=./chroma_db python experiments/ab_uniqueness_instruction.py --per-arm 12
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:14b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm.tracing import init_langsmith_project

init_langsmith_project()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from app.common.llm import get_judge_backend
from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.exam import graph as graph_mod
from app.modules.exam.tools import get_draft_items, init_session
from eval_lib import judge_one  # noqa: E402

from scripts.test_exam import PASSAGE_TEXT  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "data", "golden", "_ab_uniqueness_instruction.json")

# 23절에서 추가한 블록의 시작·끝 마커. 대조군에서 이 구간만 들어낸다.
_BLOCK_START = "**선지를 다 쓴 뒤"
_BLOCK_END = "오답(정답이 아닌 선지)"

_ORIGINAL_BUILD = graph_mod._build_system_prompt


def _build_without_uniqueness(*args, **kwargs) -> str:
    """유일성 지시 블록을 제거한 프롬프트(= 23절 변경 전 상태)."""
    p = _ORIGINAL_BUILD(*args, **kwargs)
    i, j = p.find(_BLOCK_START), p.find(_BLOCK_END)
    if i == -1 or j == -1 or j <= i:
        raise RuntimeError("유일성 블록을 찾지 못했습니다 — 프롬프트가 바뀐 듯하니 마커를 갱신할 것")
    return p[:i] + p[j:]


def run_once(arm: str, index: int, judge_llm) -> list[dict]:
    if arm == "control":
        graph_mod._build_system_prompt = _build_without_uniqueness
    else:
        graph_mod._build_system_prompt = _ORIGINAL_BUILD

    # 그래프가 캐시돼 있어도 `agent_node`는 호출 시점에 모듈 전역에서
    # `_build_system_prompt`를 찾으므로, 위 패치가 그대로 반영된다.
    spec: ExamSpec = {"passage_text": PASSAGE_TEXT, "num_items": 1}
    init_session()
    graph = get_exam_graph()
    graph.invoke({
        "spec": spec, "budget": 5, "draft_items": [],
        "agent_messages": [], "validation_passed": False, "similarity_judge_result": {},
    })

    rows = []
    for it in get_draft_items():
        if it.get("item_type") != "객관식" or not it.get("options"):
            print(f"  [{arm} #{index}] 객관식 아님 — 제외")
            continue
        s = judge_one(it, judge_llm)
        rows.append({
            "arm": arm, "run": index,
            "question": it.get("question", ""), "options": it.get("options", []),
            "answer": it.get("answer", ""),
            "정답유일성": s.get("정답유일성"), "오답매력도": s.get("오답매력도"), "근거성": s.get("근거성"),
        })
        print(f"  [{arm} #{index}] 유일성={s.get('정답유일성')} 매력도={s.get('오답매력도')} 근거성={s.get('근거성')}")
    return rows


def report(rows: list[dict], per_arm: int) -> dict:
    crit = ["정답유일성", "오답매력도", "근거성"]
    ctl = [r for r in rows if r["arm"] == "control"]
    trt = [r for r in rows if r["arm"] == "treatment"]

    def vals(group, k):
        return [r[k] for r in group if isinstance(r[k], (int, float))]

    print("\n" + "=" * 66)
    print(f"  A/B 재검증 — 정답유일성 지시 (대조 n={len(ctl)} / 처치 n={len(trt)})")
    print("=" * 66)
    print(f"{'기준':<12}{'대조(지시없음)':<16}{'처치(지시있음)':<16}{'차이':<10}")
    summary = {}
    for k in crit:
        c, t = vals(ctl, k), vals(trt, k)
        mc, mt = (st.mean(c) if c else 0), (st.mean(t) if t else 0)
        summary[k] = {"control_mean": round(mc, 3), "treatment_mean": round(mt, 3),
                      "diff": round(mt - mc, 3), "control": sorted(c), "treatment": sorted(t)}
        print(f"{k:<12}{mc:<16.3f}{mt:<16.3f}{mt - mc:+.3f}")

    # 순서형·소표본에 적합한 비모수 검정
    try:
        from scipy.stats import mannwhitneyu
        for k in crit:
            c, t = vals(ctl, k), vals(trt, k)
            if len(c) >= 3 and len(t) >= 3:
                u, p = mannwhitneyu(t, c, alternative="greater" if k == "정답유일성" else "two-sided")
                summary[k]["mannwhitney_p"] = round(float(p), 4)
        print("\nMann-Whitney U p-value (정답유일성은 단측: 처치 > 대조):")
        for k in crit:
            p = summary[k].get("mannwhitney_p")
            if p is not None:
                mark = " ← 유의(p<0.05)" if p < 0.05 else ""
                print(f"  {k}: p={p}{mark}")
    except Exception as e:  # scipy 없거나 표본 부족
        print(f"\n(유의성 검정 생략: {e})")

    # 치명적 실패율 — 평균보다 이쪽이 실사용 영향에 가깝다
    print("\n치명적 실패(정답유일성<=2, 정답이 2개인 파손 문항):")
    for lbl, g in (("대조", ctl), ("처치", trt)):
        v = vals(g, "정답유일성")
        bad = sum(1 for x in v if x <= 2)
        print(f"  {lbl}: {bad}/{len(v)} = {bad/len(v):.1%}" if v else f"  {lbl}: 표본 없음")
    summary["catastrophic_rate"] = {
        "control": round(sum(1 for x in vals(ctl, "정답유일성") if x <= 2) / max(len(vals(ctl, "정답유일성")), 1), 3),
        "treatment": round(sum(1 for x in vals(trt, "정답유일성") if x <= 2) / max(len(vals(trt, "정답유일성")), 1), 3),
    }

    return {
        "_schema": {
            "description": (
                "정답유일성 지시의 효과를 교차 실행 A/B로 재검증 (2026-08-07). 대조군은 "
                "프로덕션 코드를 바꾸지 않고 _build_system_prompt를 실험 내에서만 monkeypatch해 "
                "해당 블록을 제거한 것. 골든셋 아님 — 재실행 시 덮어씀. "
                "생성: experiments/ab_uniqueness_instruction.py"
            )
        },
        "per_arm_runs": per_arm,
        "n_control": len(ctl), "n_treatment": len(trt),
        "summary": summary,
        "items": rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arm", type=int, default=12, help="각 팔당 실행 횟수")
    args = ap.parse_args()

    judge_llm = get_judge_backend()
    print(f"교차 실행 A/B — 팔당 {args.per_arm}회 (총 {args.per_arm * 2}회)\n")

    rows = []
    try:
        for i in range(1, args.per_arm + 1):
            print(f"--- 라운드 {i}/{args.per_arm} ---")
            rows.extend(run_once("control", i, judge_llm))
            rows.extend(run_once("treatment", i, judge_llm))
            # 중간 저장 — 장시간 실행이라 중단돼도 여기까지는 남는다
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"partial": True, "items": rows}, f, ensure_ascii=False, indent=2)
    finally:
        graph_mod._build_system_prompt = _ORIGINAL_BUILD  # 반드시 원복

    if not rows:
        sys.exit("객관식 문항이 생성되지 않아 판정 불가.")
    result = report(rows, args.per_arm)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
