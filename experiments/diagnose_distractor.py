#!/usr/bin/env python
"""오답매력도 미달의 원인 파악 (2026-08-07).

## 왜 새로 재야 하는가 — 추적 중인 숫자가 유효하지 않다

로드맵·README가 추적하는 **"오답매력도 2.846"은 2026-07-09 측정치**인데, 그 시점 스택은
지금과 두 축 모두 다르다:

| | 2.846 측정 당시(2026-07-09) | 현재 |
|---|---|---|
| 생성 모델 | qwen2.5:**7b** | qwen2.5:**14b** (2026-07-15 승격) |
| Judge | 로컬 qwen2.5 | **gpt-5.6-luna** (2026-07-21 채택) |

한편 README의 **3.40**(2026-07-24, gpt-5.6-luna)은 Judge는 최신이지만 **ITEM_GOLDEN**
(스크립트에 고정된 30문항)을 채점한 값이다 — EVAL.md 9절이 명시했듯 이 경로는
`agent_node`를 **아예 호출하지 않으므로** 우리 생성 품질을 재는 값이 아니다.

즉 **"현재 스택이 생성한 문항의 오답매력도"는 측정된 적이 없다.**
(참고로 같은 고정 셋에 Judge만 바꿨을 때 2.83→3.40으로 +0.57이 움직였다. Judge 교체
효과만으로도 숫자가 크게 달라진다는 뜻이라, 옛 값과 목표를 직접 비교하면 안 된다.)

## 이 스크립트가 하는 일

1. 현재 그래프로 문항을 실제 생성(프로덕션과 같은 budget)
2. 현재 Judge(`get_judge_backend()`)로 `judge_one` 채점
3. **3개 기준을 분해**해 오답매력도가 정말 최약점인지 확인
4. 생성된 **선지 원문을 그대로 출력** — 왜 약한지 눈으로 보기 위함

실행:
    CHROMA_PERSIST_DIR=./chroma_db python experiments/diagnose_distractor.py --runs 8
"""
import argparse
import json
import os
import statistics
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
from app.modules.exam.tools import get_draft_items, init_session
from eval_lib import judge_one  # noqa: E402

from scripts.test_exam import PASSAGE_TEXT  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "data", "golden", "_distractor_diagnosis.json")

# 비교 기준선 — 전부 "현재 스택 생성물"이 아니라는 점이 이 진단의 출발점이다.
_STALE_REFS = {
    "2026-07-09_generated_7b_qwenjudge": 2.846,   # 생성·Judge 둘 다 구버전
    "2026-07-24_ITEM_GOLDEN_gptjudge": 3.40,      # Judge는 최신이나 고정 셋(생성 경로 미포함)
}
_TARGET = 4.0


def run_once(index: int, budget: int, judge_llm) -> list[dict]:
    spec: ExamSpec = {"passage_text": PASSAGE_TEXT, "num_items": 1}
    init_session()
    graph = get_exam_graph()
    graph.invoke(
        {
            "spec": spec,
            "budget": budget,
            "draft_items": [],
            "agent_messages": [],
            "validation_passed": False,
            "similarity_judge_result": {},
        }
    )
    rows = []
    for it in get_draft_items():
        # 오답매력도는 선지가 있어야 성립 — 서술형은 대상 밖(과거 측정도 "객관식만")
        if it.get("item_type") != "객관식" or not it.get("options"):
            print(f"[{index}] 객관식 아님 — 건너뜀 ({it.get('item_type')})")
            continue
        scores = judge_one(it, judge_llm)
        row = {
            "run": index,
            "question": it.get("question", ""),
            "options": it.get("options", []),
            "answer": it.get("answer", ""),
            "정답유일성": scores.get("정답유일성"),
            "오답매력도": scores.get("오답매력도"),
            "근거성": scores.get("근거성"),
        }
        rows.append(row)
        print(f"\n[{index}] 정답유일성={row['정답유일성']} 오답매력도={row['오답매력도']} 근거성={row['근거성']}")
        print(f"    Q: {row['question'][:70]}")
        for opt in row["options"]:
            mark = "✔" if row["answer"] and str(opt).startswith(row["answer"]) else " "
            print(f"    {mark} {opt}")
    return rows


def report(rows: list[dict]) -> dict:
    n = len(rows)
    crit = ["정답유일성", "오답매력도", "근거성"]

    def stat(k):
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return {
            "avg": round(statistics.mean(vals), 3) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n": len(vals),
        }

    stats = {k: stat(k) for k in crit}

    print("\n" + "=" * 64)
    print(f"  현재 스택 생성물의 문항 품질 분해 (n={n}, 객관식만)")
    print("  생성: qwen2.5:14b / Judge: gpt-5.6-luna")
    print("=" * 64)
    print(f"{'기준':<12}{'평균':<10}{'최소':<7}{'최대':<7}{'목표 대비':<12}")
    for k in crit:
        s = stats[k]
        gap = f"{s['avg'] - _TARGET:+.2f}" if s["avg"] is not None else "-"
        print(f"{k:<12}{str(s['avg']):<10}{str(s['min']):<7}{str(s['max']):<7}{gap:<12}")

    weakest = min(
        (k for k in crit if stats[k]["avg"] is not None),
        key=lambda k: stats[k]["avg"],
        default=None,
    )
    print(f"\n▶ 최약점: {weakest}")

    print("\n[참고] 기존에 추적하던 값들 — 둘 다 '현재 스택 생성물'이 아님")
    for label, val in _STALE_REFS.items():
        print(f"  {label}: {val}")

    cur = stats["오답매력도"]["avg"]
    if cur is not None:
        print(f"\n▶ 현재 스택 실측 오답매력도: {cur}  (목표 {_TARGET})")
        old = _STALE_REFS["2026-07-09_generated_7b_qwenjudge"]
        print(f"   구버전 스택 대비: {cur - old:+.3f}  "
              f"(단, 생성·Judge가 둘 다 바뀌어 어느 쪽 기여인지는 이 측정만으론 분해 불가)")

    return {
        "_schema": {
            "description": (
                "현재 스택(qwen2.5:14b 생성 + gpt-5.6-luna Judge)이 실제 생성한 객관식 문항의 "
                "문항 품질 3기준 분해 측정 (2026-08-07). 기존 추적값 2.846은 7B 생성+qwen Judge "
                "시절 값이고 3.40은 고정 ITEM_GOLDEN 채점값이라 둘 다 현재 생성 품질이 아니다. "
                "골든셋 아님 — 재실행 시 덮어씀. 생성: experiments/diagnose_distractor.py"
            )
        },
        "n_items": n,
        "target": _TARGET,
        "stats": stats,
        "weakest_criterion": weakest,
        "stale_references": _STALE_REFS,
        "items": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--budget", type=int, default=5, help="프로덕션과 동일(기본 5)")
    args = parser.parse_args()

    judge_llm = get_judge_backend()
    print(f"실행 {args.runs}회 (budget={args.budget}, num_items=1)\n")

    all_rows = []
    for i in range(1, args.runs + 1):
        all_rows.extend(run_once(i, args.budget, judge_llm))

    if not all_rows:
        sys.exit("객관식 문항이 생성되지 않아 진단 불가.")

    result = report(all_rows)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
