#!/usr/bin/env python
"""budget(재시도 예산)이 게이트 통과율에 미치는 영향 측정 (2026-08-07).

배경: 지금까지 런타임 측정은 전부 `budget=2`(스모크 조건)였다(EVAL.md 19·20절,
합산 n=12 통과율 0.500). 프로덕션은 `budget=5`(`app/main.py`)이므로 재시도가 2.5배
더 있다 — **개선 작업을 하기 전에, 이미 프로덕션 조건에서는 괜찮은지부터** 확인한다.

부분 진행 보존 설계(2026-07-10) 덕에 재시도는 처음부터 다시 만들지 않고 부족분만
이어서 만든다. 따라서 budget이 크면 통과율이 올라갈 여지가 있다.

기록 항목:
  - 통과 여부 / Judge 3지표
  - **실제로 소모한 재시도 횟수** (초기 budget - 잔여 budget = agent 노드 실행 횟수)
    → "재시도가 실패를 실제로 회복시켰는가"를 이 값으로 본다. 1회만에 통과했다면
      budget을 늘린 것과 무관하고, 2회 이상에서 통과했다면 재시도가 값을 한 것이다.

실행:
    CHROMA_PERSIST_DIR=./chroma_db python experiments/measure_budget_effect.py --budget 5 --runs 6
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:14b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm.tracing import init_langsmith_project

init_langsmith_project()

from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.exam.graph import _MIN_OVERALL_SCORE, _MIN_TYPE_RATIO_SCORE
from app.modules.exam.tools import get_draft_items, init_session

# 19·20절과 동일한 입력을 써야 비교 가능하다.
from scripts.test_exam import PASSAGE_TEXT  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "data", "golden", "_budget_effect.json")

# 비교 기준선: budget=2로 측정한 기존 결과(EVAL.md 19·20절 합산)
_BASELINE = {"budget": 2, "n": 12, "pass_rate": 0.500}


def run_once(index: int, budget: int, num_items: int) -> dict:
    spec: ExamSpec = {"passage_text": PASSAGE_TEXT, "num_items": num_items}
    init_session()
    graph = get_exam_graph()
    started = time.time()
    state = graph.invoke(
        {
            "spec": spec,
            "budget": budget,
            "draft_items": [],
            "agent_messages": [],
            "validation_passed": False,
            "similarity_judge_result": {},
        }
    )
    elapsed = time.time() - started

    items = get_draft_items()
    judge = state.get("similarity_judge_result") or {}
    # agent_node가 매 실행마다 budget을 1씩 깎는다 → 소모량 = 초기값 - 잔여값
    agent_runs = budget - int(state.get("budget", budget))
    row = {
        "run": index,
        "n_items": len(items),
        "count_match": len(items) == num_items,
        "validation_passed": bool(state.get("validation_passed", False)),
        "agent_runs": agent_runs,
        "overall_score": judge.get("overall_score"),
        "type_ratio_score": judge.get("type_ratio_score"),
        "difficulty_match": judge.get("difficulty_match"),
        "elapsed_sec": round(elapsed, 1),
    }
    print(
        f"[{index}] {'✅통과' if row['validation_passed'] else '❌실패'} "
        f"| 문항 {row['n_items']}개 | agent 실행 {agent_runs}회 "
        f"| overall={row['overall_score']} | {row['elapsed_sec']}초"
    )
    return row


def report(rows: list[dict], budget: int) -> dict:
    n = len(rows)
    passed = [r for r in rows if r["validation_passed"]]
    rate = len(passed) / n if n else 0.0

    print("\n" + "=" * 62)
    print(f"  budget={budget} 측정 결과 (n={n})")
    print("=" * 62)
    print(f"  게이트 통과율 : {len(passed)}/{n} = {rate:.3f}")
    print(f"  개수 달성     : {sum(1 for r in rows if r['count_match'])}/{n}")
    print(f"  문항 0개      : {sum(1 for r in rows if r['n_items'] == 0)}/{n}")
    print(f"  평균 소요     : {statistics.mean([r['elapsed_sec'] for r in rows]):.1f}초")
    print(f"\n  [비교] budget={_BASELINE['budget']} (n={_BASELINE['n']}): "
          f"통과율 {_BASELINE['pass_rate']:.3f}")

    # 핵심 질문: 재시도가 실제로 실패를 회복시켰는가?
    print("\n  agent 실행 횟수별 (1회 = 재시도 없이 끝남):")
    by_runs: dict[int, list[dict]] = {}
    for r in rows:
        by_runs.setdefault(r["agent_runs"], []).append(r)
    for k in sorted(by_runs):
        grp = by_runs[k]
        ok = sum(1 for r in grp if r["validation_passed"])
        print(f"    {k}회: {len(grp)}건 (통과 {ok})")

    recovered = [r for r in passed if r["agent_runs"] >= 2]
    print(f"\n  ▶ 재시도로 회복된 통과: {len(recovered)}/{len(passed) or 1}건")
    if not passed:
        verdict = "budget을 늘려도 통과 없음 — 재시도로는 해결되지 않는 문제"
    elif recovered:
        verdict = (
            f"재시도가 실제로 값을 했다({len(recovered)}건이 2회차 이상에서 통과) — "
            "budget이 큰 프로덕션은 스모크 조건보다 유리하다"
        )
    else:
        verdict = (
            "통과한 건은 전부 1회차에 끝났다 — budget을 늘린 효과는 이 표본에서 관측되지 않음"
        )
    print(f"  ▶ 판정: {verdict}")

    return {
        "_schema": {
            "description": (
                f"budget={budget}에서 게이트 통과율·재시도 소모량 측정 (2026-08-07). "
                "19·20절의 budget=2 측정과 동일 입력·동일 num_items. 골든셋 아님 — "
                "재실행 시 덮어씀. 생성: experiments/measure_budget_effect.py"
            )
        },
        "budget": budget,
        "n": n,
        "pass_rate": round(rate, 3),
        "count_match_rate": round(sum(1 for r in rows if r["count_match"]) / n, 3) if n else None,
        "zero_item_rate": round(sum(1 for r in rows if r["n_items"] == 0) / n, 3) if n else None,
        "avg_elapsed_sec": round(statistics.mean([r["elapsed_sec"] for r in rows]), 1) if n else None,
        "baseline_budget2": _BASELINE,
        "gate": {"min_overall": _MIN_OVERALL_SCORE, "min_type_ratio": _MIN_TYPE_RATIO_SCORE},
        "recovered_by_retry": len(recovered),
        "verdict": verdict,
        "runs": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=5, help="재시도 예산 (프로덕션 기본 5)")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--num-items", type=int, default=1)
    args = parser.parse_args()

    print(f"budget={args.budget}, num_items={args.num_items}, {args.runs}회 실행")
    print(f"게이트: overall>={_MIN_OVERALL_SCORE}, type_ratio>={_MIN_TYPE_RATIO_SCORE}\n")

    rows = [run_once(i, args.budget, args.num_items) for i in range(1, args.runs + 1)]
    result = report(rows, args.budget)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
