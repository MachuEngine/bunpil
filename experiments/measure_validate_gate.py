#!/usr/bin/env python
"""validate_node 게이트 임계값 재보정용 실측 (2026-08-04, 1회성).

배경: `overall_score >= 4` / `type_ratio_score >= 0.7`은 2026-07-06 도입 후 한 번도
재검토된 적이 없고 근거 기록도 없었다(그 사이 Judge 프롬프트는 4회 이상 재보정됨).
"Judge가 실제로 어떤 점수를 주는가"를 측정한 적이 없어 임계값을 검증할 수 없었다.

측정 방법: STRUCTURE_GOLDEN 45건(사람 라벨이 있는 실제 생성물)을 프로덕션 Judge
(gpt-5.6-luna)로 채점해 점수 분포와 임계값별 통과율을 구한다. 생성은 하지 않고
기존 골든셋을 재채점만 하므로 LLM 호출은 45회(약 $0.06).

결과 요약(raw: `data/golden/_validate_gate_calibration.json`):
  - overall_score 분포 {1:10, 2:12, 3:18, 4:5, 5:0} — Judge가 5점을 한 번도 주지
    않았고 4점도 5건뿐. `>=4` 통과율 8.9%로 사실상 도달 불가.
  - type_ratio_score는 45건 중 18건(40%)이 정확히 0.5 — 임계값 0.7이 0.5와 0.75
    사이를 갈라 이 40%를 통째로 탈락시킴.
  - Judge 프롬프트가 "정답"으로 가르치는 few-shot 6개 중 옛 게이트 통과는 1개뿐.
  → `overall>=3`, `type_ratio>=0.5`로 재보정(통과율 48.9%). 상세는
    `app/modules/exam/graph.py`의 임계값 상수 주석, EVAL.md 15절 참고.

실행:
    CHROMA_PERSIST_DIR=./chroma_db python experiments/measure_validate_gate.py
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm import get_judge_backend
from app.common.llm.tracing import init_langsmith_project
from app.modules.exam.judge import judge_structure

init_langsmith_project()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(_ROOT, "data", "golden", "structure_golden.json")
OUT_PATH = os.path.join(_ROOT, "data", "golden", "_validate_gate_calibration.json")

# 프로덕션과 다른 Judge로 재면 임계값 근거가 무의미해지므로 명시적으로 막는다.
if os.getenv("JUDGE_BACKEND") != "openai":
    sys.exit(
        f"JUDGE_BACKEND={os.getenv('JUDGE_BACKEND')!r} — 프로덕션 Judge(openai)로 "
        "측정해야 의미가 있다. .env를 확인할 것."
    )


def measure() -> list[dict]:
    entries = [
        e for e in json.load(open(GOLDEN_PATH, encoding="utf-8"))["entries"]
        if e.get("human_label")
    ]
    llm = get_judge_backend()
    print(f"Judge: {type(llm).__name__} model={getattr(llm, 'model', '?')} / 대상 {len(entries)}건\n")

    rows = []
    for i, e in enumerate(entries, 1):
        try:
            j = judge_structure(e["passage_text"], e["generated_items"], llm)
        except Exception as ex:  # 개별 실패는 건너뛰되 조용히 넘기지 않는다
            print(f"[{i}/{len(entries)}] {e['id']} 실패: {ex}")
            continue
        hl = e["human_label"]
        rows.append({
            "id": e["id"],
            "num_items": e.get("num_items"),
            "n_generated": len(e.get("generated_items", [])),
            "judge": j,
            "human": {
                "overall_score": hl.get("overall_score"),
                "type_ratio_score": hl.get("type_ratio_score"),
                "difficulty_match": hl.get("difficulty_match"),
            },
        })
        print(f"[{i}/{len(entries)}] {e['id']}: judge overall={j['overall_score']} "
              f"type={j['type_ratio_score']} diff={j['difficulty_match']} "
              f"| human overall={hl.get('overall_score')}")
    return rows


def report(rows: list[dict]) -> dict:
    n = len(rows)
    jo = [r["judge"]["overall_score"] for r in rows]
    ho = [r["human"]["overall_score"] for r in rows]

    print("\n" + "=" * 58)
    print(f"  validate 게이트 임계값 실측 (n={n})")
    print("=" * 58)
    print("overall_score 분포")
    print(f"  judge: {dict(sorted(collections.Counter(jo).items()))} 평균={statistics.mean(jo):.2f}")
    print(f"  human: {dict(sorted(collections.Counter(ho).items()))} 평균={statistics.mean(ho):.2f}")
    print(f"  편향={statistics.mean(jo) - statistics.mean(ho):+.2f} "
          f"MAE={statistics.mean([abs(a - b) for a, b in zip(jo, ho)]):.3f}")
    print(f"type_ratio_score(judge) 분포: "
          f"{dict(sorted(collections.Counter(r['judge']['type_ratio_score'] for r in rows).items()))}")

    print("\n임계값별 통과율 (judge 3개 지표만, count_match·all_approved 제외)")
    table = {}
    for overall_thr in (2, 3, 4, 5):
        for type_thr in (0.5, 0.7):
            c = sum(
                1 for r in rows
                if r["judge"]["overall_score"] >= overall_thr
                and r["judge"]["type_ratio_score"] >= type_thr
                and r["judge"]["difficulty_match"]
            )
            table[f"overall>={overall_thr},type>={type_thr}"] = round(c / n, 3)
            mark = ""
            if (overall_thr, type_thr) == (4, 0.7):
                mark = "  ← 옛 기준"
            elif (overall_thr, type_thr) == (3, 0.5):
                mark = "  ← 새 기준"
            print(f"  overall>={overall_thr}, type>={type_thr}: {c}/{n} = {c/n:.3f}{mark}")

    return {
        "_schema": {
            "description": (
                "validate_node 게이트 임계값 재보정 실측 (2026-08-04). "
                "STRUCTURE_GOLDEN 45건을 프로덕션 Judge(gpt-5.6-luna)로 재채점해 "
                "점수 분포·임계값별 통과율을 측정. 골든셋 아님 — 재실행 시 덮어씀. "
                "생성 스크립트: experiments/measure_validate_gate.py"
            )
        },
        "n": n,
        "judge_model": os.getenv("OPENAI_JUDGE_MODEL"),
        "judge_overall_distribution": dict(sorted(collections.Counter(jo).items())),
        "human_overall_distribution": dict(sorted(collections.Counter(ho).items())),
        "judge_overall_mean": round(statistics.mean(jo), 3),
        "human_overall_mean": round(statistics.mean(ho), 3),
        "bias": round(statistics.mean(jo) - statistics.mean(ho), 3),
        "mae": round(statistics.mean([abs(a - b) for a, b in zip(jo, ho)]), 3),
        "pass_rate_by_threshold": table,
        "per_item": rows,
    }


if __name__ == "__main__":
    rows = measure()
    if not rows:
        sys.exit("측정 결과가 없습니다 — Judge 호출이 전부 실패했는지 확인할 것.")
    result = report(rows)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
