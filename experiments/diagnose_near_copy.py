#!/usr/bin/env python
"""`overall_score=1`(근접 복사) 실패의 성격이 어휘적인가 의미적인가 진단 (2026-08-07, 1회성).

배경: 19절 측정에서 게이트 실패 3건이 **전부 `overall=1`**이었다. 루브릭상 1점은
"원문 단순 복사 또는 완전 중복에 의존"이다(`type_ratio`·`difficulty_match`는 6/6 정상이라
실패 원인이 이것 하나로 좁혀졌다).

그런데 `save_item`의 복사 게이트는 **어휘 기반**(bigram containment ≥ 0.90)이라,
저장된 문항은 정의상 containment < 0.90이다. 그래서 갈림길이 생긴다:

  - 실패 문항의 containment가 **0.73~0.90 구간**이면 → 어휘적 근접 복사.
    임계값을 낮추는 것만으로 저장 단계에서 잡을 수 있다(비용 0).
    (0.73은 기존 실측 근거: "정상적인 주제-유사 변형은 ~0.73 이하" — tools.py 주석)
  - **0.73 미만**이면 → 단어는 다른데 묻는 게 같은 **의미적** 근접 복제.
    어휘 임계값을 낮춰봐야 정상 변형만 오탐하고 못 잡는다 → 임베딩 유사도 게이트가 필요.

그래서 두 지표를 함께 잰다: bigram containment(게이트와 동일 계산) + 임베딩 코사인 유사도.
LLM Judge 점수와 나란히 놓고 어느 쪽이 `overall=1`을 설명하는지 본다.

실행:
    CHROMA_PERSIST_DIR=./chroma_db python experiments/diagnose_near_copy.py [--runs 6]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:14b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm.tracing import init_langsmith_project

init_langsmith_project()

from app.common.llm import get_judge_backend
from app.common.rag import get_embedder
from app.modules.exam import ExamSpec, get_exam_graph
from app.modules.exam.judge import judge_structure
from app.modules.exam.tools import (
    _PASSAGE_COPY_THRESHOLD,
    _bigrams,
    get_draft_items,
    init_session,
)

# 스모크 테스트와 동일한 입력을 써야 19절 결과와 비교 가능하다.
from scripts.test_exam import PASSAGE_TEXT  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "data", "golden", "_near_copy_diagnosis.json")

# tools.py 주석에 기록된 실측 분포 — 판정 기준선으로 쓴다.
_NORMAL_VARIANT_CEILING = 0.73  # "정상적인 주제-유사 변형은 ~0.73 이하"


def containment(question: str, passage: str) -> float:
    """save_item의 `_check_similarity`와 **동일한 계산**(질문 bigram 기준 포함률)."""
    qb = _bigrams(question)
    pb = _bigrams(passage)
    if not qb or not pb:
        return 0.0
    return len(qb & pb) / len(qb)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def run_once(index: int, judge_llm, embedder) -> list[dict]:
    spec: ExamSpec = {"passage_text": PASSAGE_TEXT, "num_items": 1}
    init_session()
    graph = get_exam_graph()
    graph.invoke(
        {
            "spec": spec,
            "budget": 2,
            "draft_items": [],
            "agent_messages": [],
            "validation_passed": False,
            "similarity_judge_result": {},
        }
    )
    items = get_draft_items()
    if not items:
        print(f"[{index}] 문항 0개 — 건너뜀")
        return []

    judge = judge_structure(PASSAGE_TEXT, items, judge_llm)
    rows = []
    for it in items:
        q = it.get("question", "")
        cont = containment(q, PASSAGE_TEXT)
        vecs = embedder.embed([PASSAGE_TEXT, q])
        cos = cosine(vecs[0], vecs[1])
        rows.append({
            "run": index,
            "question": q,
            "overall_score": judge["overall_score"],
            "type_ratio_score": judge["type_ratio_score"],
            "difficulty_match": judge["difficulty_match"],
            "containment": round(cont, 3),
            "embed_cosine": round(cos, 3),
        })
        print(
            f"[{index}] overall={judge['overall_score']} "
            f"containment={cont:.3f} cosine={cos:.3f} | {q[:50]}"
        )
    return rows


def report(rows: list[dict]) -> dict:
    low = [r for r in rows if r["overall_score"] <= 1]
    high = [r for r in rows if r["overall_score"] >= 3]

    def avg(xs, key):
        return round(sum(x[key] for x in xs) / len(xs), 3) if xs else None

    print("\n" + "=" * 62)
    print(f"  근접 복사 실패의 성격 진단 (문항 n={len(rows)})")
    print("=" * 62)
    print(f"{'구분':<22}{'n':<5}{'containment':<14}{'embed_cosine':<14}")
    print(f"{'overall<=1 (근접복사)':<22}{len(low):<5}{str(avg(low,'containment')):<14}{str(avg(low,'embed_cosine')):<14}")
    print(f"{'overall>=3 (통과권)':<22}{len(high):<5}{str(avg(high,'containment')):<14}{str(avg(high,'embed_cosine')):<14}")

    verdict = "판정 불가 — overall<=1 표본 없음"
    if low:
        in_band = [r for r in low if _NORMAL_VARIANT_CEILING <= r["containment"] < _PASSAGE_COPY_THRESHOLD]
        below = [r for r in low if r["containment"] < _NORMAL_VARIANT_CEILING]
        print(f"\noverall<=1 문항의 containment 분포 (게이트는 >={_PASSAGE_COPY_THRESHOLD}에서만 차단):")
        print(f"  {_NORMAL_VARIANT_CEILING}~{_PASSAGE_COPY_THRESHOLD} 구간(임계값 조정으로 잡힘): {len(in_band)}/{len(low)}")
        print(f"  {_NORMAL_VARIANT_CEILING} 미만(어휘로는 못 잡음)          : {len(below)}/{len(low)}")
        if len(in_band) > len(below):
            verdict = "어휘적 — save_item 임계값을 낮추면 저장 단계에서 차단 가능(A안)"
        elif below:
            verdict = "의미적 — 어휘 임계값으로는 못 잡음, 임베딩 유사도 게이트 필요(B안)"
        else:
            verdict = "혼재 — 표본을 늘려 재판정 필요"
    print(f"\n▶ 판정: {verdict}")

    # 오탐 위험: 통과권 문항이 낮춘 임계값에 걸리는지도 함께 봐야 한다.
    if high:
        risky = [r for r in high if r["containment"] >= _NORMAL_VARIANT_CEILING]
        print(f"▶ 오탐 위험: 통과권(overall>=3) 중 containment>={_NORMAL_VARIANT_CEILING}인 것 "
              f"{len(risky)}/{len(high)} — 임계값을 낮추면 이들이 함께 차단된다")

    return {
        "_schema": {
            "description": (
                "overall_score=1(근접 복사) 실패가 어휘적인지 의미적인지 진단 (2026-08-07). "
                "containment는 save_item 게이트와 동일 계산, embed_cosine은 BGE-M3 기준. "
                "골든셋 아님 — 재실행 시 덮어씀. 생성: experiments/diagnose_near_copy.py"
            )
        },
        "n_items": len(rows),
        "gate_threshold": _PASSAGE_COPY_THRESHOLD,
        "normal_variant_ceiling": _NORMAL_VARIANT_CEILING,
        "avg_containment_low": avg(low, "containment"),
        "avg_containment_high": avg(high, "containment"),
        "avg_cosine_low": avg(low, "embed_cosine"),
        "avg_cosine_high": avg(high, "embed_cosine"),
        "verdict": verdict,
        "items": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    judge_llm = get_judge_backend()
    embedder = get_embedder()
    print(f"실행 {args.runs}회 (num_items=1, budget=2 — 19절과 동일 조건)\n")

    all_rows = []
    for i in range(1, args.runs + 1):
        all_rows.extend(run_once(i, judge_llm, embedder))

    if not all_rows:
        sys.exit("문항이 하나도 생성되지 않아 진단할 수 없습니다.")

    result = report(all_rows)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")
