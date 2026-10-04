#!/usr/bin/env python
"""VLM figure Judge 신뢰도 골든셋 생성기.

data/golden/vlm_extraction_golden.json의 figure 15건에 대해 VLM 추출 → Judge
채점을 **한 번만** 실행해 결과를 고정(freeze)하고, data/golden/vlm_figure_judge_golden.json
에 human_label: null 상태로 저장한다. structure_golden.json과 정확히 같은 목적 —
"이 Judge 점수가 사람 판단과 얼마나 일치하는가"를 재기 위해 같은 (VLM 출력, Judge 점수)
쌍에 사람이 라벨을 매길 수 있게 고정해둔다.

사람이 human_label(1~5)을 채운 뒤 evals/eval_vlm_judge_reliability.py로 kappa를 계산한다.

재실행하면 VLM·Judge를 다시 호출해 덮어쓴다(비용 발생) — 단, 기존 파일에 human_label이
하나라도 채워져 있으면 라벨 유실을 막기 위해 실행을 중단한다(golden_gen/gen_item_quality_golden.py
의 _has_existing_labels와 같은 가드). 라벨을 다시 매기려면 OUT_PATH를 직접 지우고 실행할 것.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.common.llm import get_judge_backend, get_vlm_backend
from evals.eval_vlm import score_figure_entry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "data", "golden")
SOURCE_PATH = os.path.join(GOLDEN_DIR, "vlm_extraction_golden.json")
OUT_PATH = os.path.join(GOLDEN_DIR, "vlm_figure_judge_golden.json")


def _has_existing_labels(path: str) -> bool:
    """이미 human_label이 하나라도 채워진 기존 파일이 있으면 True(라벨 유실 방지용 가드).
    golden_gen/gen_item_quality_golden.py의 같은 이름 함수와 같은 목적."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return any(e.get("human_label") is not None for e in data.get("entries", []))


async def _call_with_retry(coro_fn, *, retries: int = 5, base_wait: float = 20.0):
    from openai import RateLimitError

    for attempt in range(retries):
        try:
            return await coro_fn()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"    [429] {wait:.0f}초 대기 후 재시도 ({attempt + 1}/{retries})...")
            await asyncio.sleep(wait)


async def main():
    if _has_existing_labels(OUT_PATH):
        print(f"중단 — {OUT_PATH}에 이미 채워진 human_label이 있습니다. 라벨 유실 방지를 위해 덮어쓰지 않습니다.")
        sys.exit(1)

    with open(SOURCE_PATH, encoding="utf-8") as f:
        source = json.load(f)
    figures = [e for e in source["entries"] if e["category"] == "figure"]

    vlm = get_vlm_backend()
    judge = get_judge_backend()

    entries = []
    for e in figures:
        img_path = os.path.join(GOLDEN_DIR, e["image"])
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        print(f"[{e['id']}] VLM 추출 중...")
        raw = await _call_with_retry(lambda: vlm.extract_text(image_bytes, "image/png"))

        print(f"[{e['id']}] Judge 채점 중...")
        result = await score_figure_entry(e["figure_summary"], raw, judge, call_with_retry=_call_with_retry)
        description = result["fig_text"] or ""
        judge_score = result["judge_score"]

        entries.append({
            "id": e["id"],
            "figure_kind": e["figure_kind"],
            "figure_summary": e["figure_summary"],
            "vlm_full_output": raw,
            "vlm_figure_description": description,
            "judge_score": judge_score,
            "human_label": None,
        })
        await asyncio.sleep(1.5)

    doc = {
        "_schema": {
            "description": (
                "VLM figure 채점(evals/eval_vlm.py 2번 축)의 Judge 신뢰도 검증용. "
                "structure_golden.json과 같은 목적 — 고정된 (VLM 출력, Judge 점수) 쌍에 "
                "사람이 human_label(1~5)을 매기면 evals/eval_vlm_judge_reliability.py가 "
                "Cohen's kappa·일치율을 계산한다."
            ),
            "entry_fields": {
                "id": "vlm_extraction_golden.json의 figure id와 동일",
                "figure_summary": "채점 기준(핵심 사실) — vlm_extraction_golden.json과 동일, 이미 사람 검수됨",
                "vlm_full_output": "이 골든셋 생성 시점에 실제로 받은 VLM 응답 전문(고정값, 재현용)",
                "vlm_figure_description": "vlm_full_output에서 [자료: ...] 또는 마크다운 표 블록만 추출",
                "judge_score": "get_judge_backend()가 매긴 1~5점(고정값)",
                "human_label": "사람이 매기는 1~5점. null이면 아직 라벨링 전 — eval 스크립트가 스킵",
            },
            "provenance": "vlm_extraction_golden.json(합성 이미지, 하드룰 1)의 figure 15건을 이 스크립트로 1회 채점해 고정.",
        },
        "entries": entries,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] {len(entries)}건 → {OUT_PATH}")
    print("human_label을 채운 뒤 evals/eval_vlm_judge_reliability.py를 실행하세요.")


if __name__ == "__main__":
    asyncio.run(main())
