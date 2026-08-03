#!/usr/bin/env python
"""Judge 모델 비교 실험 — 로컬 Ollama judge vs OpenAI 클라우드 judge.

`compare_models.py`는 "생성 모델"을 바꿔가며 고정 Judge(qwen2.5:7b)로 채점했다.
이 스크립트는 반대 축이다 — 생성물은 이미 사람이 라벨링해둔 골든셋
(ITEM_GOLDEN human_score 30개, STRUCTURE_GOLDEN human_label 45개)을 그대로 쓰고,
"Judge 모델"만 바꿔가며 같은 사람 라벨 대비 kappa/MAE/일치율/편향을 잰다.
생성 호출은 전혀 없음(순수 채점 비교) — compare_models.py보다 가볍고 빠르며,
OpenAI 비용도 작다(judge_one ~391 입력/~28 출력, judge_structure_one ~1448
입력/~23 출력 토큰. gpt-5.6-luna 기준 전체 105회 호출에 약 $0.10 수준 — 대화
기록/bunpil_roadmap.md 참고).

주의: get_judge_backend()가 항상 qwen2.5:7b를 반환하도록 고정돼 있던 기존 동작을
바꾸지 않는다 — JUDGE_BACKEND=openai일 때만 분기(app/common/llm/factory.py).
기본값(JUDGE_BACKEND=local)에서는 이 스크립트를 실행하지 않는 한 아무것도 바뀌지 않음.

사용법:
  .venv/bin/python experiments/compare_judge_models.py --judges qwen2.5-7b
  .venv/bin/python experiments/compare_judge_models.py --judges qwen2.5-7b,gpt-5.6-luna   # OpenAI 비용 발생
  .venv/bin/python experiments/compare_judge_models.py --judges gpt-5.6-luna,gpt-5.6-sol  # 플래그십까지 비교
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

from eval_lib import (  # noqa: E402
    ITEM_GOLDEN,
    _load_structure_golden,
    eval_judge_reliability,
    eval_structure_judge,
    score_items,
    score_structure,
)

# 후보 judge마다 필요한 환경변수 조합 — get_judge_backend()(factory.py)가 이 값으로 분기한다.
JUDGE_ENVS = {
    "qwen2.5-7b":   {"JUDGE_BACKEND": "local", "OLLAMA_JUDGE_MODEL": "qwen2.5:7b"},
    "qwen2.5-14b":  {"JUDGE_BACKEND": "local", "OLLAMA_JUDGE_MODEL": "qwen2.5:14b"},
    "gpt-5.6-luna": {"JUDGE_BACKEND": "openai", "OPENAI_JUDGE_MODEL": "gpt-5.6-luna"},
    "gpt-5.6-sol":  {"JUDGE_BACKEND": "openai", "OPENAI_JUDGE_MODEL": "gpt-5.6-sol"},
}

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_judge_comparison_results.json")


def _set_env(env: dict) -> dict:
    """환경변수를 설정하고 이전 값을 반환(복원용)."""
    prev = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    return prev


def _restore_env(prev: dict) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_judge(judge_key: str, structure_golden: list) -> dict:
    """고정된 사람 라벨(ITEM_GOLDEN/STRUCTURE_GOLDEN)을 judge_key 후보 하나로 재채점."""
    from app.common.llm import get_judge_backend

    prev = _set_env(JUDGE_ENVS[judge_key])
    try:
        judge_llm = get_judge_backend()
        scored = score_items(ITEM_GOLDEN, judge_llm)
        item_result = eval_judge_reliability(scored)
        structure_result = eval_structure_judge(score_structure(structure_golden, judge_llm))
        return {
            "item_quality_reliability": item_result,
            "structure_judge_reliability": structure_result,
        }
    finally:
        _restore_env(prev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges", type=str, required=True,
        help=f"쉼표 구분, 선택 가능: {','.join(JUDGE_ENVS)}",
    )
    args = parser.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges:
        if j not in JUDGE_ENVS:
            raise SystemExit(f"알 수 없는 judge: {j} (선택 가능: {list(JUDGE_ENVS)})")

    structure_golden = _load_structure_golden()
    print(
        f"ITEM_GOLDEN(human_score) n={len(ITEM_GOLDEN)}, "
        f"STRUCTURE_GOLDEN(human_label) n={len(structure_golden)}\n"
    )

    existing = {}
    if os.path.exists(_OUT_PATH):
        with open(_OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    for judge_key in judges:
        print(f"=== {judge_key} 채점 중 (생성 없음, 기존 골든셋 재채점) ===")
        result = run_judge(judge_key, structure_golden)
        existing[judge_key] = result

        iq = result["item_quality_reliability"]
        sr = result["structure_judge_reliability"]
        print(
            f"  문항품질: exact={iq['exact_agreement']:.3f} ±1={iq['agreement_within_1']:.3f} "
            f"kappa={iq['cohen_kappa']:.3f} (human_avg={iq['human_avg']}, llm_avg={iq['llm_avg']})"
        )
        print(
            f"  구조유사도: difficulty_match={sr['difficulty_match_agreement']:.3f} "
            f"overall_MAE={sr['overall_score_mae']:.3f}"
        )

        os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
        with open(_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {_OUT_PATH}")


if __name__ == "__main__":
    main()
