#!/usr/bin/env python
"""모델 비교 실험 (bunpil_roadmap.md 로드맵 항목 5).

Qwen2.5-7B / Qwen2.5-14B / Llama3.1-8B / GPT-4o-mini를 각각 "출제 생성 모델"로
교체해 동일 passage_text 샘플로 문항을 생성시키고, 채점은 고정된 하나의
Judge(qwen2.5:7b, JUDGE_TPL 5점 앵커 + STRUCTURE_JUDGE_TPL 3점 앵커 포함 최신
상태)로 통일해 편향을 배제한다. 생성 모델이 스스로를 채점하지 않도록 Judge는
항상 별도 프로세스에서 qwen2.5:7b로 고정 호출한다.

측정 항목: 문항 품질(JUDGE_TPL), 구조 유사도(STRUCTURE_JUDGE_TPL), 생성 속도,
실패율(0문항/예외).

사용법:
  .venv/bin/python scripts/compare_models.py --models qwen2.5-7b,qwen2.5-14b,llama3.1-8b
  .venv/bin/python scripts/compare_models.py --models gpt-4o-mini   # OpenAI 비용 발생
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db")

from gen_structure_golden import PASSAGE_SAMPLES  # noqa: E402

# 15개 대표 샘플 — num_items 3/5/7 및 단일·다중 지문형을 고르게 섞음
_SAMPLE_IDS = [
    "str_002", "str_005", "str_006", "str_009", "str_013",
    "str_016", "str_020", "str_023", "str_027", "str_030",
    "str_033", "str_038", "str_040", "str_044", "str_046",
]
_SAMPLES_BY_ID = {s["id"]: s for s in PASSAGE_SAMPLES}
SAMPLES = [_SAMPLES_BY_ID[i] for i in _SAMPLE_IDS]

# 각 후보를 "출제 생성 모델"로 쓸 때 필요한 환경변수 조합. 어느 것도 baseline 취급하지 않음.
MODEL_ENVS = {
    "qwen2.5-7b":  {"LLM_BACKEND": "local", "OLLAMA_MODEL": "qwen2.5:7b"},
    "qwen2.5-14b": {"LLM_BACKEND": "local", "OLLAMA_MODEL": "qwen2.5:14b"},
    "llama3.1-8b": {"LLM_BACKEND": "local", "OLLAMA_MODEL": "llama3.1:8b"},
    "gpt-4o-mini": {"LLM_BACKEND": "openai", "OPENAI_MODEL": "gpt-4o-mini"},
}

# Judge는 항상 이 값으로 고정 — 생성 모델 전환과 무관
JUDGE_ENV = {"OLLAMA_JUDGE_MODEL": "qwen2.5:7b"}

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_model_comparison_results.json")


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


def generate_one(sample: dict, budget: int = 1) -> dict:
    """단일 샘플에 대해 현재 LLM_BACKEND/모델 설정으로 문항을 생성. 시간·실패 여부 기록."""
    from app.modules.exam import ExamSpec, get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    spec: ExamSpec = {
        "passage_text": sample["passage_text"],
        "standards": sample.get("standards", []),
        "num_items": sample["num_items"],
    }
    init_session()
    graph = get_exam_graph()
    start = time.monotonic()
    error = None
    items = []
    try:
        state = graph.invoke(
            {
                "spec": spec,
                "budget": budget,
                "draft_items": [],
                "agent_messages": [],
                "validation_passed": False,
                "similarity_judge_result": {},
                "error": "",
            }
        )
        items = get_draft_items()
    except Exception as e:
        error = str(e)
    elapsed = time.monotonic() - start

    generated_items = [
        {
            "question": it.get("question", ""),
            "options": it.get("options", []),
            "answer": it.get("answer", ""),
            "item_type": it.get("item_type", ""),
            "difficulty": it.get("difficulty", ""),
        }
        for it in items
    ]
    return {
        "id": sample["id"],
        "passage_text": sample["passage_text"],
        "num_items": sample["num_items"],
        "generated_items": generated_items,
        "elapsed_sec": round(elapsed, 1),
        "error": error,
    }


def judge_all(generations: list) -> list:
    """고정 Judge(qwen2.5:7b)로 문항 품질 + 구조 유사도를 채점."""
    from eval_exam import STRUCTURE_JUDGE_TPL, JUDGE_TPL, judge_one, judge_structure_one  # noqa: F401
    from app.common.llm import get_judge_backend

    prev = _set_env(JUDGE_ENV)
    prev_backend = _set_env({"LLM_BACKEND": "local"})
    try:
        judge_llm = get_judge_backend()
        results = []
        for gen in generations:
            item_scores = [judge_one(it, judge_llm) for it in gen["generated_items"]] if gen["generated_items"] else []
            avg_overall = round(sum(s["overall"] for s in item_scores) / len(item_scores), 2) if item_scores else 0.0

            structure_entry = {"passage_text": gen["passage_text"], "generated_items": gen["generated_items"]}
            structure_score = judge_structure_one(structure_entry, judge_llm) if gen["generated_items"] else {
                "type_ratio_score": 0.0, "difficulty_match": False, "overall_score": 0,
            }

            results.append({
                **gen,
                "item_quality_avg_overall": avg_overall,
                "item_quality_scores": item_scores,
                "structure_score": structure_score,
            })
        return results
    finally:
        _restore_env(prev)
        _restore_env(prev_backend)


def run_model(model_key: str, budget: int) -> list:
    env = MODEL_ENVS[model_key]
    prev = _set_env(env)
    try:
        generations = []
        for i, sample in enumerate(SAMPLES, 1):
            print(f"  [{i}/{len(SAMPLES)}] {sample['id']} (num_items={sample['num_items']}) 생성 중...")
            gen = generate_one(sample, budget=budget)
            n = len(gen["generated_items"])
            status = f"실패({gen['error']})" if gen["error"] else f"{n}개"
            print(f"    -> {status}, {gen['elapsed_sec']}s")
            generations.append(gen)
        return generations
    finally:
        _restore_env(prev)


def summarize(model_key: str, judged: list) -> dict:
    n = len(judged)
    fail = sum(1 for g in judged if g["error"] or len(g["generated_items"]) == 0)
    ok = [g for g in judged if not g["error"] and g["generated_items"]]
    avg_time = round(sum(g["elapsed_sec"] for g in judged) / n, 1) if n else 0.0
    avg_quality = round(sum(g["item_quality_avg_overall"] for g in ok) / len(ok), 2) if ok else 0.0
    avg_structure = round(sum(g["structure_score"]["overall_score"] for g in ok) / len(ok), 2) if ok else 0.0
    return {
        "model": model_key,
        "n": n,
        "fail_count": fail,
        "fail_rate": round(fail / n, 3) if n else 0.0,
        "avg_elapsed_sec": avg_time,
        "avg_item_quality": avg_quality,
        "avg_structure_score": avg_structure,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, help="쉼표 구분: qwen2.5-7b,qwen2.5-14b,llama3.1-8b,gpt-4o-mini")
    parser.add_argument("--budget", type=int, default=1)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_ENVS:
            raise SystemExit(f"알 수 없는 모델: {m} (선택 가능: {list(MODEL_ENVS)})")

    existing = {}
    if os.path.exists(_OUT_PATH):
        with open(_OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    for model_key in models:
        print(f"\n=== {model_key} 생성 시작 ({len(SAMPLES)}개 샘플, budget={args.budget}) ===")
        generations = run_model(model_key, args.budget)
        print(f"=== {model_key} 채점 중 (Judge=qwen2.5:7b 고정) ===")
        judged = judge_all(generations)
        summary = summarize(model_key, judged)
        existing[model_key] = {"summary": summary, "generations": judged}
        print(f"=== {model_key} 완료: {json.dumps(summary, ensure_ascii=False)} ===")

        os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
        with open(_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {_OUT_PATH}")


if __name__ == "__main__":
    main()
