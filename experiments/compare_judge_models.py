#!/usr/bin/env python
"""Judge 모델 비교 실험 — 로컬 Ollama judge vs OpenAI/OpenRouter 클라우드 judge.

`compare_models.py`는 "생성 모델"을 바꿔가며 고정 Judge(qwen2.5:7b)로 채점했다.
이 스크립트는 반대 축이다 — 생성물은 이미 사람이 라벨링해둔 골든셋
(ITEM_GOLDEN human_score 30개, STRUCTURE_GOLDEN human_label 45개)을 그대로 쓰고,
"Judge 모델"만 바꿔가며 같은 사람 라벨 대비 kappa/MAE/일치율/편향을 잰다.
생성 호출은 전혀 없음(순수 채점 비교) — compare_models.py보다 가볍고 빠르며,
OpenAI 비용도 작다(judge_one ~391 입력/~28 출력, judge_structure_one ~1448
입력/~23 출력 토큰. gpt-5.6-luna 기준 전체 105회 호출에 약 $0.10 수준 — 대화
기록/bunpil_roadmap.md 참고).

2026-10 Judge 1차 선별(5종 비교): API 호출을 OpenRouter로 통일하기로 한 사용자
결정에 따라 JUDGE_ENVS에 "or-" 접두 후보(or-gpt-5.6-luna 등)를 추가했다 — 현재
운영 Judge(gpt-5.6-luna)도 OpenRouter 경유 버전을 넣어, 호출 경로 차이가 결과에
섞이지 않게 같은 축에서 비교한다. 이 후보들은 structure_golden 45건으로만
돌리는 것을 기본으로 가정한다(--structure-only) — ITEM_GOLDEN은 사람 라벨 1인
합성 라벨이라 2026-10 대체 예정(EVAL.md 참고)이고, 1차 선별의 판단 근거로는
structure_golden(실제 생성물·라벨러 1인)만 쓴다.

주의: get_judge_backend()가 항상 qwen2.5:7b를 반환하도록 고정돼 있던 기존 동작을
바꾸지 않는다 — JUDGE_BACKEND=openai/openrouter일 때만 분기(app/common/llm/factory.py).
기본값(JUDGE_BACKEND=local)에서는 이 스크립트를 실행하지 않는 한 아무것도 바뀌지 않음.

사용법:
  .venv/bin/python experiments/compare_judge_models.py --judges qwen2.5-7b
  .venv/bin/python experiments/compare_judge_models.py --judges qwen2.5-7b,gpt-5.6-luna   # OpenAI 비용 발생
  .venv/bin/python experiments/compare_judge_models.py --judges gpt-5.6-luna,gpt-5.6-sol  # 플래그십까지 비교
  # Judge 1차 선별(OpenRouter 5종, structure_golden만, 2회 반복 — temperature 0.7 변동 확인)
  .venv/bin/python experiments/compare_judge_models.py \\
      --judges or-gpt-5.6-luna,or-gpt-6.1-sol,or-claude-opus-5.5,or-claude-sonnet-5.5,or-gemini-3.8-flash \\
      --structure-only --repeat 2

주의(--structure-only, --repeat): --structure-only는 ITEM_GOLDEN 채점을 건너뛰고
structure_golden만 채점한다(위 2026-10 배경 참고). --repeat N은 같은 Judge로
N회 반복 채점해 회차 간 변동을 결과에 남긴다 — Judge temperature가 0.7이라
한 번만 채점하면 그 변동이 사람 라벨 대비 성능 차이인지 단순 노이즈인지 구분이
안 된다. 반복이 있으면 항목(entry id)별로 회차 평균을 먼저 낸 뒤, 그 항목 단위
값으로 신뢰구간을 낸다(evals/stats.py의 cluster_bootstrap_ci, 클러스터=항목 id —
structure_golden은 항목마다 지문이 달라 사실상 항목 단위 부트스트랩).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
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
from stats import cluster_bootstrap_ci  # noqa: E402

# 후보 judge마다 필요한 환경변수 조합 — get_judge_backend()(factory.py)가 이 값으로 분기한다.
JUDGE_ENVS = {
    "qwen2.5-7b":   {"JUDGE_BACKEND": "local", "OLLAMA_JUDGE_MODEL": "qwen2.5:7b"},
    "qwen2.5-14b":  {"JUDGE_BACKEND": "local", "OLLAMA_JUDGE_MODEL": "qwen2.5:14b"},
    "gpt-5.6-luna": {"JUDGE_BACKEND": "openai", "OPENAI_JUDGE_MODEL": "gpt-5.6-luna"},
    "gpt-5.6-sol":  {"JUDGE_BACKEND": "openai", "OPENAI_JUDGE_MODEL": "gpt-5.6-sol"},
    # 2026-10 Judge 1차 선별 — API 호출을 OpenRouter로 통일(사용자 결정). or-gpt-5.6-luna는
    # 위 gpt-5.6-luna와 같은 모델이고 호출 경로(OpenRouter vs OpenAI 직접)만 다르다.
    "or-gpt-5.6-luna":      {"JUDGE_BACKEND": "openrouter", "OPENROUTER_JUDGE_MODEL": "openai/gpt-5.6-luna"},
    "or-gpt-6.1-sol":       {"JUDGE_BACKEND": "openrouter", "OPENROUTER_JUDGE_MODEL": "openai/gpt-6.1-sol"},
    "or-claude-opus-5.5":   {"JUDGE_BACKEND": "openrouter", "OPENROUTER_JUDGE_MODEL": "anthropic/claude-opus-5.5"},
    "or-claude-sonnet-5.5": {"JUDGE_BACKEND": "openrouter", "OPENROUTER_JUDGE_MODEL": "anthropic/claude-sonnet-5.5"},
    "or-gemini-3.8-flash":  {"JUDGE_BACKEND": "openrouter", "OPENROUTER_JUDGE_MODEL": "google/gemini-3.8-flash"},
}

# OpenRouter /api/v1/models 단가(1M 토큰당 USD, 입력/출력) — 2026-10 확인. 실측 토큰 합이
# 없으므로(백엔드 generate()가 usage를 노출하지 않음) 콘솔 표의 비용은 전부 "추정"이며,
# _EST_STRUCTURE_TOKENS/_EST_ITEM_TOKENS(아래)의 가정된 평균 토큰 수에 이 단가를 곱한
# 근사값이다. qwen/local, gpt-5.6-luna(OpenAI 직접)처럼 이 표에 없는 후보는 비용을 내지 않는다.
_PRICE_PER_1M_OPENROUTER = {
    "or-gpt-5.6-luna":      (0.2, 1.2),
    "or-gpt-6.1-sol":       (2.0, 10.0),
    "or-claude-opus-5.5":   (4.0, 20.0),
    "or-claude-sonnet-5.5": (2.0, 10.0),
    "or-gemini-3.8-flash":  (0.75, 3.75),
}

# judge_structure_one 호출당 평균 토큰(입력/출력) — 2026-10-03 or-claude-opus-5.5로
# structure_golden 1건을 실측(입력 2746 / 출력 441, reasoning 토큰 포함)해 얻은 값을
# 전체 후보의 근사치로 쓴다. 토크나이저·추론(reasoning) 토큰 사용량은 모델마다 달라
# 실제 비용과 차이가 날 수 있다 — 콘솔 표에 "추정"이라고 명시하는 이유.
_EST_STRUCTURE_TOKENS = (2746, 441)
# judge_one 호출당 평균 토큰 — 위 모듈 docstring의 기존 추정치(gpt-5.6-luna 기준) 재사용.
_EST_ITEM_TOKENS = (391, 28)

_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "_judge_selection_round1.json")
# 기존 _judge_comparison_results.json(2종, repeat 없음)은 이력이라 보존 — 여기서는 읽거나
# 쓰지 않고 새 파일(_OUT_PATH)에만 저장한다.


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


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


def _structure_item_rows(scored: list[dict]) -> list[dict]:
    """score_structure() 결과 1회분을 항목별 (id, judge/human overall·difficulty_match)
    dict 리스트로 변환 — eval_structure_judge()는 집계만 내므로, CI 계산에 필요한
    항목 단위 쌍은 score_structure()가 이미 들고 있는 entry/judge를 그대로 꺼내 쓴다
    (eval_lib.py 반환값은 바꾸지 않음).

    likely_parse_failed: judge_structure()는 judge_one()과 달리 parse_failed 플래그를
    노출하지 않는다(app/modules/exam/judge.py — 파싱 실패 시 조용히 전부 기본값으로
    채운다). type_ratio_score=0.0·overall_score=0·difficulty_match=False가 동시에
    나온 경우를 "파싱 실패로 추정"하는 근사치로만 쓴다 — 진짜로 0점을 준 경우와
    구분할 수 없으므로 집계 시 반드시 '추정'으로 표기한다."""
    rows = []
    for s in scored:
        entry, judge = s["entry"], s["judge"]
        human = entry["human_label"]
        likely_parse_failed = (
            judge["type_ratio_score"] == 0.0
            and judge["overall_score"] == 0
            and judge["difficulty_match"] is False
        )
        rows.append({
            "id": entry["id"],
            "judge_overall": judge["overall_score"],
            "judge_difficulty_match": judge["difficulty_match"],
            "human_overall": human["overall_score"],
            "human_difficulty_match": human["difficulty_match"],
            "likely_parse_failed": likely_parse_failed,
        })
    return rows


def _average_repeats(repeat_rows: list[list[dict]]) -> list[dict]:
    """같은 항목(id)의 회차별 row를 평균 낸다.

    Judge temperature 0.7로 인한 회차 간 노이즈를 항목별로 먼저 지운 뒤(이 함수),
    남은 변동(항목 간 난이도·품질 차이)만으로 부트스트랩 CI를 내려는 목적이다.
    mae/bias는 (judge_overall - human_overall)의 회차 평균, difficulty_hit은
    (judge_difficulty_match == human_difficulty_match)를 0/1로 놓고 회차 평균 —
    회차마다 판정이 갈리면 0~1 사이 값이 된다."""
    n_items = len(repeat_rows[0])
    averaged = []
    for i in range(n_items):
        item_id = repeat_rows[0][i]["id"]
        diffs = [rep[i]["judge_overall"] - rep[i]["human_overall"] for rep in repeat_rows]
        hits = [
            1.0 if rep[i]["judge_difficulty_match"] == rep[i]["human_difficulty_match"] else 0.0
            for rep in repeat_rows
        ]
        averaged.append({
            "id": item_id,
            "mae": sum(abs(d) for d in diffs) / len(diffs),
            "bias": sum(diffs) / len(diffs),
            "difficulty_hit": sum(hits) / len(hits),
        })
    return averaged


def _ci_round(value: float | None, ndigits: int = 3) -> float | None:
    return round(value, ndigits) if value is not None else None


def _structure_ci(averaged_rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """항목 단위로 평균된 rows에 클러스터(=항목 id) 부트스트랩 CI를 적용.

    rows 하나가 항목 하나(회차 평균까지 끝낸 상태)이므로 클러스터 부트스트랩은
    사실상 항목 단위 일반 부트스트랩과 같다 — structure_golden은 항목마다 지문이
    달라 다른 클러스터링 기준이 없다."""
    ids = [r["id"] for r in averaged_rows]

    def _mean_of(key):
        def fn(rows: list[dict]) -> float | None:
            vals = [r[key] for r in rows]
            return sum(vals) / len(vals) if vals else None
        return fn

    out = {}
    for key, label in (
        ("mae", "overall_mae"),
        ("difficulty_hit", "difficulty_match_agreement"),
        ("bias", "bias_judge_minus_human"),
    ):
        point, lo, hi = cluster_bootstrap_ci(averaged_rows, ids, _mean_of(key), n_boot=n_boot, seed=seed)
        out[label] = {"point": _ci_round(point), "ci_lo": _ci_round(lo), "ci_hi": _ci_round(hi)}
    return out


def _estimate_cost_usd(judge_key: str, n_structure_calls: int, n_item_calls: int) -> float | None:
    """OpenRouter 단가 × 가정된 평균 토큰(위 _EST_*_TOKENS)으로 추정 비용(USD)을 낸다.
    가격표에 없는 후보(qwen/local, gpt-5.6-luna OpenAI 직접)는 None(추정 불가)."""
    price = _PRICE_PER_1M_OPENROUTER.get(judge_key)
    if price is None:
        return None
    price_in, price_out = price
    s_in, s_out = _EST_STRUCTURE_TOKENS
    i_in, i_out = _EST_ITEM_TOKENS
    cost = (
        n_structure_calls * (s_in * price_in + s_out * price_out)
        + n_item_calls * (i_in * price_in + i_out * price_out)
    ) / 1_000_000
    return cost


def run_judge(judge_key: str, structure_golden: list, repeat: int, structure_only: bool) -> dict:
    """고정된 사람 라벨(ITEM_GOLDEN/STRUCTURE_GOLDEN)을 judge_key 후보 하나로
    repeat회 재채점하고, structure_golden은 항목 단위 CI까지 계산한다."""
    from app.common.llm import get_judge_backend

    prev = _set_env(JUDGE_ENVS[judge_key])
    try:
        judge_llm = get_judge_backend()

        item_quality_runs = []
        if not structure_only:
            for _ in range(repeat):
                scored = score_items(ITEM_GOLDEN, judge_llm)
                item_quality_runs.append(eval_judge_reliability(scored))

        structure_runs = []
        structure_repeat_rows = []
        for _ in range(repeat):
            scored = score_structure(structure_golden, judge_llm)
            structure_runs.append(eval_structure_judge(scored))
            structure_repeat_rows.append(_structure_item_rows(scored))

        averaged = _average_repeats(structure_repeat_rows)
        structure_ci = _structure_ci(averaged)
        likely_parse_failed = sum(
            1 for rep in structure_repeat_rows for row in rep if row["likely_parse_failed"]
        )

        n_structure_calls = repeat * len(structure_golden)
        n_item_calls = 0 if structure_only else repeat * len(ITEM_GOLDEN)

        result = {
            "judge_model_env": JUDGE_ENVS[judge_key],
            "repeat": repeat,
            "structure_only": structure_only,
            "structure_judge_reliability_runs": structure_runs,
            "structure_ci": structure_ci,
            "structure_likely_parse_failed": likely_parse_failed,
            "n_structure_calls": n_structure_calls,
            "n_item_calls": n_item_calls,
            "estimated_cost_usd": _estimate_cost_usd(judge_key, n_structure_calls, n_item_calls),
        }
        if not structure_only:
            result["item_quality_reliability_runs"] = item_quality_runs
        return result
    finally:
        _restore_env(prev)


def _print_summary_row(judge_key: str, result: dict) -> None:
    ci = result["structure_ci"]
    mae = ci["overall_mae"]
    dma = ci["difficulty_match_agreement"]
    bias = ci["bias_judge_minus_human"]
    cost = result["estimated_cost_usd"]
    cost_str = f"${cost:.4f}(추정)" if cost is not None else "N/A"
    print(
        f"  {judge_key:24s} "
        f"MAE={mae['point']:.3f}[{mae['ci_lo']:.3f},{mae['ci_hi']:.3f}] "
        f"난이도일치={dma['point']:.3f}[{dma['ci_lo']:.3f},{dma['ci_hi']:.3f}] "
        f"편향(judge-human)={bias['point']:+.3f}[{bias['ci_lo']:+.3f},{bias['ci_hi']:+.3f}] "
        f"파싱실패(추정)={result['structure_likely_parse_failed']}건 "
        f"호출={result['n_structure_calls'] + result['n_item_calls']}회 비용={cost_str}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges", type=str, required=True,
        help=f"쉼표 구분, 선택 가능: {','.join(JUDGE_ENVS)}",
    )
    parser.add_argument(
        "--structure-only", action="store_true",
        help="ITEM_GOLDEN(합성 라벨, 2026-10 대체 예정) 채점을 건너뛰고 structure_golden만 채점",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="같은 Judge로 N회 반복 채점(temperature 0.7 회차 변동 확인, 기본 1)",
    )
    args = parser.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges:
        if j not in JUDGE_ENVS:
            raise SystemExit(f"알 수 없는 judge: {j} (선택 가능: {list(JUDGE_ENVS)})")
    if args.repeat < 1:
        raise SystemExit("--repeat은 1 이상이어야 합니다.")

    structure_golden = _load_structure_golden()
    print(
        f"ITEM_GOLDEN(human_score) n={len(ITEM_GOLDEN)}, "
        f"STRUCTURE_GOLDEN(human_label) n={len(structure_golden)}, "
        f"repeat={args.repeat}, structure_only={args.structure_only}\n"
    )

    existing = {}
    if os.path.exists(_OUT_PATH):
        with open(_OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    for judge_key in judges:
        print(f"=== {judge_key} 채점 중 (생성 없음, 기존 골든셋 재채점, {args.repeat}회) ===")
        result = run_judge(judge_key, structure_golden, args.repeat, args.structure_only)
        result["run_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result["code_version"] = _code_version()
        existing[judge_key] = result

        _print_summary_row(judge_key, result)

        os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
        with open(_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {_OUT_PATH} (기존 _judge_comparison_results.json은 이력 보존용으로 유지)")


if __name__ == "__main__":
    main()
