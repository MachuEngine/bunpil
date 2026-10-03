#!/usr/bin/env python
"""문항 품질 run 파일(golden_gen/gen_item_quality_golden.py `generate`가 만든
`data/golden/_item_quality_runs/{model}.jsonl`, smoke 제외)을 Judge로 채점하고
모델 간 변별력(discrimination)을 분석한다.

서브커맨드:
  judge           run의 객관식 문항을 get_judge_backend()로 채점 →
                  data/golden/_item_quality_judged/<judge-slug>/<model>.jsonl에 append(이어서 실행)
  discrimination  judged 캐시 + run + item_quality_inputs.json(case_type·format)을 모아
                  지문×모델별 지표를 내고, 모델 간 변별력(ceiling/discriminating)을 집계

하드룰 2(마스킹은 모델 호출 이전): run 파일은 이미 그 순서로 생성된 결과물이고, 이 스크립트는
run을 다시 생성하지 않는다(채점만) — 새로 모델을 호출하는 지점은 judge_one() 하나뿐이고,
입력은 run에 이미 저장된 문항(생성물)이라 PII 마스킹 대상인 passage_text를 다루지 않는다.
하드룰 4(로그·캐시에 원문 금지): 콘솔 출력은 모델명·id·개수만. 캐시 파일에도 question 등
문항 원문을 적지 않고 scores(점수)만 적는다(gen_item_quality_golden.py의 run 파일과 달리
이 캐시는 문항 원문 보관 목적이 아니라 채점 결과 재사용용).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# golden_gen/gen_item_quality_golden.py는 수정하지 않고 공개 헬퍼(load_run_records, 언더스코어
# 없음)만 가져다 쓴다 — jsonl 한 줄 = 한 dict인 범용 로더라 judged 캐시 파일을 읽는 데도 그대로
# 쓸 수 있다(이름은 run 전용처럼 보이지만 구현은 범용).
from golden_gen.gen_item_quality_golden import load_run_records

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
_INPUTS_PATH = os.path.join(_GOLDEN_DIR, "item_quality_inputs.json")
_RUNS_DIR = os.path.join(_GOLDEN_DIR, "_item_quality_runs")
_JUDGED_DIR = os.path.join(_GOLDEN_DIR, "_item_quality_judged")

# golden_gen/gen_item_quality_golden.py `_MODELS`와 동일한 값 — 그 상수는 언더스코어 프라이빗이라
# 모듈 경계를 넘어 직접 가져오지 않고 같은 값을 여기 상수로 둔다(두 파일 중 하나만 바뀌면
# 테스트에서 드러남).
_MODELS = ("qwen2.5-14b", "gpt-6-luna", "gemini-3.8-flash")

_CRITERIA = ("정답유일성", "오답매력도", "근거성")


# ── 공용 경로/캐시 헬퍼 ──────────────────────────────────────────────────

def _judge_slug(judge_model: str) -> str:
    return judge_model.replace("/", "__")


def _run_path(model: str) -> str:
    return os.path.join(_RUNS_DIR, f"{model}.jsonl")


def _judged_path(judge_model: str, model: str) -> str:
    return os.path.join(_JUDGED_DIR, _judge_slug(judge_model), f"{model}.jsonl")


# ── judge: 채점 대상 선정 + 캐시 스킵 ────────────────────────────────────

def mc_items_for_run(run: dict) -> list[dict]:
    """채점 대상 문항 선정 — error가 있는 run은 전부 건너뛰고(스펙), 나머지는 객관식만
    (JUDGE_TPL이 객관식 기준이라 서술형은 채점 대상이 아님)."""
    if run.get("error"):
        return []
    return [it for it in run.get("items", []) if it.get("item_type") == "객관식"]


def judged_keys(records: list[dict]) -> set[tuple]:
    """이미 채점 완료된 (run_id, item_id, repeat) 키 집합 — 이어서 실행 시 스킵 판정에 사용."""
    return {(r["run_id"], r["item_id"], r["repeat"]) for r in records}


def cmd_judge(args: argparse.Namespace) -> None:
    # app.* import 전에 env를 설정한다(gen_item_quality_golden.py cmd_generate와 같은 관례) —
    # get_judge_backend()가 이 두 값을 읽어 백엔드를 고른다.
    os.environ["JUDGE_BACKEND"] = "openrouter"
    os.environ["OPENROUTER_JUDGE_MODEL"] = args.judge_model

    from app.common.llm import get_judge_backend
    from evals.eval_lib import judge_one

    judge_llm = get_judge_backend()
    models = args.models or list(_MODELS)
    out_dir = os.path.join(_JUDGED_DIR, _judge_slug(args.judge_model))
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Judge: {args.judge_model} / 대상 모델: {', '.join(models)} / repeat={args.repeat} ===")

    for model in models:
        runs = load_run_records(_run_path(model))
        out_path = _judged_path(args.judge_model, model)
        done = judged_keys(load_run_records(out_path))

        n_judged = 0
        n_skipped = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for run in runs:
                for item in mc_items_for_run(run):
                    for rep in range(args.repeat):
                        key = (run["id"], item["item_id"], rep)
                        if key in done:
                            n_skipped += 1
                            continue
                        scores = judge_one(item, judge_llm)
                        record = {
                            "run_id": run["id"],
                            "model": model,
                            "item_id": item["item_id"],
                            "repeat": rep,
                            "scores": scores,
                            "judge_model": args.judge_model,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        n_judged += 1
        print(f"[{model}] 신규 채점 {n_judged}건, 캐시 스킵 {n_skipped}건 → {out_path}")


# ── discrimination: 지문×모델 지표 + 변별력 집계 ─────────────────────────

def load_inputs_meta(path: str) -> dict:
    """id -> {"case_type", "format"} — item_quality_inputs.json 전체 로드."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {e["id"]: {"case_type": e["case_type"], "format": e["format"]} for e in data["entries"]}


def group_judged_by_run(judged_records: list[dict]) -> dict:
    """judged 캐시 레코드(한 모델분)를 run_id -> scores 리스트로 묶는다(반복 포함 전부)."""
    grouped: dict[str, list[dict]] = {}
    for r in judged_records:
        grouped.setdefault(r["run_id"], []).append(r["scores"])
    return grouped


def compute_metrics(run: dict, judged_scores: list[dict]) -> dict:
    """지문 1개 x 모델 1개의 run 레코드와 그 run의 객관식 문항들에 대한 judged scores(반복 포함
    전부)로 지표를 계산한다.

    judged_scores가 비어 있으면(아직 채점 전, 또는 객관식 문항이 없음) judge_mean/judge_min은
    None, perfect는 False — "전부 5점"을 비교할 judged 점수 자체가 없는 상태를 perfect로
    잘못 판정(all()의 vacuous truth)하지 않기 위함."""
    attempt_log = run.get("attempt_log") or []
    first_pass = bool(attempt_log[0].get("validation_passed")) if attempt_log else False
    final_pass = bool(run.get("validation_passed", False))

    items = run.get("items") or []
    num_items = run.get("num_items")
    count_ok = num_items is not None and len(items) == num_items

    if judged_scores:
        overalls = [s["overall"] for s in judged_scores]
        judge_mean = round(sum(overalls) / len(overalls), 3)
        judge_min = min(min(s[c] for c in _CRITERIA) for s in judged_scores)
        all_fives = all(all(s[c] == 5 for c in _CRITERIA) for s in judged_scores)
    else:
        judge_mean = None
        judge_min = None
        all_fives = False

    perfect = first_pass and count_ok and all_fives

    return {
        "first_pass": first_pass,
        "final_pass": final_pass,
        "count_ok": count_ok,
        "judge_mean": judge_mean,
        "judge_min": judge_min,
        "perfect": perfect,
    }


def classify_passage(metrics_by_model: dict) -> dict:
    """지문 1개에 대한 모델별 compute_metrics() 결과로 ceiling/discriminating을 판정한다.

    ceiling = 분석 대상 모델 전부 perfect.
    discriminating = final_pass가 다르거나, count_ok가 다르거나, judge_mean이 둘 다 있는 모델
    쌍 중 차이가 1.0 이상인 쌍이 있음."""
    metrics = list(metrics_by_model.values())
    ceiling = bool(metrics) and all(m["perfect"] for m in metrics)

    final_passes = {m["final_pass"] for m in metrics}
    count_oks = {m["count_ok"] for m in metrics}
    discriminating = len(final_passes) > 1 or len(count_oks) > 1

    if not discriminating:
        means = [m["judge_mean"] for m in metrics if m["judge_mean"] is not None]
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                if abs(means[i] - means[j]) >= 1.0:
                    discriminating = True
                    break
            if discriminating:
                break

    return {"ceiling": ceiling, "discriminating": discriminating}


def coverage_passage_ids(runs_by_model: dict, models: list) -> list:
    """모든 대상 모델에 run 레코드가(성공/실패 무관) 존재하는 지문 id만 비교 대상으로 삼는다."""
    sets = [set(runs_by_model[m].keys()) for m in models]
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def aggregate_ceiling_discrimination(classifications: dict, meta: dict) -> dict:
    """전체·case_type별·format별 ceiling/discriminating 비율(분모 = 그룹 내 지문 수)."""

    def _bucket(key_fn):
        buckets: dict[str, list] = {}
        for pid, cls in classifications.items():
            buckets.setdefault(key_fn(pid), []).append(cls)
        return {
            key: {
                "n": len(items),
                "ceiling_rate": round(sum(c["ceiling"] for c in items) / len(items), 3),
                "discriminating_rate": round(sum(c["discriminating"] for c in items) / len(items), 3),
            }
            for key, items in buckets.items()
        }

    overall = _bucket(lambda _pid: "전체")["전체"] if classifications else {"n": 0, "ceiling_rate": 0.0, "discriminating_rate": 0.0}
    by_case_type = _bucket(lambda pid: meta[pid]["case_type"])
    by_format = _bucket(lambda pid: meta[pid]["format"])
    return {"overall": overall, "by_case_type": by_case_type, "by_format": by_format}


def score_distribution(judged_records_for_model: list) -> dict:
    """모델 1개의 judged 레코드(여러 run_id 합산)에서 기준별 1~5 빈도 + parse_failed 수."""
    dist = {c: {i: 0 for i in range(1, 6)} for c in _CRITERIA}
    parse_failed = 0
    for r in judged_records_for_model:
        s = r["scores"]
        if s.get("parse_failed"):
            parse_failed += 1
        for c in _CRITERIA:
            v = s.get(c)
            if v in dist[c]:
                dist[c][v] += 1
    return {"dist": dist, "parse_failed": parse_failed, "n": len(judged_records_for_model)}


def _print_rate_row(label: str, stats: dict) -> None:
    print(f"  {label:<14}n={stats['n']:<4}ceiling={stats['ceiling_rate']*100:>5.1f}%  discriminating={stats['discriminating_rate']*100:>5.1f}%")


def cmd_discrimination(args: argparse.Namespace) -> None:
    models = args.models or list(_MODELS)
    meta = load_inputs_meta(_INPUTS_PATH)

    runs_by_model = {}
    judged_grouped_by_model = {}
    judged_records_by_model = {}
    for model in models:
        records = load_run_records(_run_path(model))
        runs_by_model[model] = {r["id"]: r for r in records}
        judged_records = load_run_records(_judged_path(args.judge_model, model))
        judged_records_by_model[model] = judged_records
        judged_grouped_by_model[model] = group_judged_by_run(judged_records)

    passage_ids = coverage_passage_ids(runs_by_model, models)
    n_skipped_coverage = len(meta) - len(passage_ids)

    classifications = {}
    for pid in passage_ids:
        metrics_by_model = {
            model: compute_metrics(runs_by_model[model][pid], judged_grouped_by_model[model].get(pid, []))
            for model in models
        }
        classifications[pid] = classify_passage(metrics_by_model)

    agg = aggregate_ceiling_discrimination(classifications, meta)

    print(
        f"분석 대상 지문 {len(passage_ids)}개 / 전체 입력 {len(meta)}개 "
        f"(모델 {len(models)}종: {', '.join(models)}, Judge: {args.judge_model})"
    )
    if n_skipped_coverage:
        print(f"  커버리지 부족(모든 모델에 run이 있지 않음)으로 제외 {n_skipped_coverage}건")

    print("\n[ceiling / discriminating 비율]")
    _print_rate_row("전체", agg["overall"])
    print("  -- case_type별 --")
    for key in sorted(agg["by_case_type"]):
        _print_rate_row(key, agg["by_case_type"][key])
    print("  -- format별 --")
    for key in sorted(agg["by_format"]):
        _print_rate_row(key, agg["by_format"][key])

    print("\n[모델별 기준 점수 분포(1~5) + parse_failed]")
    for model in models:
        dist_result = score_distribution(judged_records_by_model[model])
        print(f"  {model} (n={dist_result['n']}, parse_failed={dist_result['parse_failed']})")
        for c in _CRITERIA:
            freq = dist_result["dist"][c]
            freq_str = " ".join(f"{i}:{freq[i]}" for i in range(1, 6))
            print(f"    {c:<10}{freq_str}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_judge = sub.add_parser("judge", help="run의 객관식 문항을 Judge로 채점해 캐시에 append(이어서 실행)")
    p_judge.add_argument("--judge-model", default="openai/gpt-5.6-luna")
    p_judge.add_argument("--models", nargs="+", choices=_MODELS, default=None)
    p_judge.add_argument("--repeat", type=int, default=1)
    p_judge.set_defaults(func=cmd_judge)

    p_disc = sub.add_parser("discrimination", help="judged 캐시 + run + 입력 메타로 모델 간 변별력 분석")
    p_disc.add_argument("--judge-model", default="openai/gpt-5.6-luna")
    p_disc.add_argument("--models", nargs="+", choices=_MODELS, default=None)
    p_disc.set_defaults(func=cmd_discrimination)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
