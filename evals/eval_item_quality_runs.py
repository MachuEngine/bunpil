#!/usr/bin/env python
"""문항 품질 run 파일(golden_gen/gen_item_quality_golden.py `generate`가 만든
`data/golden/_item_quality_runs/{model}.jsonl`, smoke 제외)을 Judge로 채점하고
모델 간 변별력(discrimination)을 분석한다.

서브커맨드:
  judge           run의 객관식 문항을 get_judge_backend()로 채점 →
                  data/golden/_item_quality_judged/<judge-slug>/<model>.jsonl에 append(이어서 실행)
                  --only-labeled: item_quality_golden.json 라벨셋에 포함된 (run_id, item_id)만
                  채점 대상으로 좁힌다 — 라벨셋은 --repeat 3, 나머지 생성 문항은 --repeat 1처럼
                  나눠 돌릴 수 있다(캐시 키에 repeat가 있어 그대로 이어서 실행됨).
  discrimination  judged 캐시 + run + item_quality_inputs.json(case_type·format)을 모아
                  지문×모델별 지표를 내고, 모델 간 변별력(ceiling/discriminating)을 집계
  reliability     item_quality_golden.json 사람 라벨과 judged 캐시(여러 Judge 후보)를 대조해
                  기준별 가중 κ·MAE·편향·짝지은 비교·생성 모델별 편향을 계산하고
                  data/golden/_judge_selection_round2.json에 저장(재분석이 API 재호출 없이 가능)
  compare-generators  생성 모델 3종(qwen2.5-14b 기준/gpt-6-luna/gemini-3.8-flash)을
                  규칙 지표(run)·Judge 지표(judged 캐시)·사람 지표(라벨셋)·짝지은 비교로
                  종합 비교하고 data/golden/_generator_comparison.json에 저장

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
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# golden_gen/gen_item_quality_golden.py는 수정하지 않고 공개 헬퍼(load_run_records,
# dedupe_run_records, 언더스코어 없음)만 가져다 쓴다 — jsonl 한 줄 = 한 dict인 범용
# 로더라 judged 캐시 파일을 읽는 데도 그대로 쓸 수 있다(이름은 run 전용처럼 보이지만
# 구현은 범용).
from golden_gen.gen_item_quality_golden import dedupe_run_records, load_run_records
from evals.stats import cluster_bootstrap_ci, mean_ci, weighted_kappa

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "golden")
_INPUTS_PATH = os.path.join(_GOLDEN_DIR, "item_quality_inputs.json")
_RUNS_DIR = os.path.join(_GOLDEN_DIR, "_item_quality_runs")
_JUDGED_DIR = os.path.join(_GOLDEN_DIR, "_item_quality_judged")
# reliability·judge --only-labeled가 함께 쓰는 라벨셋 경로 — golden_gen/gen_item_quality_golden.py
# build-labelset이 만든 산출물(해당 스크립트는 수정하지 않음, 경로만 공유).
_ITEM_QUALITY_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "item_quality_golden.json")
_MODEL_MAP_PATH = os.path.join(_GOLDEN_DIR, "_item_quality_model_map.json")
_SCORE_LABELS = (1, 2, 3, 4, 5)

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


def _code_version() -> str:
    """golden_gen/gen_item_quality_golden.py `_code_version()`과 같은 값을 내는 로직을
    여기도 둔다(그 함수는 언더스코어 프라이빗이라 모듈 경계를 넘어 직접 가져오지 않음)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── judge: 채점 대상 선정 + 캐시 스킵 ────────────────────────────────────

def mc_items_for_run(run: dict, labeled_keys: set[tuple] | None = None) -> list[dict]:
    """채점 대상 문항 선정 — error가 있는 run은 전부 건너뛰고(스펙), 나머지는 객관식만
    (JUDGE_TPL이 객관식 기준이라 서술형은 채점 대상이 아님).

    labeled_keys가 주어지면(judge --only-labeled) (run["id"], item_id)가 그 집합에 있는
    문항만 추가로 걸러낸다. None이면(기본 동작) 걸러내지 않는다."""
    if run.get("error"):
        return []
    items = [it for it in run.get("items", []) if it.get("item_type") == "객관식"]
    if labeled_keys is not None:
        items = [it for it in items if (run["id"], it["item_id"]) in labeled_keys]
    return items


def load_labeled_keys(golden_path: str, model_map_path: str) -> dict[str, set[tuple]]:
    """judge --only-labeled가 채점 범위를 좁히는 데 쓰는, 생성 모델별 (passage_id, run_item_id)
    집합 — item_quality_golden.json에 블라인드 id로 들어간 문항만 대상이다(사람 라벨이
    아직 채워졌는지는 가리지 않는다 — 라벨링 전에 미리 채점해두는 용도이기 때문)."""
    with open(golden_path, encoding="utf-8") as f:
        golden_ids = {e["id"] for e in json.load(f).get("entries", [])}
    with open(model_map_path, encoding="utf-8") as f:
        model_map = json.load(f).get("map", {})
    keys_by_model: dict[str, set[tuple]] = {}
    for blind_id, mapping in model_map.items():
        if blind_id not in golden_ids:
            continue
        keys_by_model.setdefault(mapping["model"], set()).add(
            (mapping["passage_id"], mapping["run_item_id"])
        )
    return keys_by_model


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

    labeled_keys_by_model = (
        load_labeled_keys(_ITEM_QUALITY_GOLDEN_PATH, _MODEL_MAP_PATH) if args.only_labeled else None
    )
    scope = "라벨셋만" if args.only_labeled else "전체"
    print(
        f"=== Judge: {args.judge_model} / 대상 모델: {', '.join(models)} / "
        f"repeat={args.repeat} / 범위={scope} ==="
    )

    for model in models:
        runs = load_run_records(_run_path(model))
        out_path = _judged_path(args.judge_model, model)
        done = judged_keys(load_run_records(out_path))
        labeled_keys = labeled_keys_by_model.get(model, set()) if labeled_keys_by_model is not None else None

        n_judged = 0
        n_skipped = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for run in runs:
                for item in mc_items_for_run(run, labeled_keys):
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


# ── reliability: 사람 라벨 vs Judge 후보 신뢰도(κ·MAE·편향) ───────────────
#
# item_quality_golden.json(블라인드 라벨셋) + _item_quality_model_map.json(블라인드 id →
# 실제 모델/원본) + judged 캐시(여러 Judge 후보)를 조인해 Judge 선정 지표를 계산한다.
# 핵심 계산은 전부 순수 함수(LLM 호출 없음) — 실제 Judge 호출은 `judge` 서브커맨드가
# 캐시에 미리 쌓아두고, 이 서브커맨드는 그 캐시만 읽는다.

def _round3(x: float | None) -> float | None:
    return round(x, 3) if x is not None else None


def usable_human_labels(golden_entries: list[dict]) -> tuple[dict[str, dict], int]:
    """cannot_judge=true이거나 세 기준 중 하나라도 null인 entry는 제외한다.
    반환: (블라인드 id -> human_label dict, 제외 수)."""
    usable: dict[str, dict] = {}
    excluded = 0
    for e in golden_entries:
        hl = e.get("human_label") or {}
        if hl.get("cannot_judge") or any(hl.get(c) is None for c in _CRITERIA):
            excluded += 1
            continue
        usable[e["id"]] = hl
    return usable, excluded


def repeats_for_item(judged_records: list[dict], run_id: str, item_id: str) -> list[dict]:
    """judged 캐시(모델 1개분)에서 (run_id, item_id)에 해당하는 반복들을 repeat 순서로
    정렬해 scores만 뽑는다. 매칭 없으면 빈 리스트(그 Judge 캐시가 아직 없다는 뜻)."""
    matches = [r for r in judged_records if r["run_id"] == run_id and r["item_id"] == item_id]
    matches.sort(key=lambda r: r["repeat"])
    return [r["scores"] for r in matches]


def build_rows(
    golden_entries: list[dict],
    model_map: dict[str, dict],
    judged_by_judge: dict[str, dict[str, list[dict]]],
) -> tuple[list[dict], int]:
    """블라인드 id → model_map → (run_id, item_id)로 조인해 문항별 행을 만든다.

    judged_by_judge: {judge_model: {생성모델: judged 캐시 레코드 리스트}}.
    반환: (rows, 사람 라벨 제외 수). rows[i] = {id, passage_id, model,
    human: {기준: 점수}, judges: {judge_model: [반복0 scores, 반복1 scores, ...]}}
    (judges에는 캐시가 있는 judge만 들어간다 — 없는 judge는 키 자체가 없음).
    문항 원문은 담지 않는다(하드룰 4)."""
    usable, excluded = usable_human_labels(golden_entries)
    rows = []
    for entry in golden_entries:
        blind_id = entry["id"]
        if blind_id not in usable:
            continue
        mapping = model_map.get(blind_id)
        if mapping is None:
            continue
        model, passage_id, run_item_id = mapping["model"], mapping["passage_id"], mapping["run_item_id"]
        judges: dict[str, list[dict]] = {}
        for judge_model, by_model in judged_by_judge.items():
            scores_list = repeats_for_item(by_model.get(model, []), passage_id, run_item_id)
            if scores_list:
                judges[judge_model] = scores_list
        rows.append({
            "id": blind_id,
            "passage_id": passage_id,
            "model": model,
            "human": {c: usable[blind_id][c] for c in _CRITERIA},
            "judges": judges,
        })
    return rows, excluded


def judge_avg_score(scores_list: list[dict], criterion: str) -> float:
    """반복 평균(실수) — MAE/편향처럼 정밀도가 필요한 지표용."""
    return sum(s[criterion] for s in scores_list) / len(scores_list)


def judge_avg_rounded(scores_list: list[dict], criterion: str) -> int:
    """반복 평균을 정수로 반올림 — 가중 κ는 순서형 라벨([1,2,3,4,5]) 비교라 정수가 필요."""
    return round(judge_avg_score(scores_list, criterion))


def per_repeat_kappas(rows_with_judge: list[dict], judge_model: str, criterion: str) -> list[float | None]:
    """반복 평균으로 뭉개지 않고, 반복 인덱스별로 따로 가중 κ를 계산한다(판정 안정성 확인용).
    해당 인덱스의 반복이 있는 행만 사용 — 표본이 2건 미만이면 그 인덱스는 None."""
    max_repeats = max((len(r["judges"][judge_model]) for r in rows_with_judge), default=0)
    kappas: list[float | None] = []
    for k in range(max_repeats):
        pairs = [
            (r["human"][criterion], r["judges"][judge_model][k][criterion])
            for r in rows_with_judge
            if len(r["judges"][judge_model]) > k
        ]
        if len(pairs) < 2:
            kappas.append(None)
            continue
        human_vals = [p[0] for p in pairs]
        judge_vals = [p[1] for p in pairs]
        kappas.append(weighted_kappa(human_vals, judge_vals, labels=_SCORE_LABELS))
    return kappas


def paired_abs_error_diff(rows: list[dict], judge_a: str, judge_b: str, criterion: str) -> list[tuple[str, float]]:
    """두 Judge 모두 캐시가 있는 문항에서 |judge-human| 오차 차이(A−B)를 지문 id와 함께
    반환한다. 양수면 B가 그 문항에서 더 정확(오차가 더 작음)."""
    pairs = []
    for r in rows:
        if judge_a in r["judges"] and judge_b in r["judges"]:
            human_val = r["human"][criterion]
            err_a = abs(judge_avg_score(r["judges"][judge_a], criterion) - human_val)
            err_b = abs(judge_avg_score(r["judges"][judge_b], criterion) - human_val)
            pairs.append((r["passage_id"], err_a - err_b))
    return pairs


def bias_by_generation_model(rows: list[dict], judge_model: str, criterion: str) -> dict[str, dict]:
    """생성 모델별 (judge-human) 평균 편향 + 지문 단위 클러스터 부트스트랩 CI — 같은 계열
    (예: OpenAI 생성물 vs gpt Judge)을 후하게 주는지 확인하기 위함."""
    result = {}
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model and judge_model in r["judges"]]
        if not subset:
            result[model] = {"n": 0, "bias": None, "ci_low": None, "ci_high": None}
            continue
        diffs = [judge_avg_score(r["judges"][judge_model], criterion) - r["human"][criterion] for r in subset]
        clusters = [r["passage_id"] for r in subset]
        point, lo, hi = mean_ci(diffs, clusters)
        result[model] = {
            "n": len(subset), "bias": _round3(point), "ci_low": _round3(lo), "ci_high": _round3(hi),
        }
    return result


def human_label_distribution(golden_entries: list[dict], model_map: dict[str, dict]) -> dict:
    """사람 라벨 분포 — 기준별 1~5 빈도 + 최빈값 비율(κ는 점수가 한쪽에 몰리면 불안정하므로
    같이 본다), 학생난이도 상/중/하 빈도(전체 + 생성 모델별), cannot_judge 수.
    cannot_judge 항목은 점수·난이도 분포 집계에서 제외(판단 자체를 보류한 항목이라 제외)."""
    cannot_judge_n = sum(1 for e in golden_entries if (e.get("human_label") or {}).get("cannot_judge"))

    score_dist = {c: {i: 0 for i in range(1, 6)} for c in _CRITERIA}
    difficulty_overall = {"상": 0, "중": 0, "하": 0}
    difficulty_by_model: dict[str, dict] = {}

    for e in golden_entries:
        hl = e.get("human_label") or {}
        if hl.get("cannot_judge"):
            continue
        for c in _CRITERIA:
            v = hl.get(c)
            if v in score_dist[c]:
                score_dist[c][v] += 1
        diff = hl.get("학생난이도")
        if diff in difficulty_overall:
            difficulty_overall[diff] += 1
            model = (model_map.get(e["id"]) or {}).get("model")
            if model:
                difficulty_by_model.setdefault(model, {"상": 0, "중": 0, "하": 0})
                difficulty_by_model[model][diff] += 1

    mode_ratio = {}
    for c in _CRITERIA:
        total = sum(score_dist[c].values())
        mode_ratio[c] = round(max(score_dist[c].values()) / total, 3) if total else None

    return {
        "n_entries": len(golden_entries),
        "cannot_judge_n": cannot_judge_n,
        "score_dist": score_dist,
        "mode_ratio": mode_ratio,
        "difficulty_overall": difficulty_overall,
        "difficulty_by_model": difficulty_by_model,
    }


def compute_reliability_report(
    golden_entries: list[dict],
    model_map: dict[str, dict],
    judged_by_judge: dict[str, dict[str, list[dict]]],
    judge_models: list[str],
) -> dict:
    """reliability의 핵심 계산 전체(LLM 호출 없음) — cmd_reliability와 테스트가 공유."""
    rows, excluded = build_rows(golden_entries, model_map, judged_by_judge)

    criteria_report = {}
    for c in _CRITERIA:
        per_judge = {}
        for jm in judge_models:
            rows_with_judge = [r for r in rows if jm in r["judges"]]
            n = len(rows_with_judge)
            if n < 2:
                per_judge[jm] = {"n": n, "note": "표본 부족(2건 미만)"}
                continue

            human_vals = [r["human"][c] for r in rows_with_judge]
            judge_vals_rounded = [judge_avg_rounded(r["judges"][jm], c) for r in rows_with_judge]
            clusters = [r["passage_id"] for r in rows_with_judge]

            kappa_point, kappa_lo, kappa_hi = cluster_bootstrap_ci(
                list(zip(human_vals, judge_vals_rounded)), clusters,
                lambda sample: weighted_kappa(
                    [a for a, b in sample], [b for a, b in sample], labels=_SCORE_LABELS,
                ),
            )
            mae_values = [abs(judge_avg_score(r["judges"][jm], c) - r["human"][c]) for r in rows_with_judge]
            mae_point, mae_lo, mae_hi = mean_ci(mae_values, clusters)
            bias_values = [judge_avg_score(r["judges"][jm], c) - r["human"][c] for r in rows_with_judge]
            bias_point, bias_lo, bias_hi = mean_ci(bias_values, clusters)

            repeat_kappas = [k for k in per_repeat_kappas(rows_with_judge, jm, c) if k is not None]

            per_judge[jm] = {
                "n": n,
                "weighted_kappa": _round3(kappa_point), "kappa_ci": [_round3(kappa_lo), _round3(kappa_hi)],
                "mae": _round3(mae_point), "mae_ci": [_round3(mae_lo), _round3(mae_hi)],
                "bias": _round3(bias_point), "bias_ci": [_round3(bias_lo), _round3(bias_hi)],
                "repeat_kappa_mean": _round3(sum(repeat_kappas) / len(repeat_kappas)) if repeat_kappas else None,
                "repeat_kappa_range": (
                    [_round3(min(repeat_kappas)), _round3(max(repeat_kappas))] if repeat_kappas else None
                ),
            }

        pairwise = {}
        for i in range(len(judge_models)):
            for j in range(i + 1, len(judge_models)):
                judge_a, judge_b = judge_models[i], judge_models[j]
                diff_pairs = paired_abs_error_diff(rows, judge_a, judge_b, c)
                label = f"{judge_a} vs {judge_b}"
                if len(diff_pairs) < 2:
                    pairwise[label] = {"n": len(diff_pairs), "note": "표본 부족(2건 미만)"}
                    continue
                diffs = [d for _, d in diff_pairs]
                diff_clusters = [pid for pid, _ in diff_pairs]
                point, lo, hi = mean_ci(diffs, diff_clusters)
                inconclusive = lo is not None and hi is not None and lo <= 0 <= hi
                if inconclusive:
                    verdict = "판정 불가(CI가 0을 포함)"
                elif point is not None and point > 0:
                    verdict = f"{judge_b}가 더 정확"
                else:
                    verdict = f"{judge_a}가 더 정확"
                pairwise[label] = {
                    "n": len(diff_pairs), "mean_abs_error_diff": _round3(point),
                    "ci": [_round3(lo), _round3(hi)], "verdict": verdict,
                }

        bias_by_model = {jm: bias_by_generation_model(rows, jm, c) for jm in judge_models}

        criteria_report[c] = {
            "per_judge": per_judge, "pairwise": pairwise, "bias_by_generation_model": bias_by_model,
        }

    return {
        "n_usable": len(rows),
        "n_excluded": excluded,
        "criteria": criteria_report,
        "human_label_distribution": human_label_distribution(golden_entries, model_map),
        # 문항별 행 — 블라인드 id·passage_id·model·기준별 human·judge별 반복 점수만(원문 없음).
        # 재분석이 API 재호출 없이 가능해야 한다는 요구에 맞춰 저장한다.
        "rows": rows,
    }


def _print_reliability_report(report: dict, judge_models: list[str]) -> None:
    """콘솔 출력 — 문항 원문은 절대 출력하지 않는다(하드룰 4). 지표 이름 옆에 짧은 설명을 붙인다."""
    print(f"사람 라벨 {report['n_usable']}건 사용 / 제외 {report['n_excluded']}건(cannot_judge 또는 미채점)")

    dist = report["human_label_distribution"]
    print(f"\n[사람 라벨 분포] cannot_judge={dist['cannot_judge_n']}건")
    for c in _CRITERIA:
        freq = dist["score_dist"][c]
        freq_str = " ".join(f"{i}:{freq[i]}" for i in range(1, 6))
        print(f"  {c:<10}{freq_str}  (최빈값 비율={dist['mode_ratio'][c]})")
    diff = dist["difficulty_overall"]
    print(f"  학생난이도(전체) 상:{diff['상']} 중:{diff['중']} 하:{diff['하']}")
    for model in sorted(dist["difficulty_by_model"]):
        d = dist["difficulty_by_model"][model]
        print(f"    {model:<16}상:{d['상']} 중:{d['중']} 하:{d['하']}")

    for c in _CRITERIA:
        print(f"\n[{c}]")
        cr = report["criteria"][c]
        for jm in judge_models:
            pj = cr["per_judge"].get(jm, {})
            if pj.get("note"):
                print(f"  {jm}: {pj['note']} (n={pj.get('n', 0)})")
                continue
            print(f"  {jm} (n={pj['n']})")
            print(
                f"    weighted κ(가중 카파, 사람과의 순서형 일치도, 1에 가까울수록 좋음)="
                f"{pj['weighted_kappa']} CI={pj['kappa_ci']}"
            )
            print(f"    repeat별 κ 평균/범위(반복마다 따로 계산한 κ의 안정성)={pj['repeat_kappa_mean']} / {pj['repeat_kappa_range']}")
            print(f"    MAE(평균 절대 오차, 낮을수록 좋음)={pj['mae']} CI={pj['mae_ci']}")
            print(f"    편향(judge-human 평균, 양수면 Judge가 사람보다 후함)={pj['bias']} CI={pj['bias_ci']}")

        if cr["pairwise"]:
            print("  [짝지은 비교: |judge-human| 오차 차이(A-B), CI가 0을 포함하면 판정 불가]")
            for label, pw in cr["pairwise"].items():
                if pw.get("note"):
                    print(f"    {label}: {pw['note']} (n={pw['n']})")
                else:
                    print(f"    {label}: mean_diff={pw['mean_abs_error_diff']} CI={pw['ci']} → {pw['verdict']}")

        print("  [생성 모델별 편향(judge-human 평균) — 같은 계열을 후하게 주는지 확인용]")
        for jm in judge_models:
            print(f"    {jm}:")
            for model, b in cr["bias_by_generation_model"][jm].items():
                if b["n"] == 0:
                    print(f"      {model}: 표본 없음")
                else:
                    print(f"      {model}: n={b['n']} bias={b['bias']} CI=[{b['ci_low']}, {b['ci_high']}]")


def cmd_reliability(args: argparse.Namespace) -> None:
    with open(_ITEM_QUALITY_GOLDEN_PATH, encoding="utf-8") as f:
        golden_entries = json.load(f)["entries"]
    with open(_MODEL_MAP_PATH, encoding="utf-8") as f:
        model_map = json.load(f)["map"]

    gen_models = sorted({m["model"] for m in model_map.values()})
    judged_by_judge = {
        jm: {model: load_run_records(_judged_path(jm, model)) for model in gen_models}
        for jm in args.judges
    }

    report = compute_reliability_report(golden_entries, model_map, judged_by_judge, args.judges)

    out_path = os.path.join(_GOLDEN_DIR, "_judge_selection_round2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_reliability_report(report, args.judges)
    print(f"\n문항별 상세는 {out_path}에 저장했습니다(재분석은 이 파일만으로 가능, API 재호출 불필요).")


# ── compare-generators: 생성 모델 3종 종합 비교 ───────────────────────────
#
# 규칙 지표(run 파일) + Judge 지표(judged 캐시, 모델의 모든 객관식 문항) + 사람 지표
# (item_quality_golden.json 라벨셋, 모델당 ≤33건) + 짝지은 비교(지문 단위)를 모은다.
# 핵심 계산은 전부 순수 함수(LLM 호출 없음) — cmd_compare_generators와 테스트가 공유.

def compute_rule_metrics(records: list[dict]) -> dict:
    """모델 1개의 run 레코드(지문당 1개, dedupe_run_records() 적용 후)로 지문 단위 규칙
    지표를 계산한다.

    분모: 목표 개수 달성률·첫 시도 통과율·최종 게이트 통과율은 전체 지문 수(len(records))
    — run 실패나 객관식 0개 지문도 실패로 포함한다(품질 지표에서는 빠지는 지문이 여기서는
    실패로 잡혀야 모델 간 비교가 공정하다). 평균 시도 횟수·malformed 재요청 합은 run이
    실패하지 않은 케이스 기준(실패 run은 시도 자체가 집계 대상 밖). 지연(초 평균·중앙값)과
    객관식 0개 지문 수는 전체 지문 기준. 비용 추정 평균은 토큰 사용량이 기록된(성공) 케이스
    평균이다.

    최종 게이트 통과율(final_gate_pass_rate)은 런타임 Judge(gpt-5.6-luna, generate 시
    JUDGE_BACKEND로 고정)의 구조 검증 결과(validation_passed)에 의존한다 — 이 스크립트의
    --judge-model(문항 품질 평가용 Judge, 기본 claude-sonnet-5.5)과는 다른 모델이다."""
    n = len(records)
    ok = [r for r in records if not r.get("error")]

    count_ok = sum(
        1 for r in records
        if not r.get("error") and r.get("num_items") is not None and len(r.get("items", [])) == r["num_items"]
    )
    first_pass = sum(
        1 for r in records
        if not r.get("error") and (r.get("attempt_log") or []) and r["attempt_log"][0].get("validation_passed")
    )
    final_pass = sum(1 for r in records if not r.get("error") and r.get("validation_passed"))

    avg_attempts = (sum(r.get("attempts", 0) for r in ok) / len(ok)) if ok else 0.0
    malformed_total = sum(
        sum(a.get("malformed_retries", 0) for a in r.get("attempt_log", [])) for r in ok
    )

    wall_clocks = [r.get("wall_clock_sec", 0.0) for r in records]
    avg_wall_clock = (sum(wall_clocks) / n) if n else 0.0
    median_wall_clock = statistics.median(wall_clocks) if wall_clocks else 0.0

    costs = [r["cost_usd_est"] for r in ok if r.get("cost_usd_est") is not None]
    avg_cost = (sum(costs) / len(costs)) if costs else None

    zero_mc = sum(1 for r in records if len(mc_items_for_run(r)) == 0)

    return {
        "n": n,
        "count_ok_rate": round(count_ok / n, 3) if n else 0.0,
        "first_pass_rate": round(first_pass / n, 3) if n else 0.0,
        "final_gate_pass_rate": round(final_pass / n, 3) if n else 0.0,
        "avg_attempts": round(avg_attempts, 3),
        "malformed_retries_total": malformed_total,
        "avg_wall_clock_sec": round(avg_wall_clock, 3),
        "median_wall_clock_sec": round(median_wall_clock, 3),
        "avg_cost_usd": round(avg_cost, 5) if avg_cost is not None else None,
        "n_zero_mc_passages": zero_mc,
    }


def quality_stats_by_criteria(scored_rows: list[tuple[str, dict]]) -> dict:
    """(클러스터 id, {기준: 점수}) 튜플 리스트로 기준별 평균·저품질 비율(≤2점)을 클러스터
    부트스트랩 CI와 함께 계산한다. judge_metrics_for_model·human_metrics_for_model이 공유
    — Judge는 클러스터가 지문(run_id), 사람 라벨은 클러스터가 원본 지문(passage_id)이라는
    차이만 있고 계산 자체는 같다."""
    result = {}
    for c in _CRITERIA:
        clusters = [cid for cid, _ in scored_rows]
        values = [scores[c] for _, scores in scored_rows]
        mean_point, mean_lo, mean_hi = mean_ci(values, clusters)
        low_flags = [1 if v <= 2 else 0 for v in values]
        low_point, low_lo, low_hi = mean_ci(low_flags, clusters)
        result[c] = {
            "n": len(values),
            "mean": _round3(mean_point), "mean_ci": [_round3(mean_lo), _round3(mean_hi)],
            "low_quality_rate": _round3(low_point), "low_quality_ci": [_round3(low_lo), _round3(low_hi)],
        }
    # 정답유일성의 저품질 비율을 "치명적 실패율"로 별도 노출(정답유일성이 셋 중 가장
    # 치명적인 기준이라 설계에서 따로 요구) — 값 자체는 위 정답유일성.low_quality_rate와 같다.
    result["uniqueness_critical_fail_rate"] = result["정답유일성"]["low_quality_rate"]
    result["uniqueness_critical_fail_ci"] = result["정답유일성"]["low_quality_ci"]
    return result


def judge_metrics_for_model(judged_records: list[dict]) -> dict:
    """모델 1개의 judged 캐시(모든 객관식 문항, 반복 포함)로 기준별 평균·저품질 비율을
    지문(run_id) 단위 클러스터 CI와 함께 계산한다."""
    return quality_stats_by_criteria([(r["run_id"], r["scores"]) for r in judged_records])


def human_rows_for_model(golden_entries: list[dict], model_map: dict[str, dict], model: str) -> list[dict]:
    """usable_human_labels() + model_map으로 특정 생성 모델에 해당하는 (passage_id, human
    점수) 행만 추린다(cannot_judge·점수 null 제외, 모델당 최대 33건)."""
    usable, _ = usable_human_labels(golden_entries)
    rows = []
    for blind_id, hl in usable.items():
        mapping = model_map.get(blind_id)
        if mapping is None or mapping["model"] != model:
            continue
        rows.append({"passage_id": mapping["passage_id"], "human": hl})
    return rows


def human_metrics_for_model(rows_for_model: list[dict]) -> dict:
    """특정 모델의 사람 라벨 행(usable, 최대 ≤33)으로 기준별 평균·저품질 비율을 계산한다.
    표본이 작아 참고용이다(human_rows_for_model 문서 참고)."""
    return quality_stats_by_criteria([(r["passage_id"], r["human"]) for r in rows_for_model])


def passage_level_judge_overall(judged_records: list[dict]) -> dict[str, float]:
    """모델 1개의 judged 레코드(여러 지문·반복 섞여 있음)에서 지문(run_id)별 overall
    평균 — 짝지은 비교(파생 (a))의 재료."""
    grouped = group_judged_by_run(judged_records)
    return {pid: sum(s["overall"] for s in scores) / len(scores) for pid, scores in grouped.items()}


def passage_level_final_pass(records: list[dict]) -> dict[str, int]:
    """dedupe된 run 레코드(지문당 1개)에서 지문별 최종 게이트 통과(1/0, error면 0) —
    짝지은 비교 (c)의 재료."""
    return {r["id"]: int(bool(r.get("validation_passed"))) for r in records}


def passage_level_latency(records: list[dict]) -> dict[str, float]:
    """dedupe된 run 레코드에서 지문별 wall_clock_sec — 짝지은 비교 (d)의 재료."""
    return {r["id"]: r.get("wall_clock_sec", 0.0) for r in records}


def human_score_by_passage(rows_for_model: list[dict], criterion: str) -> dict[str, float]:
    """human_rows_for_model() 결과에서 기준 1개의 지문별 점수만 뽑는다 — 짝지은 비교
    (b)의 재료."""
    return {r["passage_id"]: r["human"][criterion] for r in rows_for_model}


def paired_ci_verdict(
    values_a: dict[str, float], values_b: dict[str, float],
    higher_is_better: bool, label_a: str, label_b: str,
) -> dict:
    """passage_id -> 값 dict 두 개를 받아, 두 모델 모두 값이 있는 지문만으로 차이
    (b−a)의 평균과 클러스터(=지문) 부트스트랩 95% CI, 판정을 계산한다. 쌍 수(n)는
    공통 지문 수 — 호출부가 그대로 출력/저장해 분모를 명시한다.

    higher_is_better=True면 값이 클수록 좋다는 뜻(품질 점수·게이트 통과 등) — 그 경우
    diff>0(CI가 0을 포함하지 않을 때)이면 b가 더 우수. False면 값이 작을수록 좋다는 뜻
    (지연 등) — diff>0이면 b가 더 오래 걸려 a가 더 우수."""
    common = sorted(set(values_a) & set(values_b))
    if not common:
        return {"n": 0, "mean_diff": None, "ci": [None, None], "verdict": "표본 없음(공통 지문 없음)"}

    diffs = [values_b[p] - values_a[p] for p in common]
    point, lo, hi = mean_ci(diffs, common)
    if lo is None or hi is None:
        verdict = "판정 불가(표본 부족)"
    elif lo <= 0 <= hi:
        verdict = "판정 불가(CI가 0을 포함)"
    else:
        better = label_b if (point > 0) == higher_is_better else label_a
        verdict = f"{better}가 더 우수"

    return {
        "n": len(common), "mean_diff": _round3(point), "ci": [_round3(lo), _round3(hi)], "verdict": verdict,
    }


def judge_case_type_means(judged_records: list[dict], inputs_meta: dict) -> dict:
    """참고용 — case_type별 기준 평균(item 단위 단순 평균, CI 없음). inputs_meta는
    load_inputs_meta()의 반환값(id -> case_type/format)."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in judged_records:
        ct = (inputs_meta.get(r["run_id"]) or {}).get("case_type")
        if ct is None:
            continue
        bucket = buckets.setdefault(ct, {c: [] for c in _CRITERIA})
        for c in _CRITERIA:
            bucket[c].append(r["scores"][c])
    return {
        ct: {c: (round(sum(vals) / len(vals), 3) if vals else None) for c, vals in crit.items()}
        for ct, crit in buckets.items()
    }


def human_case_type_means(rows_for_model: list[dict], inputs_meta: dict) -> dict:
    """참고용 — case_type별 사람 점수 기준 평균(단순 평균, CI 없음)."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in rows_for_model:
        ct = (inputs_meta.get(r["passage_id"]) or {}).get("case_type")
        if ct is None:
            continue
        bucket = buckets.setdefault(ct, {c: [] for c in _CRITERIA})
        for c in _CRITERIA:
            bucket[c].append(r["human"][c])
    return {
        ct: {c: (round(sum(vals) / len(vals), 3) if vals else None) for c, vals in crit.items()}
        for ct, crit in buckets.items()
    }


_MISSING_BIAS_NOTE = (
    "객관식 0개(run 실패 포함) 지문은 Judge·사람 점수 쌍에서는 빠지지만, 규칙 지표"
    "(목표 개수 달성률·통과율)에는 실패로 포함된다 — 품질 지표만 보면 해당 모델이 "
    "유리하게 보일 수 있다는 점에 주의."
)
_GATE_DEPENDENCY_NOTE = (
    "최종 게이트 통과율(final_gate_pass_rate)은 런타임 Judge(gpt-5.6-luna, 생성 시 고정)의 "
    "구조 검증 결과에 의존한다 — 이 커맨드의 --judge-model(문항 품질 평가용)과는 다른 모델."
)
_HUMAN_METRICS_NOTE = "표본이 작아(모델당 최대 33건) 참고용입니다."


def compute_generator_comparison(
    models: list[str],
    baseline: str,
    judge_model: str,
    records_by_model: dict[str, list[dict]],
    judged_by_model: dict[str, list[dict]],
    golden_entries: list[dict],
    model_map: dict[str, dict],
    inputs_meta: dict,
    code_version: str,
) -> dict:
    """compare-generators의 핵심 계산 전체(LLM 호출 없음) — cmd_compare_generators와
    테스트가 공유. records_by_model은 이미 dedupe_run_records()가 적용된 값이어야 한다."""
    rule_metrics = {m: compute_rule_metrics(records_by_model[m]) for m in models}
    judge_metrics = {m: judge_metrics_for_model(judged_by_model.get(m, [])) for m in models}

    human_rows = {m: human_rows_for_model(golden_entries, model_map, m) for m in models}
    human_metrics = {m: human_metrics_for_model(human_rows[m]) for m in models}

    judge_overall_by_model = {m: passage_level_judge_overall(judged_by_model.get(m, [])) for m in models}
    final_pass_by_model = {m: passage_level_final_pass(records_by_model[m]) for m in models}
    latency_by_model = {m: passage_level_latency(records_by_model[m]) for m in models}

    pairs = [(baseline, m) for m in models if m != baseline]
    extra_pair = ("gpt-6-luna", "gemini-3.8-flash")
    if (
        extra_pair[0] in models and extra_pair[1] in models
        and baseline not in extra_pair
    ):
        pairs.append(extra_pair)

    pairwise = {}
    for a, b in pairs:
        human_diff = {
            c: paired_ci_verdict(
                human_score_by_passage(human_rows[a], c), human_score_by_passage(human_rows[b], c),
                higher_is_better=True, label_a=a, label_b=b,
            )
            for c in _CRITERIA
        }
        pairwise[f"{a}_vs_{b}"] = {
            "judge_overall_diff": paired_ci_verdict(
                judge_overall_by_model[a], judge_overall_by_model[b],
                higher_is_better=True, label_a=a, label_b=b,
            ),
            "human_diff": human_diff,
            "final_gate_diff": paired_ci_verdict(
                final_pass_by_model[a], final_pass_by_model[b],
                higher_is_better=True, label_a=a, label_b=b,
            ),
            "latency_diff": paired_ci_verdict(
                latency_by_model[a], latency_by_model[b],
                higher_is_better=False, label_a=a, label_b=b,
            ),
        }

    case_type_table = {
        "human": {m: human_case_type_means(human_rows[m], inputs_meta) for m in models},
        "judge": {m: judge_case_type_means(judged_by_model.get(m, []), inputs_meta) for m in models},
    }

    return {
        "code_version": code_version,
        "judge_model": judge_model,
        "baseline": baseline,
        "models": models,
        "rule_metrics": rule_metrics,
        "judge_metrics": judge_metrics,
        "human_metrics": human_metrics,
        "human_metrics_note": _HUMAN_METRICS_NOTE,
        "pairwise": pairwise,
        "case_type_table": case_type_table,
        "gate_dependency_note": _GATE_DEPENDENCY_NOTE,
        "missing_bias_note": _MISSING_BIAS_NOTE,
    }


def _print_paired(label: str, pw: dict) -> None:
    print(f"    {label}: n={pw['n']} mean_diff={pw['mean_diff']} CI={pw['ci']} → {pw['verdict']}")


def _print_generator_comparison(report: dict) -> None:
    """콘솔 출력 — 문항 원문은 절대 출력하지 않는다(하드룰 4). 지표 이름 옆에 짧은 설명을 붙인다."""
    print(f"Judge: {report['judge_model']} / 기준 모델(baseline): {report['baseline']}")
    print(f"[주의] {report['gate_dependency_note']}")
    print(f"[주의] {report['missing_bias_note']}")

    print("\n[규칙 지표] (분모는 전체 지문 수 — run 실패·객관식 0개도 실패로 포함)")
    for m in report["models"]:
        rm = report["rule_metrics"][m]
        print(
            f"  {m} (n={rm['n']})\n"
            f"    목표 개수 달성률={rm['count_ok_rate']*100:.1f}%  "
            f"첫 시도 통과율={rm['first_pass_rate']*100:.1f}%  "
            f"최종 게이트 통과율={rm['final_gate_pass_rate']*100:.1f}%\n"
            f"    평균 시도 횟수={rm['avg_attempts']}  malformed 재요청 합={rm['malformed_retries_total']}\n"
            f"    지연(초) 평균/중앙값={rm['avg_wall_clock_sec']}/{rm['median_wall_clock_sec']}  "
            f"세트당 비용 추정 평균(USD)={rm['avg_cost_usd']}\n"
            f"    객관식 0개 지문 수={rm['n_zero_mc_passages']}"
        )

    print(f"\n[Judge 지표(Judge: {report['judge_model']}, 모델의 모든 객관식 문항, 지문 단위 클러스터 CI)]")
    for m in report["models"]:
        jm = report["judge_metrics"][m]
        print(f"  {m}")
        for c in _CRITERIA:
            s = jm[c]
            print(
                f"    {c:<10}n={s['n']:<4}평균={s['mean']} CI={s['mean_ci']}  "
                f"저품질(≤2점) 비율={s['low_quality_rate']} CI={s['low_quality_ci']}"
            )
        print(
            f"    정답유일성 치명적 실패율(≤2점)={jm['uniqueness_critical_fail_rate']} "
            f"CI={jm['uniqueness_critical_fail_ci']}"
        )

    print(f"\n[사람 지표] {report['human_metrics_note']}")
    for m in report["models"]:
        hm = report["human_metrics"][m]
        print(f"  {m}")
        for c in _CRITERIA:
            s = hm[c]
            print(
                f"    {c:<10}n={s['n']:<4}평균={s['mean']} CI={s['mean_ci']}  "
                f"저품질(≤2점) 비율={s['low_quality_rate']} CI={s['low_quality_ci']}"
            )
        print(
            f"    정답유일성 치명적 실패율(≤2점)={hm['uniqueness_critical_fail_rate']} "
            f"CI={hm['uniqueness_critical_fail_ci']}"
        )

    print("\n[짝지은 비교] (같은 지문에 두 모델 모두 값이 있는 경우만, n=쌍 수, CI가 0을 포함하면 판정 불가)")
    for label, pw in report["pairwise"].items():
        print(f"  {label}")
        print("    Judge 지문 평균 overall 차이:")
        _print_paired("overall", pw["judge_overall_diff"])
        print("    사람 점수 기준별 차이:")
        for c in _CRITERIA:
            _print_paired(c, pw["human_diff"][c])
        print("    최종 게이트 통과 차이:")
        _print_paired("final_gate", pw["final_gate_diff"])
        print("    지연(초) 차이(양수면 b가 더 오래 걸림):")
        _print_paired("latency", pw["latency_diff"])

    print("\n[case_type별 평균] (참고용, CI 없음)")
    for kind, label in (("human", "사람"), ("judge", "Judge")):
        print(f"  [{label}]")
        for m in report["models"]:
            print(f"    {m}: {report['case_type_table'][kind].get(m)}")


def cmd_compare_generators(args: argparse.Namespace) -> None:
    models = list(_MODELS)
    baseline = args.baseline
    judge_model = args.judge_model

    records_by_model = {m: dedupe_run_records(load_run_records(_run_path(m))) for m in models}
    judged_by_model = {m: load_run_records(_judged_path(judge_model, m)) for m in models}

    with open(_ITEM_QUALITY_GOLDEN_PATH, encoding="utf-8") as f:
        golden_entries = json.load(f)["entries"]
    with open(_MODEL_MAP_PATH, encoding="utf-8") as f:
        model_map = json.load(f)["map"]
    inputs_meta = load_inputs_meta(_INPUTS_PATH)

    report = compute_generator_comparison(
        models, baseline, judge_model, records_by_model, judged_by_model,
        golden_entries, model_map, inputs_meta, _code_version(),
    )

    out_path = os.path.join(_GOLDEN_DIR, "_generator_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_generator_comparison(report)
    print(f"\n상세는 {out_path}에 저장했습니다.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_judge = sub.add_parser("judge", help="run의 객관식 문항을 Judge로 채점해 캐시에 append(이어서 실행)")
    p_judge.add_argument("--judge-model", default="openai/gpt-5.6-luna")
    p_judge.add_argument("--models", nargs="+", choices=_MODELS, default=None)
    p_judge.add_argument("--repeat", type=int, default=1)
    p_judge.add_argument(
        "--only-labeled", action="store_true",
        help="item_quality_golden.json 라벨셋에 포함된 (run_id, item_id)만 채점",
    )
    p_judge.set_defaults(func=cmd_judge)

    p_disc = sub.add_parser("discrimination", help="judged 캐시 + run + 입력 메타로 모델 간 변별력 분석")
    p_disc.add_argument("--judge-model", default="openai/gpt-5.6-luna")
    p_disc.add_argument("--models", nargs="+", choices=_MODELS, default=None)
    p_disc.set_defaults(func=cmd_discrimination)

    p_rel = sub.add_parser(
        "reliability",
        help="item_quality_golden.json 사람 라벨 vs judged 캐시(여러 Judge)로 κ·MAE·편향 분석",
    )
    p_rel.add_argument(
        "--judges", nargs="+", required=True,
        help="비교할 Judge 후보 모델 id (예: anthropic/claude-sonnet-5.5 openai/gpt-5.6-luna)",
    )
    p_rel.set_defaults(func=cmd_reliability)

    p_cmp = sub.add_parser(
        "compare-generators",
        help="생성 모델 3종을 규칙 지표·Judge 지표·사람 지표·짝지은 비교로 종합 비교",
    )
    p_cmp.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    p_cmp.add_argument("--baseline", default="qwen2.5-14b", choices=_MODELS)
    p_cmp.set_defaults(func=cmd_compare_generators)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
