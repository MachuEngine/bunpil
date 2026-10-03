"""evals/eval_item_quality_runs.py 순수 로직 단위 테스트. LLM 호출 없음.

가짜 run 레코드(golden_gen/gen_item_quality_golden.py `_run_passage()`가 만드는 형태)와
가짜 judged 캐시 레코드(evals/eval_item_quality_runs.py `judge` 서브커맨드가 쓰는 형태)만
사용한다. requirements.txt 의존성만 필요(이 모듈 자체는 app.* 모듈 레벨 import가 없어 가볍다)."""
from evals.eval_item_quality_runs import (
    aggregate_ceiling_discrimination,
    classify_passage,
    compute_metrics,
    coverage_passage_ids,
    group_judged_by_run,
    judged_keys,
    load_inputs_meta,
    mc_items_for_run,
    score_distribution,
)


# ── mc_items_for_run ─────────────────────────────────────────────────────

def _mc(item_id="it1", q="질문"):
    return {"item_id": item_id, "question": q, "item_type": "객관식", "options": ["①", "②"], "answer": "①"}


def _essay(item_id="it2"):
    return {"item_id": item_id, "question": "서술형", "item_type": "서술형", "options": [], "answer": ""}


def test_mc_items_for_run_selects_only_multiple_choice():
    run = {"items": [_mc("it1"), _essay("it2"), _mc("it3")], "error": None}
    result = mc_items_for_run(run)
    assert [it["item_id"] for it in result] == ["it1", "it3"]


def test_mc_items_for_run_skips_run_with_error():
    run = {"items": [_mc("it1")], "error": {"type": "RuntimeError", "message": "boom"}}
    assert mc_items_for_run(run) == []


# ── judged_keys (캐시 스킵) ────────────────────────────────────────────────

def test_judged_keys_builds_skip_set():
    records = [
        {"run_id": "p1", "item_id": "it1", "repeat": 0, "scores": {}},
        {"run_id": "p1", "item_id": "it2", "repeat": 0, "scores": {}},
        {"run_id": "p2", "item_id": "it1", "repeat": 1, "scores": {}},
    ]
    keys = judged_keys(records)
    assert keys == {("p1", "it1", 0), ("p1", "it2", 0), ("p2", "it1", 1)}
    assert ("p1", "it1", 1) not in keys


# ── compute_metrics ──────────────────────────────────────────────────────

def _score(정답유일성=5, 오답매력도=5, 근거성=5, parse_failed=False):
    overall = round((정답유일성 + 오답매력도 + 근거성) / 3, 2)
    return {
        "정답유일성": 정답유일성, "오답매력도": 오답매력도, "근거성": 근거성,
        "overall": overall, "parse_failed": parse_failed,
    }


def _run(num_items=2, n_items=2, first_pass=True, final_pass=True, error=None):
    return {
        "num_items": num_items,
        "items": [_mc(f"it{i}") for i in range(n_items)],
        "validation_passed": final_pass,
        "attempt_log": [{"validation_passed": first_pass}],
        "error": error,
    }


def test_compute_metrics_perfect_when_first_pass_count_ok_and_all_fives():
    run = _run()
    scores = [_score(5, 5, 5), _score(5, 5, 5)]
    m = compute_metrics(run, scores)
    assert m["first_pass"] is True
    assert m["final_pass"] is True
    assert m["count_ok"] is True
    assert m["judge_mean"] == 5.0
    assert m["judge_min"] == 5
    assert m["perfect"] is True


def test_compute_metrics_not_perfect_when_any_score_below_five():
    run = _run()
    scores = [_score(5, 5, 5), _score(4, 5, 5)]
    m = compute_metrics(run, scores)
    assert m["judge_min"] == 4
    assert m["perfect"] is False


def test_compute_metrics_no_judged_scores_is_not_perfect_and_means_are_none():
    run = _run()
    m = compute_metrics(run, [])
    assert m["judge_mean"] is None
    assert m["judge_min"] is None
    assert m["perfect"] is False  # vacuous all()을 perfect로 잘못 판정하지 않음


def test_compute_metrics_count_mismatch_breaks_perfect_and_count_ok():
    run = _run(num_items=3, n_items=2)  # 목표 3개인데 실제 2개
    scores = [_score(5, 5, 5), _score(5, 5, 5)]
    m = compute_metrics(run, scores)
    assert m["count_ok"] is False
    assert m["perfect"] is False


def test_compute_metrics_first_pass_false_when_attempt_log_empty():
    run = {"num_items": 1, "items": [_mc()], "validation_passed": False, "attempt_log": [], "error": None}
    m = compute_metrics(run, [])
    assert m["first_pass"] is False
    assert m["final_pass"] is False


# ── classify_passage ──────────────────────────────────────────────────────

def _metrics(first_pass=True, final_pass=True, count_ok=True, judge_mean=5.0, judge_min=5, perfect=True):
    return {
        "first_pass": first_pass, "final_pass": final_pass, "count_ok": count_ok,
        "judge_mean": judge_mean, "judge_min": judge_min, "perfect": perfect,
    }


def test_classify_passage_ceiling_when_all_models_perfect():
    metrics_by_model = {"m1": _metrics(), "m2": _metrics()}
    cls = classify_passage(metrics_by_model)
    assert cls["ceiling"] is True
    assert cls["discriminating"] is False


def test_classify_passage_not_ceiling_when_one_model_not_perfect():
    metrics_by_model = {"m1": _metrics(), "m2": _metrics(perfect=False)}
    cls = classify_passage(metrics_by_model)
    assert cls["ceiling"] is False


def test_classify_passage_discriminating_when_final_pass_differs():
    metrics_by_model = {
        "m1": _metrics(final_pass=True, perfect=True),
        "m2": _metrics(final_pass=False, perfect=False),
    }
    cls = classify_passage(metrics_by_model)
    assert cls["discriminating"] is True


def test_classify_passage_discriminating_when_count_ok_differs():
    metrics_by_model = {
        "m1": _metrics(count_ok=True, perfect=True),
        "m2": _metrics(count_ok=False, perfect=False),
    }
    cls = classify_passage(metrics_by_model)
    assert cls["discriminating"] is True


def test_classify_passage_discriminating_when_judge_mean_gap_at_least_one():
    metrics_by_model = {
        "m1": _metrics(judge_mean=5.0),
        "m2": _metrics(judge_mean=4.0, perfect=False),
    }
    cls = classify_passage(metrics_by_model)
    assert cls["discriminating"] is True


def test_classify_passage_not_discriminating_when_judge_mean_gap_below_one():
    metrics_by_model = {
        "m1": _metrics(judge_mean=5.0),
        "m2": _metrics(judge_mean=4.5, perfect=False),
    }
    cls = classify_passage(metrics_by_model)
    assert cls["discriminating"] is False


def test_classify_passage_handles_missing_judge_mean_without_crashing():
    metrics_by_model = {
        "m1": _metrics(judge_mean=None, perfect=False),
        "m2": _metrics(judge_mean=None, perfect=False),
    }
    cls = classify_passage(metrics_by_model)
    assert cls["ceiling"] is False
    assert cls["discriminating"] is False


# ── coverage_passage_ids / aggregate_ceiling_discrimination ───────────────

def test_coverage_passage_ids_requires_all_models_present():
    runs_by_model = {
        "m1": {"p1": {}, "p2": {}},
        "m2": {"p1": {}},  # p2 없음 — 커버리지 부족
    }
    assert coverage_passage_ids(runs_by_model, ["m1", "m2"]) == ["p1"]


def test_aggregate_ceiling_discrimination_buckets_by_case_type_and_format():
    meta = {
        "p1": {"case_type": "normal", "format": "mc4"},
        "p2": {"case_type": "normal", "format": "mc5"},
        "p3": {"case_type": "hard", "format": "mc4"},
    }
    classifications = {
        "p1": {"ceiling": True, "discriminating": False},
        "p2": {"ceiling": False, "discriminating": True},
        "p3": {"ceiling": False, "discriminating": True},
    }
    agg = aggregate_ceiling_discrimination(classifications, meta)
    assert agg["overall"]["n"] == 3
    assert agg["overall"]["ceiling_rate"] == round(1 / 3, 3)
    assert agg["overall"]["discriminating_rate"] == round(2 / 3, 3)
    assert agg["by_case_type"]["normal"]["n"] == 2
    assert agg["by_case_type"]["hard"]["n"] == 1
    assert agg["by_format"]["mc4"]["n"] == 2
    assert agg["by_format"]["mc5"]["n"] == 1


# ── score_distribution ─────────────────────────────────────────────────

def test_score_distribution_counts_frequencies_and_parse_failed():
    records = [
        {"scores": _score(5, 4, 3, parse_failed=False)},
        {"scores": _score(5, 4, 3, parse_failed=True)},
        {"scores": _score(1, 1, 1, parse_failed=False)},
    ]
    result = score_distribution(records)
    assert result["n"] == 3
    assert result["parse_failed"] == 1
    assert result["dist"]["정답유일성"][5] == 2
    assert result["dist"]["정답유일성"][1] == 1
    assert result["dist"]["오답매력도"][4] == 2
    assert result["dist"]["근거성"][3] == 2


# ── group_judged_by_run / load_inputs_meta ────────────────────────────────

def test_group_judged_by_run_groups_scores_by_run_id():
    records = [
        {"run_id": "p1", "item_id": "it1", "repeat": 0, "scores": {"overall": 5}},
        {"run_id": "p1", "item_id": "it2", "repeat": 0, "scores": {"overall": 4}},
        {"run_id": "p2", "item_id": "it1", "repeat": 0, "scores": {"overall": 3}},
    ]
    grouped = group_judged_by_run(records)
    assert [s["overall"] for s in grouped["p1"]] == [5, 4]
    assert [s["overall"] for s in grouped["p2"]] == [3]


def test_load_inputs_meta_reads_case_type_and_format(tmp_path):
    import json
    path = tmp_path / "inputs.json"
    data = {
        "entries": [
            {"id": "p1", "format": "mc4", "case_type": "normal", "subject": "경제", "passage_text": "지문"},
            {"id": "p2", "format": "data", "case_type": "hard", "subject": "지리", "passage_text": "지문2"},
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    meta = load_inputs_meta(str(path))
    assert meta == {
        "p1": {"case_type": "normal", "format": "mc4"},
        "p2": {"case_type": "hard", "format": "data"},
    }
