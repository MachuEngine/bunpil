"""evals/eval_item_quality_runs.py 순수 로직 단위 테스트. LLM 호출 없음.

가짜 run 레코드(golden_gen/gen_item_quality_golden.py `_run_passage()`가 만드는 형태)와
가짜 judged 캐시 레코드(evals/eval_item_quality_runs.py `judge` 서브커맨드가 쓰는 형태)만
사용한다. requirements.txt 의존성만 필요(이 모듈 자체는 app.* 모듈 레벨 import가 없어 가볍다).

reliability(2차 선별 분석) 테스트는 사람 라벨도 전부 가짜다 — 실제 라벨링은 아직 이뤄지지
않았고(iq_golden.json의 human_label은 전부 null), 조인·제외·반복 평균·짝지은 비교·생성
모델별 편향 로직이 올바른지를 가짜 숫자로 미리 검증한다."""
import json

import pytest

from evals.eval_item_quality_runs import (
    aggregate_ceiling_discrimination,
    bias_by_generation_model,
    build_rows,
    classify_passage,
    compute_generator_comparison,
    compute_metrics,
    compute_reliability_report,
    compute_rule_metrics,
    compute_structure_section,
    coverage_passage_ids,
    gate_judge_contrast,
    gate_judge_diff_by_passage,
    generator_pairs,
    group_judged_by_run,
    human_case_type_means,
    human_label_distribution,
    human_metrics_for_model,
    human_rows_for_model,
    human_score_by_passage,
    judge_avg_rounded,
    judge_avg_score,
    judge_case_type_means,
    judge_metrics_for_model,
    judged_keys,
    load_inputs_meta,
    load_labeled_keys,
    load_passage_texts,
    mc_items_for_run,
    paired_abs_error_diff,
    paired_bias_diff_ci,
    paired_ci_verdict,
    passage_level_final_pass,
    passage_level_judge_overall,
    passage_level_latency,
    per_repeat_kappas,
    quality_stats_by_criteria,
    repeats_for_item,
    runtime_luna_overall_by_passage,
    score_distribution,
    struct_judge_mean_by_run,
    struct_judge_targets,
    struct_judged_keys,
    structure_gate_pass_rate,
    structure_quality_stats,
    usable_human_labels,
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


# ── judge --only-labeled: labeled_keys 필터 ───────────────────────────────

def test_mc_items_for_run_filters_by_labeled_keys_when_given():
    run = {"id": "p1", "items": [_mc("it1"), _mc("it2"), _mc("it3")], "error": None}
    labeled_keys = {("p1", "it1"), ("p1", "it3")}
    result = mc_items_for_run(run, labeled_keys)
    assert [it["item_id"] for it in result] == ["it1", "it3"]


def test_mc_items_for_run_none_labeled_keys_keeps_all_mc_items():
    run = {"id": "p1", "items": [_mc("it1"), _mc("it2")], "error": None}
    result = mc_items_for_run(run, None)
    assert [it["item_id"] for it in result] == ["it1", "it2"]


def test_load_labeled_keys_groups_by_model_and_drops_entries_outside_golden(tmp_path):
    golden_path = tmp_path / "golden.json"
    model_map_path = tmp_path / "model_map.json"
    golden_path.write_text(
        json.dumps({"entries": [{"id": "iq_001"}, {"id": "iq_002"}]}, ensure_ascii=False), encoding="utf-8"
    )
    model_map_path.write_text(
        json.dumps(
            {
                "map": {
                    "iq_001": {"model": "qwen2.5-14b", "passage_id": "p1", "run_item_id": "it1"},
                    "iq_002": {"model": "gpt-6-luna", "passage_id": "p1", "run_item_id": "it9"},
                    # golden.json entries에 없는 블라인드 id — missing 조합이라 제외돼야 함
                    "iq_999": {"model": "qwen2.5-14b", "passage_id": "p9", "run_item_id": "it9"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    keys = load_labeled_keys(str(golden_path), str(model_map_path))
    assert keys == {
        "qwen2.5-14b": {("p1", "it1")},
        "gpt-6-luna": {("p1", "it9")},
    }


# ── reliability: 조인·제외·반복 평균·짝지은 비교·생성 모델별 편향 ──────────
#
# 가짜 사람 라벨 + 가짜 judged 캐시로 검증한다. 실제 사람 라벨은 아직 없다(라벨링 전).
#
# 블라인드 id   passage  생성모델         human(정답유일성,오답매력도,근거성)  비고
# iq_001       p1       qwen2.5-14b      (5, 4, 5)                           정상(사용)
# iq_002       p1       gpt-6-luna       (3, 3, 4)                           정상(사용)
# iq_003       p2       qwen2.5-14b      cannot_judge=True                   제외
# iq_004       p2       gpt-6-luna       정답유일성=None                      제외(점수 null)

def _human_label(정답유일성=5, 오답매력도=4, 근거성=5, 학생난이도="상", cannot_judge=False):
    return {
        "정답유일성": 정답유일성, "오답매력도": 오답매력도, "근거성": 근거성,
        "학생난이도": 학생난이도, "cannot_judge": cannot_judge, "reason": "",
    }


_RELIABILITY_GOLDEN_ENTRIES = [
    {"id": "iq_001", "human_label": _human_label(5, 4, 5, "상")},
    {"id": "iq_002", "human_label": _human_label(3, 3, 4, "중")},
    {"id": "iq_003", "human_label": _human_label(cannot_judge=True)},
    {"id": "iq_004", "human_label": {**_human_label(오답매력도=3, 근거성=4, 학생난이도="하"), "정답유일성": None}},
]

_RELIABILITY_MODEL_MAP = {
    "iq_001": {"model": "qwen2.5-14b", "passage_id": "p1", "run_item_id": "it1"},
    "iq_002": {"model": "gpt-6-luna", "passage_id": "p1", "run_item_id": "it9"},
    "iq_003": {"model": "qwen2.5-14b", "passage_id": "p2", "run_item_id": "it2"},
    "iq_004": {"model": "gpt-6-luna", "passage_id": "p2", "run_item_id": "it8"},
}


def _judge_scores(정답유일성=4, 오답매력도=4, 근거성=4):
    return {
        "정답유일성": 정답유일성, "오답매력도": 오답매력도, "근거성": 근거성,
        "overall": round((정답유일성 + 오답매력도 + 근거성) / 3, 2), "parse_failed": False,
    }


def _rec(run_id, item_id, repeat, scores):
    return {"run_id": run_id, "item_id": item_id, "repeat": repeat, "scores": scores}


def _reliability_judged_by_judge():
    """judgeA: iq_001(2회 반복)·iq_002(1회) 모두 캐시 있음.
    judgeB: iq_001만(1회) 캐시 있음 — gpt-6-luna 캐시가 없어 iq_002는 judgeB 없음."""
    return {
        "judgeA": {
            "qwen2.5-14b": [
                _rec("p1", "it1", 0, _judge_scores(5, 4, 5)),
                _rec("p1", "it1", 1, _judge_scores(4, 4, 5)),
            ],
            "gpt-6-luna": [
                _rec("p1", "it9", 0, _judge_scores(3, 3, 4)),
            ],
        },
        "judgeB": {
            "qwen2.5-14b": [
                _rec("p1", "it1", 0, _judge_scores(4, 4, 4)),
            ],
            "gpt-6-luna": [],
        },
    }


def test_usable_human_labels_excludes_cannot_judge_and_null_scores():
    usable, excluded = usable_human_labels(_RELIABILITY_GOLDEN_ENTRIES)
    assert set(usable.keys()) == {"iq_001", "iq_002"}
    assert excluded == 2


def test_repeats_for_item_sorts_by_repeat_and_returns_scores_only():
    records = [
        _rec("p1", "it1", 1, _judge_scores(4, 4, 5)),
        _rec("p1", "it1", 0, _judge_scores(5, 4, 5)),
        _rec("p1", "it9", 0, _judge_scores(3, 3, 4)),  # 다른 item — 매칭 안 됨
    ]
    scores_list = repeats_for_item(records, "p1", "it1")
    assert [s["정답유일성"] for s in scores_list] == [5, 4]


def test_repeats_for_item_no_match_returns_empty():
    assert repeats_for_item([_rec("p1", "it1", 0, _judge_scores())], "p1", "it9") == []


def test_build_rows_joins_and_reports_excluded_count():
    rows, excluded = build_rows(
        _RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP, _reliability_judged_by_judge()
    )
    assert excluded == 2
    by_id = {r["id"]: r for r in rows}
    assert set(by_id) == {"iq_001", "iq_002"}

    row1 = by_id["iq_001"]
    assert row1["passage_id"] == "p1"
    assert row1["model"] == "qwen2.5-14b"
    assert row1["human"] == {"정답유일성": 5, "오답매력도": 4, "근거성": 5}
    assert [s["정답유일성"] for s in row1["judges"]["judgeA"]] == [5, 4]
    assert [s["정답유일성"] for s in row1["judges"]["judgeB"]] == [4]

    row2 = by_id["iq_002"]
    assert row2["model"] == "gpt-6-luna"
    assert "judgeA" in row2["judges"]
    assert "judgeB" not in row2["judges"]  # gpt-6-luna 캐시가 judgeB엔 없음


def test_judge_avg_score_and_rounded():
    scores_list = [_judge_scores(정답유일성=4), _judge_scores(정답유일성=5), _judge_scores(정답유일성=5)]
    avg = judge_avg_score(scores_list, "정답유일성")
    assert avg == pytest.approx(14 / 3)
    assert judge_avg_rounded(scores_list, "정답유일성") == round(14 / 3)  # == 5


def test_per_repeat_kappas_none_when_repeat_index_has_fewer_than_two_pairs():
    rows_with_judge = [
        {
            "human": {"정답유일성": 5}, "passage_id": "p1",
            "judges": {"judgeA": [_judge_scores(정답유일성=5), _judge_scores(정답유일성=4)]},
        },
        {
            "human": {"정답유일성": 3}, "passage_id": "p1-2",
            "judges": {"judgeA": [_judge_scores(정답유일성=3)]},  # repeat 1 없음
        },
    ]
    kappas = per_repeat_kappas(rows_with_judge, "judgeA", "정답유일성")
    assert len(kappas) == 2
    assert kappas[0] is not None  # repeat 0: 두 행 모두 있음 → 계산 가능
    assert kappas[1] is None  # repeat 1: 표본 1건뿐 → None


def test_paired_abs_error_diff_only_includes_items_with_both_judges():
    rows, _ = build_rows(
        _RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP, _reliability_judged_by_judge()
    )
    diff_pairs = paired_abs_error_diff(rows, "judgeA", "judgeB", "정답유일성")
    # iq_002는 judgeB 캐시가 없어 짝지은 비교에서 제외 — iq_001만 남는다.
    assert len(diff_pairs) == 1
    passage_id, diff = diff_pairs[0]
    assert passage_id == "p1"
    # judgeA 평균=(5+4)/2=4.5, human=5 → err_a=0.5 / judgeB=4, human=5 → err_b=1.0
    assert diff == pytest.approx(0.5 - 1.0)
    assert diff < 0  # 음수면 judgeA가 이 문항에서 더 정확(오차가 더 작음)


def test_bias_by_generation_model_splits_by_model():
    rows, _ = build_rows(
        _RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP, _reliability_judged_by_judge()
    )
    bias = bias_by_generation_model(rows, "judgeA", "정답유일성")
    assert bias["qwen2.5-14b"]["n"] == 1
    assert bias["qwen2.5-14b"]["bias"] == pytest.approx(4.5 - 5)  # iq_001
    assert bias["gpt-6-luna"]["n"] == 1
    assert bias["gpt-6-luna"]["bias"] == pytest.approx(3 - 3)  # iq_002


def test_human_label_distribution_excludes_cannot_judge_but_keeps_partial_null_entry():
    dist = human_label_distribution(_RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP)
    assert dist["cannot_judge_n"] == 1
    # iq_003(cannot_judge)만 제외 — iq_004는 정답유일성만 null이라 다른 기준·난이도는 집계된다.
    assert dist["score_dist"]["정답유일성"] == {1: 0, 2: 0, 3: 1, 4: 0, 5: 1}  # iq_001(5), iq_002(3)
    assert dist["score_dist"]["오답매력도"][4] == 1  # iq_001
    assert dist["score_dist"]["오답매력도"][3] == 2  # iq_002, iq_004
    assert dist["difficulty_overall"] == {"상": 1, "중": 1, "하": 1}
    assert dist["difficulty_by_model"]["gpt-6-luna"] == {"상": 0, "중": 1, "하": 1}  # iq_002, iq_004
    assert dist["difficulty_by_model"]["qwen2.5-14b"] == {"상": 1, "중": 0, "하": 0}  # iq_001


def test_compute_reliability_report_end_to_end_with_fake_data():
    report = compute_reliability_report(
        _RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP, _reliability_judged_by_judge(),
        ["judgeA", "judgeB"],
    )
    assert report["n_usable"] == 2
    assert report["n_excluded"] == 2

    criterion_report = report["criteria"]["정답유일성"]
    assert criterion_report["per_judge"]["judgeA"]["n"] == 2
    # judgeB는 iq_001만 있어 n=1 — 표본 부족으로 note만 남고 κ 등은 계산하지 않는다.
    assert criterion_report["per_judge"]["judgeB"]["n"] == 1
    assert "note" in criterion_report["per_judge"]["judgeB"]

    pairwise = criterion_report["pairwise"]["judgeA vs judgeB"]
    assert "note" in pairwise  # 짝지은 비교도 표본 1건이라 판정 불가 note

    # 문항별 행이 원문 없이 저장돼 재분석(재호출 없이)이 가능해야 한다.
    for row in report["rows"]:
        assert "question" not in row
        assert "passage_text" not in row


# ── compare-generators: 규칙 지표 ──────────────────────────────────────────

def test_compute_rule_metrics_basic():
    records = [
        {
            "id": "p1", "error": None, "num_items": 2, "items": [_mc("it1"), _mc("it2")],
            "validation_passed": True, "attempts": 1,
            "attempt_log": [{"validation_passed": True, "malformed_retries": 0}],
            "wall_clock_sec": 10.0, "cost_usd_est": 0.01,
        },
        {
            "id": "p2", "error": None, "num_items": 2, "items": [_mc("it3")],  # 개수 불일치
            "validation_passed": False, "attempts": 3,
            "attempt_log": [
                {"validation_passed": False, "malformed_retries": 1},
                {"validation_passed": False, "malformed_retries": 0},
                {"validation_passed": True, "malformed_retries": 0},
            ],
            "wall_clock_sec": 30.0, "cost_usd_est": 0.03,
        },
        {"id": "p3", "error": {"type": "RuntimeError"}, "wall_clock_sec": 1.0},  # 실패(객관식 0개)
    ]
    m = compute_rule_metrics(records)
    assert m["n"] == 3
    # 분모는 전체 지문 수(3) — run 실패·개수 불일치도 실패로 포함
    assert m["count_ok_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["first_pass_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["final_gate_pass_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["avg_attempts"] == pytest.approx(2.0)  # 성공 케이스(p1,p2)만: (1+3)/2
    assert m["malformed_retries_total"] == 1
    assert m["avg_wall_clock_sec"] == pytest.approx((10 + 30 + 1) / 3, abs=1e-3)
    assert m["median_wall_clock_sec"] == pytest.approx(10.0)
    assert m["avg_cost_usd"] == pytest.approx(0.02)
    assert m["n_zero_mc_passages"] == 1  # p3만 객관식 0개


def test_compute_rule_metrics_empty_records_no_zero_division():
    m = compute_rule_metrics([])
    assert m["n"] == 0
    assert m["count_ok_rate"] == 0.0
    assert m["avg_cost_usd"] is None
    assert m["n_zero_mc_passages"] == 0


# ── compare-generators: Judge/사람 지표(quality_stats_by_criteria 공유) ──────

def test_quality_stats_by_criteria_computes_mean_and_low_quality_rate():
    rows = [
        ("p1", _score(5, 5, 5)),
        ("p1", _score(1, 1, 1)),
        ("p2", _score(3, 3, 3)),
    ]
    stats = quality_stats_by_criteria(rows)
    assert stats["정답유일성"]["n"] == 3
    assert stats["정답유일성"]["mean"] == pytest.approx(3.0)
    assert stats["정답유일성"]["low_quality_rate"] == pytest.approx(1 / 3, abs=1e-3)
    # 정답유일성 치명적 실패율은 정답유일성.low_quality_rate와 같은 값의 별도 노출
    assert stats["uniqueness_critical_fail_rate"] == stats["정답유일성"]["low_quality_rate"]
    assert stats["uniqueness_critical_fail_ci"] == stats["정답유일성"]["low_quality_ci"]


def test_judge_metrics_for_model_uses_run_id_as_cluster():
    judged_records = [
        {"run_id": "p1", "scores": _score(5, 5, 5)},
        {"run_id": "p1", "scores": _score(1, 1, 1)},
        {"run_id": "p2", "scores": _score(3, 3, 3)},
    ]
    jm = judge_metrics_for_model(judged_records)
    assert jm["정답유일성"]["n"] == 3


def test_human_rows_for_model_filters_by_model_and_usable():
    rows = human_rows_for_model(_RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP, "qwen2.5-14b")
    assert len(rows) == 1  # iq_001만(iq_003은 cannot_judge라 usable에서 제외)
    assert rows[0]["passage_id"] == "p1"
    assert rows[0]["human"]["정답유일성"] == 5


def test_human_metrics_for_model_uses_passage_id_as_cluster():
    rows = human_rows_for_model(_RELIABILITY_GOLDEN_ENTRIES, _RELIABILITY_MODEL_MAP, "gpt-6-luna")
    hm = human_metrics_for_model(rows)
    assert hm["정답유일성"]["n"] == 1  # iq_002만(iq_004는 정답유일성 null이라 제외)
    assert hm["정답유일성"]["mean"] == pytest.approx(3.0)


# ── compare-generators: 짝지은 비교 재료(패시지 단위 값 dict) ────────────────

def test_passage_level_judge_overall_averages_per_passage():
    judged_records = [
        _rec("p1", "it1", 0, _judge_scores(5, 5, 5)),
        _rec("p1", "it1", 1, _judge_scores(3, 3, 3)),
        _rec("p2", "it9", 0, _judge_scores(4, 4, 4)),
    ]
    overall = passage_level_judge_overall(judged_records)
    assert overall["p1"] == pytest.approx((5.0 + 3.0) / 2)
    assert overall["p2"] == pytest.approx(4.0)


def test_passage_level_final_pass_maps_id_to_int_and_treats_error_as_fail():
    records = [
        {"id": "p1", "validation_passed": True, "error": None},
        {"id": "p2", "validation_passed": False, "error": None},
        {"id": "p3", "error": {"type": "X"}},
    ]
    assert passage_level_final_pass(records) == {"p1": 1, "p2": 0, "p3": 0}


def test_passage_level_latency_maps_id_to_wall_clock():
    records = [{"id": "p1", "wall_clock_sec": 5.0}, {"id": "p2", "wall_clock_sec": 2.0}]
    assert passage_level_latency(records) == {"p1": 5.0, "p2": 2.0}


def test_human_score_by_passage_extracts_single_criterion():
    rows = [{"passage_id": "p1", "human": {"정답유일성": 5, "오답매력도": 4, "근거성": 3}}]
    assert human_score_by_passage(rows, "오답매력도") == {"p1": 4}


# ── compare-generators: paired_ci_verdict(짝지은 비교의 부호·판정 불가 분기) ──

def test_paired_ci_verdict_b_better_when_higher_is_better_and_diff_positive():
    values_a = {"p1": 3.0, "p2": 3.0, "p3": 3.0, "p4": 3.0}
    values_b = {"p1": 5.0, "p2": 5.0, "p3": 5.0, "p4": 5.0}
    result = paired_ci_verdict(values_a, values_b, higher_is_better=True, label_a="A", label_b="B")
    assert result["n"] == 4
    assert result["mean_diff"] == pytest.approx(2.0)
    assert result["verdict"] == "B가 더 우수"


def test_paired_ci_verdict_a_better_when_lower_is_better_and_diff_positive():
    # 지연처럼 작을수록 좋은 지표 — b가 더 크면(느리면) a가 더 우수
    values_a = {"p1": 1.0, "p2": 1.0, "p3": 1.0, "p4": 1.0}
    values_b = {"p1": 3.0, "p2": 3.0, "p3": 3.0, "p4": 3.0}
    result = paired_ci_verdict(values_a, values_b, higher_is_better=False, label_a="A", label_b="B")
    assert result["verdict"] == "A가 더 우수"


def test_paired_ci_verdict_inconclusive_when_ci_spans_zero():
    values_a = {"p1": 3.0, "p2": 5.0, "p3": 2.0, "p4": 4.0}
    values_b = {"p1": 3.5, "p2": 2.0, "p3": 5.0, "p4": 3.0}
    result = paired_ci_verdict(values_a, values_b, higher_is_better=True, label_a="A", label_b="B")
    assert "판정 불가" in result["verdict"]


def test_paired_ci_verdict_only_uses_common_passages():
    values_a = {"p1": 3.0, "p2": 3.0}
    values_b = {"p2": 5.0, "p3": 5.0}  # p1은 b에 없음(missing), p3는 a에 없음 — 공통은 p2뿐
    result = paired_ci_verdict(values_a, values_b, higher_is_better=True, label_a="A", label_b="B")
    assert result["n"] == 1


def test_paired_ci_verdict_empty_common_returns_no_sample():
    result = paired_ci_verdict({"p1": 1.0}, {"p2": 2.0}, higher_is_better=True, label_a="A", label_b="B")
    assert result["n"] == 0
    assert result["mean_diff"] is None
    assert "표본 없음" in result["verdict"]


# ── compare-generators: case_type별 평균(참고용) ────────────────────────────

def test_judge_case_type_means_groups_by_case_type_simple_average():
    judged_records = [
        {"run_id": "p1", "scores": _score(5, 4, 3)},
        {"run_id": "p2", "scores": _score(3, 2, 1)},
    ]
    inputs_meta = {"p1": {"case_type": "normal"}, "p2": {"case_type": "hard"}}
    result = judge_case_type_means(judged_records, inputs_meta)
    assert result["normal"]["정답유일성"] == pytest.approx(5.0)
    assert result["hard"]["정답유일성"] == pytest.approx(3.0)


def test_human_case_type_means_groups_by_case_type_simple_average():
    rows = [
        {"passage_id": "p1", "human": {"정답유일성": 5, "오답매력도": 4, "근거성": 3}},
        {"passage_id": "p2", "human": {"정답유일성": 1, "오답매력도": 1, "근거성": 1}},
    ]
    inputs_meta = {"p1": {"case_type": "normal"}, "p2": {"case_type": "normal"}}
    result = human_case_type_means(rows, inputs_meta)
    assert result["normal"]["정답유일성"] == pytest.approx(3.0)


# ── compare-generators: compute_generator_comparison 통합(쌍 구성 + missing 처리) ──
#
# 모델 3개(qwen2.5-14b 기준/gpt-6-luna/gemini-3.8-flash), 지문 2개(p1, p2).
# gpt-6-luna는 p2에서 run이 실패한다(= 객관식 0개, missing) — 규칙 지표에는 실패로
# 잡히지만, Judge·사람 점수 짝지은 비교에서는 그 지문이 자연히 제외돼야 한다(값 dict에
# 키 자체가 없으므로 공통 지문 교집합에서 빠짐).

def _gen_cmp_record(pid, ok=True, wall_clock=10.0):
    if not ok:
        return {"id": pid, "error": {"type": "X"}, "wall_clock_sec": wall_clock}
    return {
        "id": pid, "error": None, "num_items": 1, "items": [_mc(f"{pid}-it1")],
        "validation_passed": True, "attempts": 1,
        "attempt_log": [{"validation_passed": True, "malformed_retries": 0}],
        "wall_clock_sec": wall_clock, "cost_usd_est": 0.0,
    }


def test_compute_generator_comparison_pairs_and_missing_passage_handling():
    records_by_model = {
        "qwen2.5-14b": [_gen_cmp_record("p1", wall_clock=10.0), _gen_cmp_record("p2", wall_clock=10.0)],
        "gpt-6-luna": [_gen_cmp_record("p1", wall_clock=5.0), _gen_cmp_record("p2", ok=False, wall_clock=2.0)],
        "gemini-3.8-flash": [_gen_cmp_record("p1", wall_clock=7.0), _gen_cmp_record("p2", wall_clock=7.0)],
    }
    judged_by_model = {
        "qwen2.5-14b": [_rec("p1", "it1", 0, _judge_scores(3, 3, 3)), _rec("p2", "it1", 0, _judge_scores(3, 3, 3))],
        "gpt-6-luna": [_rec("p1", "it1", 0, _judge_scores(5, 5, 5))],  # p2는 run 실패라 judged 캐시도 없음
        "gemini-3.8-flash": [_rec("p1", "it1", 0, _judge_scores(4, 4, 4)), _rec("p2", "it1", 0, _judge_scores(4, 4, 4))],
    }
    golden_entries = [
        {"id": "iq_001", "human_label": _human_label(5, 4, 5, "상")},  # p1, qwen
        {"id": "iq_002", "human_label": _human_label(3, 3, 4, "중")},  # p1, gpt-6-luna
        {"id": "iq_003", "human_label": _human_label(4, 4, 4, "중")},  # p1, gemini
        {"id": "iq_004", "human_label": _human_label(5, 4, 5, "상")},  # p2, qwen
        {"id": "iq_005", "human_label": _human_label(4, 4, 4, "중")},  # p2, gemini(gpt-6-luna는 p2 missing)
    ]
    model_map = {
        "iq_001": {"model": "qwen2.5-14b", "passage_id": "p1", "run_item_id": "it1"},
        "iq_002": {"model": "gpt-6-luna", "passage_id": "p1", "run_item_id": "it1"},
        "iq_003": {"model": "gemini-3.8-flash", "passage_id": "p1", "run_item_id": "it1"},
        "iq_004": {"model": "qwen2.5-14b", "passage_id": "p2", "run_item_id": "it1"},
        "iq_005": {"model": "gemini-3.8-flash", "passage_id": "p2", "run_item_id": "it1"},
    }
    inputs_meta = {"p1": {"case_type": "normal", "format": "mc4"}, "p2": {"case_type": "hard", "format": "mc4"}}

    report = compute_generator_comparison(
        ["qwen2.5-14b", "gpt-6-luna", "gemini-3.8-flash"], "qwen2.5-14b", "anthropic/claude-sonnet-5.5",
        records_by_model, judged_by_model, golden_entries, model_map, inputs_meta, "abc1234",
    )

    # 규칙 지표에는 gpt-6-luna의 p2 실패가 객관식 0개로 잡힌다.
    assert report["rule_metrics"]["gpt-6-luna"]["n_zero_mc_passages"] == 1
    assert report["rule_metrics"]["gpt-6-luna"]["final_gate_pass_rate"] == pytest.approx(0.5)  # p1만 통과

    # 쌍 구성: 기준(qwen) vs 나머지 2개 + gpt-6-luna vs gemini-3.8-flash 명시 쌍 = 3개.
    assert set(report["pairwise"]) == {
        "qwen2.5-14b_vs_gpt-6-luna", "qwen2.5-14b_vs_gemini-3.8-flash", "gpt-6-luna_vs_gemini-3.8-flash",
    }

    qwen_vs_gpt = report["pairwise"]["qwen2.5-14b_vs_gpt-6-luna"]
    # Judge·사람 점수 쌍은 p2가 gpt-6-luna에 없어 공통 지문이 p1 하나뿐 — missing이 빠진다.
    assert qwen_vs_gpt["judge_overall_diff"]["n"] == 1
    assert qwen_vs_gpt["human_diff"]["정답유일성"]["n"] == 1
    # 규칙 성격의 최종 게이트·지연 차이는 run 레코드가 있는 지문 전부(p1, p2) 기준 —
    # gpt-6-luna의 p2 실패가 "통과 못함(0)"으로 그대로 들어간다.
    assert qwen_vs_gpt["final_gate_diff"]["n"] == 2
    assert qwen_vs_gpt["latency_diff"]["n"] == 2

    gpt_vs_gemini = report["pairwise"]["gpt-6-luna_vs_gemini-3.8-flash"]
    assert gpt_vs_gemini["judge_overall_diff"]["n"] == 1  # p1만 공통

    assert report["missing_bias_note"]
    assert report["gate_dependency_note"]
    assert report["code_version"] == "abc1234"
    # 콘솔/저장 출력에 문항 원문이 섞이지 않는다(하드룰 4) — report 전체를 문자열로
    # 직렬화해도 question 키 자체가 어디에도 없어야 한다.
    assert "question" not in json.dumps(report, ensure_ascii=False)


# ── structure-judge: 캐시 대상 선정/스킵 ──────────────────────────────────

def test_load_passage_texts_reads_id_to_passage_text(tmp_path):
    path = tmp_path / "inputs.json"
    data = {"entries": [{"id": "p1", "passage_text": "지문1"}, {"id": "p2", "passage_text": "지문2"}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert load_passage_texts(str(path)) == {"p1": "지문1", "p2": "지문2"}


def test_struct_judge_targets_excludes_runs_without_items():
    records = [
        {"id": "p1", "items": [_mc("it1")]},
        {"id": "p2", "items": []},  # run 실패 등 — 제외
        {"id": "p3"},  # items 키 자체가 없는 경우도 제외
    ]
    targets = struct_judge_targets(records)
    assert [r["id"] for r in targets] == ["p1"]


def test_struct_judged_keys_builds_skip_set():
    records = [
        {"run_id": "p1", "repeat": 0, "result": {}},
        {"run_id": "p1", "repeat": 1, "result": {}},
        {"run_id": "p2", "repeat": 0, "result": {}},
    ]
    keys = struct_judged_keys(records)
    assert keys == {("p1", 0), ("p1", 1), ("p2", 0)}
    assert ("p1", 2) not in keys


# ── structure-judge 재채점 캐시 집계/대조 ──────────────────────────────────

def _struct_rec(run_id, repeat, type_ratio=1.0, difficulty_match=True, overall=5):
    return {
        "run_id": run_id, "model": "m", "repeat": repeat,
        "result": {"type_ratio_score": type_ratio, "difficulty_match": difficulty_match, "overall_score": overall},
        "judge_model": "anthropic/claude-sonnet-5.5",
    }


def test_struct_judge_mean_by_run_averages_repeats():
    records = [_struct_rec("p1", 0, 1.0, True, 5), _struct_rec("p1", 1, 0.5, True, 3)]
    mean_by_run = struct_judge_mean_by_run(records)
    assert mean_by_run["p1"]["type_ratio_score"] == pytest.approx(0.75)
    assert mean_by_run["p1"]["overall_score"] == pytest.approx(4.0)
    assert mean_by_run["p1"]["difficulty_match"] is True
    assert mean_by_run["p1"]["n_repeats"] == 2


def test_struct_judge_mean_by_run_difficulty_match_majority_tie_counts_as_true():
    # 반복 2회 중 1회만 true — 0.5 >= 0.5라 True로 판정(동점은 true 쪽).
    records = [_struct_rec("p1", 0, 1.0, True, 5), _struct_rec("p1", 1, 1.0, False, 5)]
    mean_by_run = struct_judge_mean_by_run(records)
    assert mean_by_run["p1"]["difficulty_match"] is True


def test_struct_judge_mean_by_run_difficulty_match_minority_false():
    records = [
        _struct_rec("p1", 0, 1.0, False, 5),
        _struct_rec("p1", 1, 1.0, False, 5),
        _struct_rec("p1", 2, 1.0, True, 5),
    ]
    mean_by_run = struct_judge_mean_by_run(records)
    assert mean_by_run["p1"]["difficulty_match"] is False


def test_structure_quality_stats_computes_mean_ci_and_rates():
    mean_by_run = {
        "p1": {"type_ratio_score": 1.0, "overall_score": 5, "difficulty_match": True, "n_repeats": 1},
        "p2": {"type_ratio_score": 0.5, "overall_score": 3, "difficulty_match": False, "n_repeats": 1},
    }
    stats = structure_quality_stats(mean_by_run)
    assert stats["n"] == 2
    assert stats["overall_mean"] == pytest.approx(4.0)
    assert stats["type_ratio_mean"] == pytest.approx(0.75)
    assert stats["difficulty_match_rate"] == pytest.approx(0.5)


def test_structure_quality_stats_empty_returns_none_without_crashing():
    stats = structure_quality_stats({})
    assert stats["n"] == 0
    assert stats["overall_mean"] is None


# 런타임 게이트 조건(graph.py validate_node와 동일): count_match and type_ratio>=0.5
# and difficulty_match and overall>=3. structure_gate_pass_rate가 sonnet 점수에 이 조건을
# 그대로 재적용하는지 경계값으로 확인한다.
def test_structure_gate_pass_rate_reapplies_runtime_gate_condition():
    mean_by_run = {
        "p1": {"type_ratio_score": 0.5, "overall_score": 3, "difficulty_match": True},  # 경계값 — 통과
        "p2": {"type_ratio_score": 0.4, "overall_score": 5, "difficulty_match": True},  # type_ratio 미달
        "p3": {"type_ratio_score": 1.0, "overall_score": 5, "difficulty_match": True},  # count_match 불일치
    }
    records_by_id = {
        "p1": {"num_items": 1, "items": [_mc("it1")]},
        "p2": {"num_items": 1, "items": [_mc("it1")]},
        "p3": {"num_items": 2, "items": [_mc("it1")]},  # 목표 2개인데 1개만 저장됨
    }
    result = structure_gate_pass_rate(mean_by_run, records_by_id)
    assert result["n"] == 3
    assert result["pass_rate"] == pytest.approx(round(1 / 3, 3))


def test_structure_gate_pass_rate_empty_returns_none_without_crashing():
    assert structure_gate_pass_rate({}, {}) == {"n": 0, "pass_rate": None}


def test_runtime_luna_overall_by_passage_uses_last_attempt_and_skips_missing():
    records = [
        {"id": "p1", "attempt_log": [{"judge_result": {"overall_score": 2}}, {"judge_result": {"overall_score": 4}}]},
        {"id": "p2", "attempt_log": []},  # run 실패 등 — 제외
        {"id": "p3", "attempt_log": [{"validation_feedback": "x"}]},  # judge_result 자체가 없음 — 제외
    ]
    overall = runtime_luna_overall_by_passage(records)
    assert overall == {"p1": 4}  # 최종(마지막) 시도 점수만


def test_gate_judge_contrast_only_common_passages_and_sign():
    records = [
        {"id": "p1", "attempt_log": [{"judge_result": {"overall_score": 5}}]},
        {"id": "p2", "attempt_log": [{"judge_result": {"overall_score": 3}}]},  # struct_mean에 없음
    ]
    struct_mean_by_run = {
        "p1": {"overall_score": 3, "type_ratio_score": 1.0, "difficulty_match": True},
        "p3": {"overall_score": 1, "type_ratio_score": 1.0, "difficulty_match": True},  # luna 쪽에 없음
    }
    result = gate_judge_contrast(records, struct_mean_by_run)
    assert result["n"] == 1  # p1만 공통
    assert result["mean_diff"] == pytest.approx(2.0)  # luna(5) - sonnet(3) = 2, 양수면 luna가 더 후함


def test_gate_judge_diff_by_passage_matches_contrast_values():
    records = [{"id": "p1", "attempt_log": [{"judge_result": {"overall_score": 5}}]}]
    struct_mean_by_run = {"p1": {"overall_score": 3, "type_ratio_score": 1.0, "difficulty_match": True}}
    assert gate_judge_diff_by_passage(records, struct_mean_by_run) == {"p1": pytest.approx(2.0)}


def test_paired_bias_diff_ci_positive_when_b_more_biased():
    diff_a = {"p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}
    diff_b = {"p1": 2.0, "p2": 2.0, "p3": 2.0, "p4": 2.0}
    result = paired_bias_diff_ci(diff_a, diff_b)
    assert result["n"] == 4
    assert result["mean_diff"] == pytest.approx(2.0)


def test_paired_bias_diff_ci_negative_when_a_more_biased():
    diff_a = {"p1": 2.0, "p2": 2.0, "p3": 2.0, "p4": 2.0}
    diff_b = {"p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}
    result = paired_bias_diff_ci(diff_a, diff_b)
    assert result["mean_diff"] == pytest.approx(-2.0)


def test_paired_bias_diff_ci_no_common_passages_returns_no_sample():
    result = paired_bias_diff_ci({"p1": 1.0}, {"p2": 1.0})
    assert result == {"n": 0, "mean_diff": None, "ci": [None, None]}


# ── compare-generators: 구조 절(compute_structure_section) ────────────────

def test_compute_structure_section_returns_none_when_no_struct_cache_anywhere():
    result = compute_structure_section(
        ["qwen2.5-14b", "gpt-6-luna"], "qwen2.5-14b", "anthropic/claude-sonnet-5.5",
        {}, {"qwen2.5-14b": [], "gpt-6-luna": []},
    )
    assert result is None


def _structure_run(pid, overall_luna, num_items=1, n_items=1, validation_passed=True):
    return {
        "id": pid, "num_items": num_items, "items": [_mc(f"{pid}-it{i}") for i in range(n_items)],
        "validation_passed": validation_passed, "error": None,
        "attempt_log": [{"validation_passed": validation_passed, "judge_result": {"overall_score": overall_luna}}],
    }


def test_compute_structure_section_end_to_end_with_fake_cache():
    models = ["qwen2.5-14b", "gpt-6-luna"]
    records_by_model = {
        "qwen2.5-14b": [_structure_run("p1", overall_luna=2), _structure_run("p2", overall_luna=5)],
        "gpt-6-luna": [_structure_run("p1", overall_luna=5), _structure_run("p2", overall_luna=5)],
    }
    struct_judged_by_model = {
        "qwen2.5-14b": [_struct_rec("p1", 0, 1.0, True, 3), _struct_rec("p2", 0, 1.0, True, 4)],
        "gpt-6-luna": [_struct_rec("p1", 0, 1.0, True, 5), _struct_rec("p2", 0, 1.0, True, 5)],
    }

    report = compute_structure_section(
        models, "qwen2.5-14b", "anthropic/claude-sonnet-5.5", records_by_model, struct_judged_by_model,
    )

    assert report is not None
    assert report["judge_model"] == "anthropic/claude-sonnet-5.5"

    assert report["quality"]["qwen2.5-14b"]["n"] == 2
    assert report["quality"]["qwen2.5-14b"]["overall_mean"] == pytest.approx(3.5)
    assert report["quality"]["gpt-6-luna"]["overall_mean"] == pytest.approx(5.0)

    # sonnet 게이트 재적용: 둘 다 count_match·type_ratio·difficulty_match·overall(>=3) 충족 → 전부 통과.
    assert report["gate"]["qwen2.5-14b"]["sonnet_gate"]["pass_rate"] == pytest.approx(1.0)
    assert report["gate"]["gpt-6-luna"]["sonnet_gate"]["pass_rate"] == pytest.approx(1.0)
    # 런타임 luna 최종 게이트 통과율도 함께 노출(비교 대조용) — validation_passed=True 전부.
    assert report["gate"]["qwen2.5-14b"]["runtime_luna_final_gate_pass_rate"] == pytest.approx(1.0)

    assert "qwen2.5-14b_vs_gpt-6-luna" in report["pairwise_overall_diff"]
    pw = report["pairwise_overall_diff"]["qwen2.5-14b_vs_gpt-6-luna"]
    assert pw["n"] == 2
    assert pw["mean_diff"] == pytest.approx(1.5)  # gpt-6-luna(5.0) - qwen(3.5)

    assert report["gate_judge_diff_by_model"]["qwen2.5-14b"]["n"] == 2
    assert report["gate_judge_diff_by_model"]["gpt-6-luna"]["mean_diff"] == pytest.approx(0.0)
    assert "qwen2.5-14b_vs_gpt-6-luna" in report["gate_judge_diff_contrast"]

    assert report["gate_judge_diff_note"]
    assert report["no_human_baseline_note"]

    # 문항 원문은 어디에도 없어야 한다(하드룰 4).
    assert "question" not in json.dumps(report, ensure_ascii=False)


def test_generator_pairs_includes_extra_pair_only_when_both_present_and_not_baseline():
    assert generator_pairs(["qwen2.5-14b", "gpt-6-luna", "gemini-3.8-flash"], "qwen2.5-14b") == [
        ("qwen2.5-14b", "gpt-6-luna"), ("qwen2.5-14b", "gemini-3.8-flash"), ("gpt-6-luna", "gemini-3.8-flash"),
    ]
    # 기준 모델이 extra pair 중 하나면 중복 추가하지 않는다.
    assert generator_pairs(["qwen2.5-14b", "gpt-6-luna", "gemini-3.8-flash"], "gpt-6-luna") == [
        ("gpt-6-luna", "qwen2.5-14b"), ("gpt-6-luna", "gemini-3.8-flash"),
    ]


def test_judge_discrimination_default_judge_model_follows_offline_env(monkeypatch):
    """judge/discrimination 서브커맨드의 --judge-model 기본값이 OFFLINE_JUDGE_MODEL
    (미설정 시 anthropic/claude-sonnet-5.5)을 따른다 — compare-generators·structure-judge와
    일관시킨 2026-10 결정(evals/eval_lib.py get_offline_judge_backend()와 같은 기본값).
    env는 모듈 import 시점에 한 번 읽히므로(importlib.reload로) 재평가해 확인한다."""
    import importlib

    import evals.eval_item_quality_runs as mod

    monkeypatch.delenv("OFFLINE_JUDGE_MODEL", raising=False)
    importlib.reload(mod)
    try:
        assert mod._OFFLINE_JUDGE_MODEL_DEFAULT == "anthropic/claude-sonnet-5.5"

        monkeypatch.setenv("OFFLINE_JUDGE_MODEL", "openai/gpt-6.1-sol")
        importlib.reload(mod)
        assert mod._OFFLINE_JUDGE_MODEL_DEFAULT == "openai/gpt-6.1-sol"
    finally:
        monkeypatch.delenv("OFFLINE_JUDGE_MODEL", raising=False)
        importlib.reload(mod)
