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
    compute_metrics,
    compute_reliability_report,
    coverage_passage_ids,
    group_judged_by_run,
    human_label_distribution,
    judge_avg_rounded,
    judge_avg_score,
    judged_keys,
    load_inputs_meta,
    load_labeled_keys,
    mc_items_for_run,
    paired_abs_error_diff,
    per_repeat_kappas,
    repeats_for_item,
    score_distribution,
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
