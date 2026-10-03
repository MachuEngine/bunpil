"""evals/ci_report.py 순수 계산 함수 단위 테스트. API·LLM 호출 없음 — 전부 가짜
원자료 dict로 검증한다(실제 data/golden/_*.json 파일은 읽지 않음)."""
from evals.ci_report import (
    _mae_bias_agreement_ci,
    build_report,
    ragas_ci,
    structure_judge_eval_ci,
    validate_gate_calibration_ci,
)


def test_mae_bias_agreement_ci_computes_expected_point_values():
    out = _mae_bias_agreement_ci(
        judge_overall=[4, 2], human_overall=[5, 3],
        judge_diff=[True, False], human_diff=[True, True],
        ids=["a", "b"],
    )
    assert out["overall_score_mae"]["n"] == 2
    assert out["overall_score_mae"]["point"] == 1.0  # |4-5|=1, |2-3|=1 → 평균 1.0
    assert out["bias_judge_minus_human"]["point"] == -1.0  # (4-5)+(2-3) 평균
    assert out["difficulty_match_agreement"]["point"] == 0.5  # 1건 일치, 1건 불일치


def test_mae_bias_agreement_ci_bounds_contain_point():
    out = _mae_bias_agreement_ci(
        judge_overall=[4, 2, 3], human_overall=[5, 3, 3],
        judge_diff=[True, False, True], human_diff=[True, True, True],
        ids=["a", "b", "c"],
    )
    for key in ("overall_score_mae", "bias_judge_minus_human", "difficulty_match_agreement"):
        lo, hi = out[key]["ci"]
        assert lo <= out[key]["point"] <= hi


def test_structure_judge_eval_ci_reads_flat_per_item_schema():
    data = {
        "per_item": [
            {"id": "str_001", "human_overall": 5, "judge_overall": 4, "human_diff": True, "judge_diff": True},
            {"id": "str_002", "human_overall": 3, "judge_overall": 2, "human_diff": True, "judge_diff": False},
        ]
    }
    out = structure_judge_eval_ci(data)
    assert out["overall_score_mae"]["n"] == 2
    assert out["overall_score_mae"]["point"] == 1.0
    assert out["difficulty_match_agreement"]["point"] == 0.5


def test_validate_gate_calibration_ci_reads_nested_per_item_schema():
    data = {
        "per_item": [
            {"id": "x1", "judge": {"overall_score": 3, "difficulty_match": True},
             "human": {"overall_score": 3, "difficulty_match": True}},
            {"id": "x2", "judge": {"overall_score": 4, "difficulty_match": False},
             "human": {"overall_score": 2, "difficulty_match": False}},
        ]
    }
    out = validate_gate_calibration_ci(data)
    assert out["overall_score_mae"]["point"] == 1.0  # (0 + 2) / 2
    assert out["bias_judge_minus_human"]["point"] == 1.0  # (0 + 2) / 2
    assert out["difficulty_match_agreement"]["point"] == 1.0  # 둘 다 일치


def test_ragas_ci_computes_faithfulness_relevancy_and_item_quality():
    data = {
        "results": [
            {"id": "str_001", "faithfulness": {"score": 0.0}, "answer_relevancy": {"score": 0.3}},
            {"id": "str_002", "faithfulness": {"score": 1.0}, "answer_relevancy": {"score": 0.5}},
        ],
        "item_quality_scored": [
            {"item": {"item_id": "i1"}, "scores": {"정답유일성": 5, "오답매력도": 3, "근거성": 4, "overall": 4.0}},
            {"item": {"item_id": "i2"}, "scores": {"정답유일성": 3, "오답매력도": 3, "근거성": 4, "overall": 3.33}},
        ],
    }
    out = ragas_ci(data)
    assert out["faithfulness"]["n"] == 2
    assert out["faithfulness"]["point"] == 0.5
    assert out["answer_relevancy"]["point"] == 0.4
    assert out["item_quality_정답유일성"]["n"] == 2
    assert out["item_quality_정답유일성"]["point"] == 4.0


def test_build_report_marks_missing_files_as_no_raw_data(monkeypatch, tmp_path):
    """존재하지 않는 경로로 바꾸면 '원자료 없음' note가 붙어야 한다(API 호출 없음 확인용)."""
    import evals.ci_report as mod

    missing = str(tmp_path / "does_not_exist.json")
    monkeypatch.setattr(mod, "_RAGAS_PATH", missing)
    monkeypatch.setattr(mod, "_STRUCTURE_JUDGE_EVAL_PATH", missing)
    monkeypatch.setattr(mod, "_VALIDATE_GATE_CALIBRATION_PATH", missing)

    report = build_report()
    assert report["sources"]["ragas"]["ci"] is None
    assert "원자료 없음" in report["sources"]["ragas"]["note"]
    assert report["sources"]["vlm"]["ci"] is None
    assert len(report["no_raw_data"]) >= 1
