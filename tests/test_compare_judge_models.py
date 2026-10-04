"""experiments/compare_judge_models.py — Judge 1차 선별 보강분 단위 테스트.

LLM 호출 없음 — JUDGE_ENVS 설정값과, 가짜 score_structure() 반환 모양으로 만든
per-item row에 대한 회차 평균·부트스트랩 CI 계산 경로만 검증한다. 실제 모듈 import는
eval_lib.py(ITEM_GOLDEN 등 JSON 로드)까지 거치지만 네트워크·LLM 호출은 없다.
"""
from experiments.compare_judge_models import (
    JUDGE_ENVS,
    _average_repeats,
    _estimate_cost_usd,
    _structure_ci,
    _structure_item_rows,
)


def test_openrouter_judge_candidates_have_openrouter_judge_model():
    """or- 접두 후보는 모두 JUDGE_BACKEND=openrouter + OPENROUTER_JUDGE_MODEL을 갖는다 —
    factory.py의 get_judge_backend()가 이 값이 없으면 바로 실패하기 때문(생성 모델로
    조용히 폴백하지 않음)."""
    or_candidates = [k for k in JUDGE_ENVS if k.startswith("or-")]
    assert len(or_candidates) == 5
    for key in or_candidates:
        env = JUDGE_ENVS[key]
        assert env["JUDGE_BACKEND"] == "openrouter"
        assert env.get("OPENROUTER_JUDGE_MODEL"), f"{key}에 OPENROUTER_JUDGE_MODEL이 없음"


def test_existing_judge_candidates_unchanged():
    """기존 4개 후보(qwen·openai 직접)는 그대로 남아있어야 한다."""
    for key in ("qwen2.5-7b", "qwen2.5-14b", "gpt-5.6-luna", "gpt-5.6-sol"):
        assert key in JUDGE_ENVS


def _fake_scored(pairs: list[tuple]) -> list[dict]:
    """pairs: (entry_id, judge_overall, judge_diff_match, human_overall, human_diff_match)."""
    scored = []
    for entry_id, j_overall, j_diff, h_overall, h_diff in pairs:
        scored.append({
            "entry": {
                "id": entry_id,
                "human_label": {"overall_score": h_overall, "difficulty_match": h_diff},
            },
            "judge": {
                "overall_score": j_overall,
                "difficulty_match": j_diff,
                "type_ratio_score": 1.0,
            },
        })
    return scored


def test_structure_item_rows_extracts_judge_and_human_pairs():
    scored = _fake_scored([("str_001", 4, True, 5, True), ("str_002", 2, False, 3, True)])
    rows = _structure_item_rows(scored)
    assert rows[0]["id"] == "str_001"
    assert rows[0]["judge_overall"] == 4
    assert rows[0]["human_overall"] == 5
    assert rows[0]["likely_parse_failed"] is False
    assert rows[1]["judge_difficulty_match"] is False
    assert rows[1]["human_difficulty_match"] is True


def test_structure_item_rows_flags_likely_parse_failure():
    """type_ratio_score=0.0·overall_score=0·difficulty_match=False가 동시에 나오면
    파싱 실패로 추정한다(진짜 0점과 구분 불가 — 근사치)."""
    scored = _fake_scored([("str_001", 0, False, 3, True)])
    scored[0]["judge"]["type_ratio_score"] = 0.0
    rows = _structure_item_rows(scored)
    assert rows[0]["likely_parse_failed"] is True


def test_average_repeats_averages_metric_per_item():
    # str_001: 1회차 judge=5(human=5, diff=0), 2회차 judge=3(human=5, diff=-2) → mae 평균=1.0, bias=-1.0
    rep1 = _structure_item_rows(_fake_scored([("str_001", 5, True, 5, True)]))
    rep2 = _structure_item_rows(_fake_scored([("str_001", 3, False, 5, True)]))
    averaged = _average_repeats([rep1, rep2])
    assert len(averaged) == 1
    assert averaged[0]["id"] == "str_001"
    assert averaged[0]["mae"] == 1.0
    assert averaged[0]["bias"] == -1.0
    # difficulty_match: 1회차 True==True(hit), 2회차 False==True(miss) → 평균 0.5
    assert averaged[0]["difficulty_hit"] == 0.5


def test_structure_ci_reproducible_and_bounded():
    rows = [
        {"id": "str_001", "mae": 0.0, "bias": 0.0, "difficulty_hit": 1.0},
        {"id": "str_002", "mae": 2.0, "bias": -2.0, "difficulty_hit": 0.0},
        {"id": "str_003", "mae": 1.0, "bias": -1.0, "difficulty_hit": 1.0},
    ]
    ci1 = _structure_ci(rows, n_boot=200, seed=0)
    ci2 = _structure_ci(rows, n_boot=200, seed=0)
    assert ci1 == ci2  # 같은 seed면 재현 가능

    mae = ci1["overall_mae"]
    assert mae["point"] == 1.0
    assert mae["ci_lo"] <= mae["point"] <= mae["ci_hi"]

    dma = ci1["difficulty_match_agreement"]
    assert dma["point"] == round(2 / 3, 3)  # _structure_ci가 소수 3자리로 반올림


def test_estimate_cost_none_for_candidates_without_price_table():
    assert _estimate_cost_usd("qwen2.5-7b", n_structure_calls=45, n_item_calls=0) is None
    assert _estimate_cost_usd("gpt-5.6-luna", n_structure_calls=45, n_item_calls=0) is None


def test_estimate_cost_scales_with_calls_for_openrouter_candidate():
    cost_45 = _estimate_cost_usd("or-gpt-5.6-luna", n_structure_calls=45, n_item_calls=0)
    cost_90 = _estimate_cost_usd("or-gpt-5.6-luna", n_structure_calls=90, n_item_calls=0)
    assert cost_45 is not None
    assert cost_90 == cost_45 * 2
