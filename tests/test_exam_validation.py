"""출제 validate 노드가 통과/재시도를 판정하고 피드백을 만드는지 검증한다."""
from app.modules.exam.graph import validate_node
from app.modules.exam.tools import init_session, save_item

_ITEM = {
    "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
    "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
    "answer": "①",
    "item_type": "객관식",
    "difficulty": "중",
    "standard": "",
}


def _single_item_state(judge_result: dict) -> dict:
    """문항 1개를 저장한 뒤, 주어진 judge 결과로 validate 입력 state를 만든다."""
    init_session("합성 예시", target_num=1)
    save_item.invoke(_ITEM)
    return {
        "spec": {"passage_text": "합성 예시", "num_items": 1},
        "similarity_judge_result": judge_result,
    }


def test_validate_reports_structure_failure_when_judge_missing():
    state = _single_item_state({})

    result = validate_node(state)

    assert result["validation_passed"] is False
    assert "구조 유사도 미채점" in result["validation_feedback"]


def test_validate_passes_complete_set_with_good_judge_scores():
    state = _single_item_state(
        {"type_ratio_score": 0.8, "difficulty_match": True, "overall_score": 4}
    )

    result = validate_node(state)

    assert result["validation_passed"] is True
    assert result["validation_feedback"] == ""


def test_validate_reports_count_mismatch():
    """개수 미달은 여전히 게이트한다 — 이건 코드가 직접 세는 결정론적 조건."""
    init_session("합성 예시", target_num=2)
    save_item.invoke(_ITEM)
    state = {
        "spec": {"passage_text": "합성 예시", "num_items": 2},
        "similarity_judge_result": {
            "type_ratio_score": 1.0,
            "difficulty_match": True,
            "overall_score": 5,
        },
    }

    result = validate_node(state)

    assert result["validation_passed"] is False
    assert "문항 개수 불일치" in result["validation_feedback"]


# ── 2026-08-04 게이트 임계값 재보정 회귀 방지 ────────────────────────────
# overall>=4·type_ratio>=0.7이던 옛 임계값은 실측 통과율 6.7~8.9%로 사실상 도달
# 불가였다(근거: graph.py 상단 주석, data/golden/_validate_gate_calibration.json).
# 아래 두 테스트가 새 경계값(overall>=3, type_ratio>=0.5)을 고정한다.


def test_validate_accepts_new_threshold_boundary():
    """overall=3·type_ratio=0.5는 새 기준의 정확한 경계 — 통과해야 한다.
    (옛 기준 overall>=4·type>=0.7에서는 둘 다 탈락했다.)"""
    state = _single_item_state(
        {"type_ratio_score": 0.5, "difficulty_match": True, "overall_score": 3}
    )

    result = validate_node(state)

    assert result["validation_passed"] is True
    assert result["validation_feedback"] == ""


def test_validate_still_rejects_below_new_threshold():
    """경계 바로 아래(overall=2, type_ratio=0.2)는 여전히 막아야 한다 —
    임계값을 낮춘 것이지 게이트를 없앤 게 아니다."""
    state = _single_item_state(
        {"type_ratio_score": 0.2, "difficulty_match": True, "overall_score": 2}
    )

    result = validate_node(state)

    assert result["validation_passed"] is False
    assert "유형 비율 유사도 미달" in result["validation_feedback"]
    assert "종합 구조 유사도 점수 미달" in result["validation_feedback"]


# ── 2026-08-05/06 자기채점 제거 회귀 방지 ────────────────────────────────
# 문항 품질 점수를 생성 에이전트가 자기 자신에게 매기던 record_score는 사람 라벨과
# 대조된 적이 없어 게이트에서 빠졌고(08-05), 이후 도구 자체가 제거됐다(08-06).
# 판정은 별도 Judge(judge_node)의 구조 유사도만으로 이뤄져야 한다.


def test_validate_passes_without_any_self_assigned_score():
    """자기채점이 존재하지 않아도 Judge 조건만 충족하면 통과한다."""
    state = _single_item_state(
        {"type_ratio_score": 1.0, "difficulty_match": True, "overall_score": 3}
    )

    assert all("judge_score" not in i for i in validate_node(state)["draft_items"])
    assert validate_node(state)["validation_passed"] is True
