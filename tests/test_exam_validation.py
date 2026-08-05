"""출제 validate 노드가 재시도 피드백을 만드는지 검증한다."""
from app.modules.exam.graph import validate_node
from app.modules.exam.tools import init_session, record_score, save_item


def test_validate_reports_structure_failure_when_judge_missing():
    init_session("합성 예시", target_num=1)
    save_item.invoke({
        "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
        "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    state = {
        "spec": {"passage_text": "합성 예시", "num_items": 1},
        "similarity_judge_result": {},
    }

    result = validate_node(state)

    assert result["validation_passed"] is False
    assert "구조 유사도 미채점" in result["validation_feedback"]


def test_validate_passes_only_approved_complete_set():
    init_session("합성 예시", target_num=1)
    saved = save_item.invoke({
        "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
        "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    item_id = saved.split("item_id=")[1].split(")")[0]
    record_score.invoke({"item_id": item_id, "score": 4})
    state = {
        "spec": {"passage_text": "합성 예시", "num_items": 1},
        "similarity_judge_result": {
            "type_ratio_score": 0.8,
            "difficulty_match": True,
            "overall_score": 4,
        },
    }

    result = validate_node(state)

    assert result["validation_passed"] is True
    assert result["validation_feedback"] == ""


# ── 2026-08-04 게이트 임계값 재보정 회귀 방지 ────────────────────────────
# overall>=4·type_ratio>=0.7이던 옛 임계값은 실측 통과율 8.9%로 사실상 도달
# 불가였다(근거: graph.py 상단 주석, data/golden/_validate_gate_calibration.json).
# 아래 두 테스트는 새 경계값(overall>=3, type_ratio>=0.5)을 고정한다.


def _approved_single_item_state(judge_result: dict) -> dict:
    init_session("합성 예시", target_num=1)
    saved = save_item.invoke({
        "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
        "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    item_id = saved.split("item_id=")[1].split(")")[0]
    record_score.invoke({"item_id": item_id, "score": 4})
    return {
        "spec": {"passage_text": "합성 예시", "num_items": 1},
        "similarity_judge_result": judge_result,
    }


def test_validate_accepts_new_threshold_boundary():
    """overall=3·type_ratio=0.5는 새 기준의 정확한 경계 — 통과해야 한다.
    (옛 기준 overall>=4·type>=0.7에서는 둘 다 탈락했다.)"""
    state = _approved_single_item_state(
        {"type_ratio_score": 0.5, "difficulty_match": True, "overall_score": 3}
    )

    result = validate_node(state)

    assert result["validation_passed"] is True
    assert result["validation_feedback"] == ""


def test_validate_still_rejects_below_new_threshold():
    """경계 바로 아래(overall=2, type_ratio=0.2)는 여전히 막아야 한다 —
    임계값을 낮춘 것이지 게이트를 없앤 게 아니다."""
    state = _approved_single_item_state(
        {"type_ratio_score": 0.2, "difficulty_match": True, "overall_score": 2}
    )

    result = validate_node(state)

    assert result["validation_passed"] is False
    assert "유형 비율 유사도 미달" in result["validation_feedback"]
    assert "종합 구조 유사도 점수 미달" in result["validation_feedback"]


# ── 2026-08-05 자체 평가 점수를 게이트에서 제외 ──────────────────────────
# record_score는 문항을 생성한 에이전트 자신이 매기는 자체 평가라 사람 라벨과
# 대조된 적이 없다. 검증 안 된 지표가 통과 여부를 결정하지 않도록 게이트에서
# 뺐다(EVAL.md 15절). 점수 자체는 참고값으로 draft_items에 남는다.


def test_unscored_item_does_not_block_validation():
    """record_score 호출이 실패해 미채점으로 남아도 다른 조건이 충족되면 통과한다.
    (이전에는 미채점이 곧 rejected로 취급돼 all_approved가 깨지면서 반려됐다.)"""
    init_session("합성 예시", target_num=1)
    save_item.invoke({
        "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
        "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    state = {
        "spec": {"passage_text": "합성 예시", "num_items": 1},
        "similarity_judge_result": {
            "type_ratio_score": 1.0,
            "difficulty_match": True,
            "overall_score": 3,
        },
    }

    result = validate_node(state)

    assert result["validation_passed"] is True
    assert result["validation_feedback"] == ""


def test_low_self_score_does_not_block_validation():
    """자체 평가가 낮아도(1점) 게이트는 막지 않는다 — 판정은 별도 Judge가 한다."""
    init_session("합성 예시", target_num=1)
    saved = save_item.invoke({
        "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
        "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    item_id = saved.split("item_id=")[1].split(")")[0]
    record_score.invoke({"item_id": item_id, "score": 1})
    state = {
        "spec": {"passage_text": "합성 예시", "num_items": 1},
        "similarity_judge_result": {
            "type_ratio_score": 1.0,
            "difficulty_match": True,
            "overall_score": 3,
        },
    }

    result = validate_node(state)

    assert result["validation_passed"] is True


def test_draft_items_distinguishes_unscored_from_rejected():
    """미채점(unscored)과 저품질(rejected)은 원인도 대응도 달라 구분되어야 한다."""
    from app.modules.exam.tools import get_draft_items

    init_session("합성 예시", target_num=2)
    unscored_raw = save_item.invoke({
        "question": "민주주의의 기본 원리로 옳은 것은?",
        "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    scored_raw = save_item.invoke({
        "question": "법치주의가 보장하려는 가치로 가장 적절한 것은?",
        "options": ["① 자의적 지배 방지", "② 신분 세습", "③ 전제 정치", "④ 왕권 강화"],
        "answer": "①",
        "item_type": "객관식",
        "difficulty": "중",
        "standard": "",
    })
    scored_id = scored_raw.split("item_id=")[1].split(")")[0]
    record_score.invoke({"item_id": scored_id, "score": 1})

    by_id = {i["item_id"]: i for i in get_draft_items()}
    unscored_id = unscored_raw.split("item_id=")[1].split(")")[0]

    assert by_id[unscored_id]["status"] == "unscored"
    assert by_id[scored_id]["status"] == "rejected"
