"""출제 validate 노드가 재시도 피드백을 만드는지 검증한다."""
from app.modules.exam.graph import validate_node
from app.modules.exam.tools import init_session, record_score, save_item, similarity_judge


def test_validate_reports_unscored_item_and_structure_failure():
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
    assert "미채점" in result["validation_feedback"]
    assert "similarity_judge 미호출" in result["validation_feedback"]


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
    similarity_judge.invoke({
        "type_ratio_score": 0.8,
        "difficulty_match": True,
        "overall_score": 4,
    })
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
