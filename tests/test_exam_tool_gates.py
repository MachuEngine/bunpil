"""출제 저장·점수·교체 게이트의 결정론적 동작 테스트."""
from app.modules.exam.tools import (
    discard_item,
    get_draft_items,
    init_session,
    record_score,
    save_item,
)


VALID_ITEM = {
    "question": "다음 중 민주주의 원리에 대한 설명으로 옳은 것은?",
    "options": ["① 국민 주권", "② 왕권 신수설", "③ 신분 세습", "④ 전제 정치"],
    "answer": "①",
    "item_type": "객관식",
    "difficulty": "중",
    "standard": "",
}


def test_save_rejects_malformed_item_even_without_validate_call():
    init_session("합성 예시", target_num=2)
    result = save_item.invoke({
        **VALID_ITEM,
        "options": ["보기 하나"],
        "answer": "⑤",
        "difficulty": "최상",
    })

    assert result.startswith("저장 거부 — 형식 오류")
    assert get_draft_items() == []


def test_score_requires_explicit_existing_item_id():
    init_session("합성 예시", target_num=2)
    saved = save_item.invoke(VALID_ITEM)
    item_id = saved.split("item_id=")[1].split(")")[0]

    rejected = record_score.invoke({"item_id": "missing", "score": 5})
    accepted = record_score.invoke({"item_id": item_id, "score": 4})

    assert rejected.startswith("점수 기록 거부")
    assert accepted == "품질 점수 4/5 기록됨"
    assert get_draft_items()[0]["status"] == "approved"


def test_target_cap_requires_discard_before_replacement():
    init_session("합성 예시", target_num=1)
    saved = save_item.invoke(VALID_ITEM)
    item_id = saved.split("item_id=")[1].split(")")[0]

    capped = save_item.invoke({**VALID_ITEM, "question": "시장 실패의 원인에 대한 설명으로 옳은 것은?"})
    discarded = discard_item.invoke({"item_id": item_id})
    replacement = save_item.invoke({
        **VALID_ITEM,
        "question": "시장 실패의 원인에 대한 설명으로 옳은 것은?",
    })

    assert capped.startswith("저장 거부 — 목표 문항 수")
    assert discarded.startswith("문항 폐기 완료")
    assert replacement.startswith("저장 완료")
