"""출제 저장·교체 게이트의 결정론적 동작 테스트."""
from app.modules.exam.tools import (
    TOOLS,
    discard_item,
    get_draft_items,
    init_session,
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


def test_saved_item_exposes_no_self_assigned_score():
    """2026-08-06: record_score 제거 — 저장된 문항에 자기채점 흔적이 남으면 안 된다.

    도구 자체가 사라졌는지도 함께 확인한다(에이전트에게 bind_tools로 노출되는 목록).
    """
    init_session("합성 예시", target_num=2)
    save_item.invoke(VALID_ITEM)

    item = get_draft_items()[0]

    assert "judge_score" not in item
    assert "status" not in item
    assert "record_score" not in {t.name for t in TOOLS}


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


# ── 2026-08-06: Judge 입력 payload 통일 회귀 방지 ────────────────────────
# 런타임(get_draft_items)과 오프라인 eval(골든셋 generated_items)이 Judge에게
# 서로 다른 필드를 넘기고 있었다 — EVAL.md의 Judge 신뢰도 수치가 프로덕션이
# 실제로 보내지 않는 입력으로 측정된 값이었다는 뜻(EVAL.md 17절).


def test_judge_payload_is_identical_for_runtime_and_offline():
    """런타임 문항과 골든셋 문항이 Judge에게 동일한 모양으로 정규화되는지 확인."""
    from app.modules.exam.judge import _JUDGE_ITEM_FIELDS, _to_judge_payload

    init_session("합성 예시", target_num=1)
    save_item.invoke(VALID_ITEM)
    runtime_item = get_draft_items()[0]
    # 골든셋(structure_golden.json) generated_items의 실제 필드 구성
    golden_item = {k: VALID_ITEM[k] for k in _JUDGE_ITEM_FIELDS}

    runtime_payload = _to_judge_payload([runtime_item])[0]
    golden_payload = _to_judge_payload([golden_item])[0]

    assert runtime_payload.keys() == golden_payload.keys()
    assert runtime_payload == golden_payload
    # item_id·standard 같은 런타임 전용 필드는 Judge에게 새어나가면 안 된다
    assert "item_id" in runtime_item and "item_id" not in runtime_payload
    assert "standard" in runtime_item and "standard" not in runtime_payload


# ── 2026-09: 형식 인식 — 5지선다·<보기>(stimulus) ────────────────────────

FIVE_OPTION_PASSAGE = (
    "1. 다음 자료에 대한 설명으로 옳은 것은?\n<보기>\nㄱ. 합성 진술 하나\nㄴ. 합성 진술 둘\n"
    "① ㄱ ② ㄴ ③ ㄱ, ㄴ ④ ㄴ, ㄷ ⑤ ㄱ, ㄴ, ㄷ"
)
FIVE_OPTION_ITEM = {
    **VALID_ITEM,
    "question": "<보기>에서 시장 실패의 사례로 옳은 것만을 있는 대로 고른 것은?",
    "stimulus": "<보기>\nㄱ. 공공재의 무임승차 문제\nㄴ. 완전 경쟁 시장의 가격 결정\nㄷ. 공장 매연으로 인한 외부 불경제",
    "options": ["① ㄱ", "② ㄴ", "③ ㄱ, ㄷ", "④ ㄴ, ㄷ", "⑤ ㄱ, ㄴ, ㄷ"],
    "answer": "③",
}


def test_five_option_example_accepts_five_options_and_stimulus():
    init_session(FIVE_OPTION_PASSAGE, target_num=1)
    result = save_item.invoke(FIVE_OPTION_ITEM)

    assert result.startswith("저장 완료")
    assert get_draft_items()[0]["stimulus"] == FIVE_OPTION_ITEM["stimulus"]


def test_four_option_example_rejects_five_options():
    init_session("합성 예시", target_num=1)
    result = save_item.invoke(FIVE_OPTION_ITEM)

    assert result.startswith("저장 거부 — 형식 오류")
    assert "선지는 4개" in result


def test_five_option_example_rejects_four_options():
    init_session(FIVE_OPTION_PASSAGE, target_num=1)
    result = save_item.invoke(VALID_ITEM)

    assert result.startswith("저장 거부 — 형식 오류")
    assert "선지는 5개" in result


def test_stimulus_copied_from_example_is_rejected():
    stimulus = "<보기>\nㄱ. 공공재의 무임승차 문제가 발생한다\nㄴ. 외부 불경제로 사회적 비용이 커진다"
    init_session(f"1. 다음 설명으로 옳은 것은?\n{stimulus}\n① ㄱ ② ㄴ ③ ㄱ, ㄴ ④ ㄴ", target_num=1)
    result = save_item.invoke({
        **VALID_ITEM,
        "question": "<보기>에서 시장 실패와 관련된 진술로 옳은 것을 고른 것은?",
        "stimulus": stimulus,
        "options": ["① ㄱ", "② ㄴ", "③ ㄱ, ㄴ", "④ ㄴ, ㄱ"],
    })

    assert result.startswith("저장 거부 — <보기>·자료(stimulus)가 예시 문제를")
    assert get_draft_items() == []


def test_judge_payload_includes_stimulus_only_when_present():
    """제시문이 없는 문항은 Judge 입력이 기존과 같아야 한다(기존 골든셋 신뢰도 유지)."""
    from app.modules.exam.judge import _to_judge_payload

    without = _to_judge_payload([{**VALID_ITEM, "stimulus": ""}])[0]
    with_stimulus = _to_judge_payload([FIVE_OPTION_ITEM])[0]

    assert "stimulus" not in without
    assert with_stimulus["stimulus"] == FIVE_OPTION_ITEM["stimulus"]


def test_bogi_example_requires_stimulus_for_multiple_choice():
    init_session(FIVE_OPTION_PASSAGE, target_num=1)
    result = save_item.invoke({**FIVE_OPTION_ITEM, "stimulus": ""})

    assert result.startswith("저장 거부 — 형식 오류")
    assert "stimulus" in result


def test_bogi_label_without_content_is_rejected():
    init_session(FIVE_OPTION_PASSAGE, target_num=1)
    result = save_item.invoke({**FIVE_OPTION_ITEM, "stimulus": "<보기>"})

    assert result.startswith("저장 거부 — 형식 오류")


def test_combo_example_rejects_sentence_options():
    init_session(FIVE_OPTION_PASSAGE, target_num=1)
    result = save_item.invoke({
        **FIVE_OPTION_ITEM,
        "options": ["① 공공재 문제", "② 가격 결정", "③ 외부 불경제", "④ 독점", "⑤ 정보 비대칭"],
    })

    assert result.startswith("저장 거부 — 형식 오류")
    assert "합답형" in result


def test_non_combo_example_allows_sentence_options():
    init_session("1. 옳은 것은?\n① 국민 주권 ② 권력 분립 ③ 기본권 ④ 계획 경제", target_num=1)
    assert save_item.invoke(VALID_ITEM).startswith("저장 완료")


def test_combo_option_letters_must_exist_in_stimulus():
    init_session(FIVE_OPTION_PASSAGE, target_num=1)
    result = save_item.invoke({
        **FIVE_OPTION_ITEM,
        "options": ["① ㄱ", "② ㄴ", "③ ㄱ, ㄷ", "④ ㄴ, ㄹ", "⑤ ㄱ, ㄴ, ㄷ"],  # <보기>에 ㄹ 없음
    })

    assert result.startswith("저장 거부 — 형식 오류")
    assert "ㄹ" in result
