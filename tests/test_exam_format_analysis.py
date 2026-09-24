"""요청 분석(문항 수 + 예시 형식)과 형식 판정 단일화 테스트 — 2026-09-24. 모델 호출 없음."""
import asyncio
import json

import app.common.llm
from app.main import _build_spec, _parse_analysis
from app.modules.exam.graph import _build_system_prompt
from app.modules.exam.tools import _get_ctx, init_session, rule_format

FOUR = "1. 옳은 것은?\n① 가 ② 나 ③ 다 ④ 라"


def _analysis(**over) -> str:
    base = {"requested_num_items": 3, "num_options": 4, "has_stimulus": False, "combo": False, "has_essay": False,
            "unsupported": [], "nearest": ""}
    return json.dumps({**base, **over}, ensure_ascii=False)


# ── 응답 파싱 ────────────────────────────────────────────────────────

def test_valid_analysis_is_used():
    r = _parse_analysis(_analysis(requested_num_items=5, num_options=5, has_stimulus=True), FOUR)
    assert r["num_items"] == 5
    assert r["format"] == {"num_options": 5, "has_stimulus": True, "combo": False, "has_essay": False}
    assert r["format_notice"] == "" and r["format_instruction"] == ""


def test_no_explicit_request_defaults_to_two():
    assert _parse_analysis(_analysis(requested_num_items=None), FOUR)["num_items"] == 2


def test_digits_are_not_concatenated():
    """이전 구현은 '3~5문제'의 숫자를 이어 붙여 35(→20)로 읽었다."""
    assert _parse_analysis("3~5문제 정도요", FOUR)["num_items"] == 2
    assert _parse_analysis("7", FOUR)["num_items"] == 7  # 숫자 하나뿐인 응답은 하위 호환


def test_invalid_fields_fall_back_to_rules_per_field():
    raw = _analysis(requested_num_items=99, num_options=6, has_stimulus="yes", combo=None)
    r = _parse_analysis(raw, "1. 다음 <보기>에서 고른 것은?\n<보기>\nㄱ. 가\nㄴ. 나\n① ㄱ ② ㄴ ③ ㄱ, ㄴ ④ 없음 ⑤ ㄱ, ㄴ, ㄷ")
    assert r["num_items"] == 2                       # 범위 밖 → 기본값
    assert r["format"]["num_options"] == 5           # 6 → 규칙(선지 줄의 ⑤)
    assert r["format"]["has_stimulus"] is True       # "yes"(bool 아님) → 규칙(<보기>)


def test_broken_response_uses_rule_format():
    text = "1. (가), (나) 국가에 대한 설명으로 옳은 것은?\n(가) 대통령제\n(나) 의원 내각제\n① 가 ② 나 ③ 다 ④ 라"
    r = _parse_analysis("죄송합니다", text)
    assert r["format"] == rule_format(text) and r["num_items"] == 2


def test_combo_implies_stimulus():
    assert _parse_analysis(_analysis(combo=True, has_stimulus=False), FOUR)["format"]["has_stimulus"] is True


def test_unsupported_format_produces_notice_and_instruction():
    r = _parse_analysis(_analysis(unsupported=["OX"], nearest="4지 선다"), FOUR)
    assert "OX" in r["format_notice"] and "4지 선다" in r["format_notice"]
    assert "OX" in r["format_instruction"]
    # 지원 형식이 아닌 nearest는 4지 선다로 대체
    assert "4지 선다" in _parse_analysis(_analysis(unsupported=["OX"], nearest="OX 문제"), FOUR)["format_notice"]


# ── 규칙 오판 수정 ────────────────────────────────────────────────────

def test_circled_five_in_body_is_not_five_options():
    assert rule_format("1. 다음 자료의 ⑤번 항목에 해당하는 권리는?\n① 자유권 ② 평등권 ③ 참정권 ④ 청구권")["num_options"] == 4
    assert rule_format("1. 옳은 것은?\n① 가 ② 나 ③ 다 ④ 라 ⑤ 마")["num_options"] == 5


def test_gana_text_material_requires_stimulus():
    text = "1. (가), (나) 국가에 대한 설명으로 옳은 것은?\n(가) 국가는 대통령제이다.\n(나) 국가는 의원 내각제이다.\n① 가 ② 나 ③ 다 ④ 라"
    assert rule_format(text)["has_stimulus"] is True
    # 선지 안의 (가)-(나)는 제시문이 아니다(줄 머리가 아님)
    assert rule_format("1. 순서로 옳은 것은?\n① (가)-(나) ② (나)-(가) ③ 가 ④ 나")["has_stimulus"] is False


# ── 판정 단일화: 게이트와 프롬프트가 같은 dict를 본다 ──────────────────

def test_gate_and_prompt_use_same_format():
    fmt = {"num_options": 5, "has_stimulus": False, "combo": False, "has_essay": False}
    init_session(FOUR, 1, fmt)  # 원문은 4지지만 분석 결과(5지)를 따른다
    assert _get_ctx()["num_options"] == 5
    prompt = _build_system_prompt(FOUR, 1, [], "", fmt)
    assert "①②③④⑤ 형식으로 5개" in prompt


def test_combo_instruction_is_outside_stimulus_branch():
    fmt = {"num_options": 4, "has_stimulus": False, "combo": True, "has_essay": False}
    assert "합답형" in _build_system_prompt(FOUR, 1, [], "", fmt)


def test_unsupported_instruction_reaches_prompt():
    prompt = _build_system_prompt(FOUR, 1, [], "", rule_format(FOUR), "예시 중 OX 형식은 지원하지 않으므로 4지 선다로 작성하세요.")
    assert "OX 형식은 지원하지 않으므로" in prompt


# ── 마스킹 순서: 분석 호출 입력에 원본 PII가 없다 ─────────────────────

class _RecordingBackend:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self.reply


def test_analysis_call_sees_only_masked_text(monkeypatch):
    backend = _RecordingBackend(_analysis(unsupported=["OX"], nearest="4지 선다"))
    monkeypatch.setattr(app.common.llm, "get_llm_backend", lambda: backend)

    spec, _, pii = asyncio.run(_build_spec("김철수 학생(010-1234-5678) 오답이에요.\n" + FOUR))

    assert len(backend.calls) == 1  # 문항 수 + 형식을 한 번에 — 호출 수는 늘지 않았다
    sent = json.dumps(backend.calls[0], ensure_ascii=False)
    assert "김철수" not in sent and "010-1234-5678" not in sent
    assert spec["num_items"] == 3 and "OX" in spec["format_notice"] and set(pii) == {"이름", "전화번호"}


def test_unsupported_as_single_string_still_notifies():
    r = _parse_analysis(_analysis(unsupported="OX", nearest="4지 선다"), FOUR)
    assert "OX" in r["format_notice"]
