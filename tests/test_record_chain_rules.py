"""app/modules/record/chain.py — 규칙 기반 탐지 순수 로직 유닛테스트.
LLM 호출 없음. VALIDATE_TPL(LLM 기반 보완 탐지)은 검증 범위 밖.

2026-08-03: 키워드 규칙이 `_rule_violations`(차단) → `_rule_warnings`(경고)로 바뀌고,
PII만 `_pii_violations`(차단)로 분리됐다. 탐지 자체는 그대로이고 결과 처리만 달라진다.
"""
from app.modules.record.chain import _pii_violations, _rule_warnings


def test_no_warning_on_clean_text():
    assert _rule_warnings("독서 토론에서 근거 들어 주장함. 다른 의견도 잘 수용함.") == []


def test_detects_background_mention():
    warnings = _rule_warnings("한부모 가정에서 성장했으나 밝은 성격을 보임.")
    assert any("가정환경" in w for w in warnings)


def test_detects_religion_politics_mention():
    warnings = _rule_warnings("종교적 신념이 뚜렷하고 정치적 견해를 자주 표현함.")
    assert any("종교·정치성향" in w for w in warnings)


def test_detects_appearance_mention():
    warnings = _rule_warnings("외모가 준수하고 인상이 좋아 친구들에게 인기가 많음.")
    assert any("외모·신체" in w for w in warnings)


def test_detects_comparison_expression():
    warnings = _rule_warnings("다른 학생보다 다소 느린 편이나 꾸준히 노력함.")
    assert any("비교·서열화" in w for w in warnings)


def test_keyword_rules_are_warnings_not_violations():
    """키워드 규칙은 전부 WARNING 접두사여야 한다 — 차단이 아니라 경고이므로."""
    warnings = _rule_warnings("한부모 가정에서 성장했고 외모가 준수함.")
    assert warnings
    assert all(w.startswith("WARNING:") for w in warnings)


def test_keyword_rules_do_not_flag_pii():
    """PII는 키워드 규칙이 아니라 _pii_violations 담당 — 역할이 섞이면 안 됨."""
    assert _rule_warnings("김철수(010-1234-5678) 학생은 발표를 잘 함.") == []


def test_detects_pii_as_violation():
    """PII는 경고가 아니라 차단(하드룰 2·4) — VIOLATION 접두사 유지."""
    violations = _pii_violations("김철수(010-1234-5678) 학생은 발표를 잘 함.")
    assert any("개인정보" in v for v in violations)
    assert all(v.startswith("VIOLATION:") for v in violations)


def test_no_pii_violation_on_clean_text():
    assert _pii_violations("독서 토론에서 근거 들어 주장함.") == []
