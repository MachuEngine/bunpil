"""app/modules/record/chain.py — 규칙 기반 위반 탐지(_rule_violations) 순수 로직 유닛테스트.
LLM 호출 없음. VALIDATE_TPL(LLM 기반 보완 탐지)은 검증 범위 밖.
"""
from app.modules.record.chain import _rule_violations


def test_no_violation_on_clean_text():
    assert _rule_violations("독서 토론에서 근거 들어 주장함. 다른 의견도 잘 수용함.") == []


def test_detects_background_mention():
    violations = _rule_violations("한부모 가정에서 성장했으나 밝은 성격을 보임.")
    assert any("가정환경" in v for v in violations)


def test_detects_religion_politics_mention():
    violations = _rule_violations("종교적 신념이 뚜렷하고 정치적 견해를 자주 표현함.")
    assert any("종교·정치성향" in v for v in violations)


def test_detects_appearance_mention():
    violations = _rule_violations("외모가 준수하고 인상이 좋아 친구들에게 인기가 많음.")
    assert any("외모·신체" in v for v in violations)


def test_detects_comparison_expression():
    violations = _rule_violations("다른 학생보다 다소 느린 편이나 꾸준히 노력함.")
    assert any("비교·서열화" in v for v in violations)


def test_detects_pii_as_violation():
    violations = _rule_violations("김철수(010-1234-5678) 학생은 발표를 잘 함.")
    assert any("개인정보" in v for v in violations)
