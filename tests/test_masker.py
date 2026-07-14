"""app/modules/record/masker.py — PII 마스킹 순수 로직 유닛테스트.
LLM 호출 없음. scripts/test_record.py의 MASK_CASES와 동일한 시나리오를 pytest로 검증.
"""
import pytest

from app.modules.record.masker import mask_pii


@pytest.mark.parametrize(
    "text,expected_pii,leaked",
    [
        ("김철수(010-1234-5678) 수학 시간에 발표 잘 함.", ["전화번호"], "010-"),
        ("900101-1234567 학생이 조별 과제에서 리더 역할.", ["주민번호"], "900101"),
        ("한국고등학교 2학년 학생, 이메일 student@school.kr로 자료 제출.", ["학교명", "이메일"], "@school.kr"),
    ],
)
def test_mask_pii_detects_and_removes(text, expected_pii, leaked):
    masked, found = mask_pii(text)
    assert set(found) == set(expected_pii)
    assert leaked not in masked


def test_mask_pii_no_false_positive_on_clean_text():
    text = "독서 토론에서 근거 들어 주장함. 다른 의견 수용함."
    masked, found = mask_pii(text)
    assert found == []
    assert masked == text
