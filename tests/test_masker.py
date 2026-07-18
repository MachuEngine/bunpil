"""app/common/privacy.py — PII 마스킹 순수 로직 유닛테스트.
LLM 호출 없음. scripts/test_record.py의 MASK_CASES와 동일한 시나리오를 pytest로 검증.
"""
import pytest

from app.common.privacy import mask_pii


@pytest.mark.parametrize(
    "text,expected_pii,leaked",
    [
        ("김철수(010-1234-5678) 수학 시간에 발표 잘 함.", ["전화번호", "이름"], ["김철수", "010-"]),
        ("김영희 학생 연락처는 01012345678임.", ["전화번호", "이름"], ["김영희", "01012345678"]),
        ("900101-1234567 학생이 조별 과제에서 리더 역할.", ["주민번호"], "900101"),
        ("한국고등학교 2학년 학생, 이메일 student@school.kr로 자료 제출.", ["학교명", "이메일"], "@school.kr"),
        ("학생 이름: 박민수, 거주지: 서울특별시 종로구 사직로 1", ["주소", "이름"], ["박민수", "사직로"]),
        ("서울특별시 종로구에 거주하며 토론에 참여함.", ["주소"], "서울특별시"),
    ],
)
def test_mask_pii_detects_and_removes(text, expected_pii, leaked):
    masked, found = mask_pii(text)
    assert set(found) == set(expected_pii)
    leaked_values = leaked if isinstance(leaked, list) else [leaked]
    assert all(value not in masked for value in leaked_values)


@pytest.mark.parametrize(
    "text",
    [
        "독서 토론에서 근거 들어 주장함. 다른 의견 수용함.",
        "다른 학생의 의견을 존중함.",
        "2학년 학생이 발표에 참여함.",
    ],
)
def test_mask_pii_no_false_positive_on_clean_text(text):
    masked, found = mask_pii(text)
    assert found == []
    assert masked == text
