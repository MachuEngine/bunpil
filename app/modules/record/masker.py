"""PII 마스킹 — 모델 호출 이전에 적용 (보안 하드룰).
감지 대상: 전화번호, 주민번호, 이메일, 학교명 패턴.
"""
import re
from typing import Tuple, List

_PHONE = re.compile(r"\d{2,3}-\d{3,4}-\d{4}")
_JUMIN = re.compile(r"\d{6}-[1-4]\d{6}")
_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# "XX고등학교", "XX중학교", "XX초등학교" (약칭 "XX고" 등은 오탐 위험으로 제외)
_SCHOOL = re.compile(r"[가-힣]{1,8}(고등학교|중학교|초등학교)")
# 숫자 학번: 5-10자리 순수 숫자 (주민번호 미포함)
_STUDENT_ID = re.compile(r"(?<!\d)\d{5,10}(?!\d)")

_RULES: List[Tuple[re.Pattern, str, str]] = [
    (_JUMIN,      "주민번호",  "[주민번호]"),
    (_PHONE,      "전화번호",  "[연락처]"),
    (_EMAIL,      "이메일",    "[이메일]"),
    (_SCHOOL,     "학교명",    "[학교]"),
    (_STUDENT_ID, "학번",      "[학번]"),
]


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """PII를 마스킹한 텍스트와 발견된 PII 유형 목록을 반환."""
    found: List[str] = []
    masked = text
    for pattern, label, placeholder in _RULES:
        if pattern.search(masked):
            found.append(label)
            masked = pattern.sub(placeholder, masked)
    return masked, found
