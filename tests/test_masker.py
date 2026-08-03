"""app/common/privacy.py — PII 마스킹 순수 로직 유닛테스트. LLM 호출 없음.

mask_pii()는 출제 경로가 쓴다 — `app/main.py`의 `_build_spec()`이 `passage_text`를
모델에 넘기기 전에 호출한다(하드룰 2: 마스킹은 외부/모델 호출 이전).
"""
import json
from pathlib import Path

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


# ── MASKING_GOLDEN 20건 (PII 10 + 정상 10) ──────────────────────────────
# 2026-08-03: 생기부 모듈과 함께 evals/eval_record.py가 제거되면서 이 골든셋을
# 채점하던 코드가 사라졌다. 그런데 mask_pii()는 출제 경로가 계속 쓰므로
# (app/main.py `_build_spec` — passage_text를 모델에 넘기기 전 마스킹, 하드룰 2)
# 커버리지를 잃지 않도록 골든셋을 이 유닛테스트로 흡수했다.
# 원 지표였던 "누락률(FN) = 0"은 아래 파라미터화 테스트가 그대로 강제한다.

_MASKING_GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "golden" / "masking_golden.json"
)


def _load_masking_golden() -> list[dict]:
    with open(_MASKING_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)["entries"]


@pytest.mark.parametrize("item", _load_masking_golden(), ids=lambda i: i["text"][:20])
def test_masking_golden_has_no_false_negative(item):
    """골든셋이 지정한 PII 유형을 하나도 놓치지 않아야 한다(FN=0)."""
    _, found = mask_pii(item["text"])
    missing = set(item["pii"]) - set(found)
    assert not missing, f"놓친 PII 유형: {missing}"
