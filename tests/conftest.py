"""pytest 세션 전체에 적용되는 공용 설정.

2026-07-24 출제 모듈 프로덕션 트레이싱 허용(CLAUDE.md 하드룰 3 예외) 이후, app.main이
LANGCHAIN_TRACING_V2를 더 이상 강제로 끄지 않는다. 그대로 두면 테스트가 개발자 로컬 .env의
값에 영향받아 순수 로직 테스트에서도 실제 LangSmith 네트워크 호출을 시도할 수 있어(테스트
비결정성·속도 저하·API 사용량 소모), 테스트 세션 동안은 항상 꺼둔다.
"""
import os

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture(autouse=True)
def _reset_exam_request_ctx():
    """app/modules/exam/tools.py의 `_request_ctx`(contextvars)를 매 테스트 전 비운다.

    2026-08-04: 이전엔 이 격리가 "모든 exam 도구 테스트가 init_session()을 먼저
    호출한다"는 관례에만 의존했다 — pytest는 같은 프로세스·스레드에서 테스트를
    순차 실행하므로, 어떤 테스트가 init_session()을 빠뜨리면 예외 없이 직전 테스트가
    남긴 상태를 조용히 이어받을 수 있었다(구조적 보장이 아니라 관례였다는 뜻).
    이제 매 테스트가 빈 dict에서 시작하므로, init_session()을 빠뜨린 테스트는
    KeyError로 즉시 드러난다.
    """
    from app.modules.exam.tools import _request_ctx

    _request_ctx.set({})
    yield
