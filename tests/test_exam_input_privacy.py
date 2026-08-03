"""출제 입력이 어떤 LLM 호출보다 먼저 마스킹되는지 검증한다."""
import asyncio

import app.common.llm
from app.main import _build_spec


class _FakeBackend:
    def __init__(self):
        self.messages = []

    async def generate(self, messages, **kwargs):
        self.messages = messages
        return "5"


def test_build_spec_masks_pii_before_num_items_llm(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(app.common.llm, "get_llm_backend", lambda: backend)

    spec, truncated, pii_found = asyncio.run(
        _build_spec("김철수 학생 연락처 01012345678, 3문제 만들어줘.")
    )

    model_input = backend.messages[-1]["content"]
    assert "김철수" not in model_input
    assert "01012345678" not in model_input
    assert spec["passage_text"] == model_input
    assert set(pii_found) == {"이름", "전화번호"}
    assert truncated is False


def test_plain_backends_are_not_langchain_traceable():
    """`LLMBackend` 구현체들이 LangChain Runnable이 아님을 확인한다.

    이 백엔드들은 `_extract_num_items()` 같은 비-에이전트 경로가 쓰는데, LangChain
    Runnable이 아니라 순수 클래스이므로 `LANGCHAIN_TRACING_V2` 값과 무관하게 애초에
    LangSmith 콜백에 걸리지 않는다 — 트레이싱 범위가 의도치 않게 넓어지지 않는다는
    구조적 보장이다(하드룰 3, CLAUDE.md 참고).

    2026-08-03: 원래 생기부 모듈이 이 보장의 주 수혜자였으나 모듈 제거 후에도
    이 성질 자체는 유지되어야 하므로 테스트를 남긴다."""
    from langchain_core.runnables import Runnable

    from app.common.llm.backends.ollama import OllamaBackend
    from app.common.llm.backends.openai import OpenAIBackend
    from app.common.llm.backends.runpod import RunPodBackend

    for backend_cls in (OllamaBackend, OpenAIBackend, RunPodBackend):
        assert not issubclass(backend_cls, Runnable)
