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


def test_record_backends_are_not_langchain_traceable():
    """생기부 모듈(record/chain.py)이 쓰는 백엔드가 LangChain Runnable이 아님을 확인한다 —
    2026-07-24부터 출제 모듈은 LANGCHAIN_TRACING_V2=true 시 프로덕션에서도 트레이싱되지만
    (하드룰 3 예외, CLAUDE.md 참고), 생기부는 이 값과 무관하게 트레이싱되면 안 된다.
    이 구조(순수 클래스, LangChain Runnable 미상속)가 그 보장의 근거다."""
    from langchain_core.runnables import Runnable

    from app.common.llm.backends.ollama import OllamaBackend
    from app.common.llm.backends.openai import OpenAIBackend
    from app.common.llm.backends.runpod import RunPodBackend

    for backend_cls in (OllamaBackend, OpenAIBackend, RunPodBackend):
        assert not issubclass(backend_cls, Runnable)
