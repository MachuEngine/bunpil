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


def test_api_server_forces_runtime_tracing_off():
    import os

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"
