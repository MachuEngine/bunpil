"""출제 입력이 어떤 LLM 호출보다 먼저 마스킹되는지 검증한다."""
import asyncio

import app.common.llm
from app.main import _build_spec, app as fastapi_app


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
    이 성질 자체는 유지되어야 하므로 테스트를 남긴다.

    2026-08-19: OpenAIVLMBackend도 같은 이유로 추가 — /exam/extract는 마스킹 전
    원본 이미지·VLM 원문을 다루므로, 이 백엔드가 실수로 langchain_openai.ChatOpenAI
    (Runnable)를 다시 쓰게 바뀌면 LANGCHAIN_TRACING_V2=true일 때 그 마스킹 전
    데이터가 LangSmith로 샐 수 있다 — 이 테스트가 그 회귀를 구조적으로 차단한다."""
    from langchain_core.runnables import Runnable

    from app.common.llm.backends.ollama import OllamaBackend
    from app.common.llm.backends.openai import OpenAIBackend
    from app.common.llm.backends.openai_vlm import OpenAIVLMBackend
    from app.common.llm.backends.runpod import RunPodBackend

    for backend_cls in (OllamaBackend, OpenAIBackend, RunPodBackend, OpenAIVLMBackend):
        assert not issubclass(backend_cls, Runnable)


# ── 2026-08-19: /exam/extract(이미지 입력) — 하드룰 2/3 예외 경계 검증 ────────
# 원본 이미지는 마스킹 전에 VLM으로 전달되는 것이 이 경로의 명시적 예외지만
# (app/main.py 주석 참고), VLM이 반환한 텍스트는 응답으로 나가기 전에 반드시
# mask_pii()를 거쳐야 한다 — 그 경계만 검증한다.


class _FakeVLMBackend:
    """VLM 호출 없이, 이름·학교명·연락처가 섞인 텍스트를 그대로 반환하는 목."""

    async def extract_text(self, image_bytes, mime_type):
        return "이름: 김철수, 분필고등학교 3학년, 연락처 010-1234-5678. 다음 자료를 보고 답하시오."


def test_exam_extract_masks_pii_before_response(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    monkeypatch.setattr(app.common.llm, "get_vlm_backend", lambda: _FakeVLMBackend())

    response = TestClient(fastapi_app).post(
        "/exam/extract",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        files={"image": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "김철수" not in data["text"]
    assert "분필고등학교" not in data["text"]
    assert "010-1234-5678" not in data["text"]
    assert set(data["pii_found"]) == {"이름", "학교명", "전화번호"}
