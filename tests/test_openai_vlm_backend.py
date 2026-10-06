"""OpenAIVLMBackend/_call_openai_vlm이 gpt-6-luna 전환(2026-10)에 맞춰 보내는 요청
파라미터를 네트워크 없이 검증한다 — AsyncOpenAI(http_client=...)에 httpx.MockTransport를
꽂아 실제 HTTP 전송만 가로채는 방식(test_ollama_vlm_backend.py의 MockTransport 패턴을
openai SDK 클라이언트에 맞게 적용). OpenRouterVLMBackend(_call_vlm)는 기존 요청 형식
(max_tokens·temperature)을 그대로 유지하는지만 확인한다 — 동작을 바꾸지 않았음을 보증."""
import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from app.common.llm.backends import openai_vlm
from app.common.llm.backends.openai_vlm import OpenAIVLMBackend, OpenRouterVLMBackend, _EXTRACT_PROMPT


def _client_with_transport(handler):
    transport = httpx.MockTransport(handler)
    return AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))


def test_openai_vlm_backend_sends_max_completion_tokens_not_max_tokens(monkeypatch):
    captured = {}

    def handler(request: httpx.Request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": "추출됨"}}]})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_vlm, "AsyncOpenAI", lambda **kwargs: _client_with_transport(handler))
    backend = OpenAIVLMBackend(model="gpt-6-luna")

    result = asyncio.run(backend.extract_text(b"fake-image-bytes", "image/png"))

    assert result == "추출됨"
    body = captured["body"]
    assert body["model"] == "gpt-6-luna"
    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    assert body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": _EXTRACT_PROMPT}


def test_openai_vlm_backend_retries_without_temperature_on_400(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        body = json.loads(request.read())
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                    },
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "재시도 성공"}}]})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_vlm, "AsyncOpenAI", lambda **kwargs: _client_with_transport(handler))
    backend = OpenAIVLMBackend(model="gpt-6-luna")

    result = asyncio.run(backend.extract_text(b"fake-image-bytes", "image/png"))

    assert result == "재시도 성공"
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert calls[1]["max_completion_tokens"] == 2048  # 토큰 파라미터는 재시도에도 유지


def test_openai_vlm_backend_propagates_non_temperature_400(monkeypatch):
    def handler(request: httpx.Request):
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'foo'"}})

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai_vlm, "AsyncOpenAI", lambda **kwargs: _client_with_transport(handler))
    backend = OpenAIVLMBackend(model="gpt-6-luna")

    with pytest.raises(Exception):
        asyncio.run(backend.extract_text(b"fake-image-bytes", "image/png"))


def test_openai_vlm_backend_default_model_is_gpt6luna(monkeypatch):
    monkeypatch.delenv("OPENAI_VLM_MODEL", raising=False)
    backend = OpenAIVLMBackend()
    assert backend.model == "gpt-6-luna"


def test_openrouter_vlm_backend_keeps_max_tokens_param_unchanged(monkeypatch):
    """공유 파라미터를 분리한 뒤에도 OpenRouter 경로는 기존 요청 형식(max_tokens)을
    그대로 보낸다 — gpt-6-luna 전환이 OpenRouter 평가 경로를 깨지 않았는지 확인."""
    captured = {}

    def handler(request: httpx.Request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json={"choices": [{"message": {"content": "추출됨"}}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openai_vlm, "AsyncOpenAI", lambda **kwargs: _client_with_transport(handler))
    backend = OpenRouterVLMBackend(model="gemini-3.8-flash")

    result = asyncio.run(backend.extract_text(b"fake-image-bytes", "image/png"))

    assert result == "추출됨"
    body = captured["body"]
    assert body["model"] == "gemini-3.8-flash"
    assert body["max_tokens"] == 2048
    assert "max_completion_tokens" not in body
    assert body["temperature"] == 0
