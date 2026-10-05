"""OllamaVLMBackend가 Ollama /api/chat에 보내는 요청 본문을 네트워크 없이 검증한다.
(test_runpod_backend.py의 httpx.MockTransport 패턴을 따름)"""
import asyncio
import base64
import json

import httpx

from app.common.llm.backends import ollama_vlm
from app.common.llm.backends.ollama_vlm import OllamaVLMBackend
from app.common.llm.backends.openai_vlm import _EXTRACT_PROMPT


def test_extract_text_sends_expected_request_body(monkeypatch):
    monkeypatch.setenv("OLLAMA_VLM_MODEL", "qwen3-vl:8b")
    captured = {}

    def handler(request: httpx.Request):
        captured["body"] = request.read()
        return httpx.Response(200, json={"message": {"content": "추출된 텍스트"}})

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        ollama_vlm.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    backend = OllamaVLMBackend()
    result = asyncio.run(backend.extract_text(b"fake-image-bytes", "image/png"))

    assert result == "추출된 텍스트"
    payload = json.loads(captured["body"])
    assert payload["model"] == "qwen3-vl:8b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["messages"][0] == {"role": "system", "content": _EXTRACT_PROMPT}
    user_message = payload["messages"][1]
    assert user_message["role"] == "user"
    assert user_message["images"] == [base64.b64encode(b"fake-image-bytes").decode("ascii")]


def test_missing_model_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OLLAMA_VLM_MODEL", raising=False)
    try:
        OllamaVLMBackend()
    except RuntimeError as exc:
        assert "OLLAMA_VLM_MODEL" in str(exc)
    else:
        raise AssertionError("OLLAMA_VLM_MODEL 미설정 시 RuntimeError가 발생해야 합니다.")
