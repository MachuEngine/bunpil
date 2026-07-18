"""RunPod 백엔드가 작업을 중복 제출하지 않는지 검증한다."""
import asyncio

import httpx
import pytest

from app.common.llm.backends import runpod
from app.common.llm.backends.runpod import RunPodBackend


async def _no_sleep(_seconds):
    return None


def _backend(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "synthetic-endpoint")
    monkeypatch.setattr(runpod.asyncio, "sleep", _no_sleep)
    return RunPodBackend()


def test_submits_once_then_polls_same_job(monkeypatch):
    calls = []
    statuses = iter(["IN_QUEUE", "COMPLETED"])

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-1"})
        status = next(statuses)
        payload = {"status": status}
        if status == "COMPLETED":
            payload["output"] = {"response": "ok", "tool_calls": None}
        return httpx.Response(200, json=payload)

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        runpod.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    result = asyncio.run(_backend(monkeypatch).generate([{"role": "user", "content": "합성"}]))

    assert result == "ok"
    assert calls.count(("POST", "/v2/synthetic-endpoint/run")) == 1
    assert calls.count(("GET", "/v2/synthetic-endpoint/status/job-1")) == 2
    assert all("runsync" not in path for _, path in calls)


def test_submission_timeout_does_not_resubmit(monkeypatch):
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timeout", request=request)

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        runpod.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    with pytest.raises(TimeoutError, match="재제출하지 않습니다"):
        asyncio.run(_backend(monkeypatch).generate([{"role": "user", "content": "합성"}]))

    assert calls == 1


def test_failed_status_does_not_expose_response_body(monkeypatch):
    def handler(request: httpx.Request):
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-secret"})
        return httpx.Response(
            200,
            json={"status": "FAILED", "error": "sensitive provider detail"},
        )

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        runpod.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(_backend(monkeypatch).generate([{"role": "user", "content": "합성"}]))

    assert "FAILED" in str(exc_info.value)
    assert "sensitive provider detail" not in str(exc_info.value)
