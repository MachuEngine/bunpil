"""RunPod 서버리스 vLLM 백엔드.

handler.py 응답 형식:
  {"output": {"response": str | None, "tool_calls": list | None}}
runsync가 타임아웃(30s)되면 비동기 run → status 폴링으로 전환.
"""
import asyncio
import os

import httpx

from ..base import LLMBackend

_BASE = "https://api.runpod.ai/v2"
_POLL_INTERVAL = 5    # seconds
_MAX_POLL      = 120  # 최대 10분 대기


class RunPodBackend(LLMBackend):
    def __init__(self):
        self.api_key     = os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: list, **kwargs) -> dict:
        payload: dict = {
            "input": {
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
            }
        }
        if kwargs.get("tools"):
            payload["input"]["tools"] = kwargs["tools"]
        if kwargs.get("stop"):
            payload["input"]["stop"] = kwargs["stop"]
        return payload

    async def _call_raw(self, messages: list[dict], **kwargs) -> dict:
        """HTTP 요청 후 output dict 반환: {"response": str|None, "tool_calls": list|None}"""
        if not self.api_key or not self.endpoint_id:
            raise RuntimeError(
                "RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID 환경변수가 설정되지 않았습니다."
            )
        async with httpx.AsyncClient(timeout=httpx.Timeout(35, read=35)) as client:
            try:
                resp = await client.post(
                    f"{_BASE}/{self.endpoint_id}/runsync",
                    headers=self._headers(),
                    json=self._payload(messages, **kwargs),
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "COMPLETED":
                    return data["output"]
                job_id = data.get("id")
            except (httpx.ReadTimeout, httpx.TimeoutException):
                resp2 = await client.post(
                    f"{_BASE}/{self.endpoint_id}/run",
                    headers=self._headers(),
                    json=self._payload(messages, **kwargs),
                )
                resp2.raise_for_status()
                job_id = resp2.json()["id"]

        for _ in range(_MAX_POLL):
            await asyncio.sleep(_POLL_INTERVAL)
            async with httpx.AsyncClient(timeout=10) as poll:
                r = await poll.get(
                    f"{_BASE}/{self.endpoint_id}/status/{job_id}",
                    headers=self._headers(),
                )
                r.raise_for_status()
                d = r.json()
                if d.get("status") == "COMPLETED":
                    return d["output"]
                if d.get("status") in ("FAILED", "CANCELLED"):
                    raise RuntimeError(f"RunPod job {job_id} 실패: {d}")

        raise TimeoutError(f"RunPod job {job_id} 응답 초과 ({_MAX_POLL * _POLL_INTERVAL}s)")

    async def generate(self, messages: list[dict], **kwargs) -> str:
        """텍스트 생성 전용 (tool calling 불필요한 경우). 문자열 반환."""
        result = await self._call_raw(messages, **kwargs)
        return result.get("response") or ""

    async def generate_chat(self, messages: list[dict], **kwargs) -> dict:
        """tool calling 포함 생성. {"response": str|None, "tool_calls": list|None} 반환."""
        return await self._call_raw(messages, **kwargs)
