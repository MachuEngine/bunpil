import os

import httpx

from ..base import LLMBackend


class RunPodBackend(LLMBackend):
    """RunPod 서버리스 vLLM 엔드포인트 백엔드. Phase 8에서 연결."""

    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")
        self.model = os.getenv("RUNPOD_MODEL", "qwen2.5-7b")

    async def generate(self, messages: list[dict], **kwargs) -> str:
        if not self.api_key or not self.endpoint_id:
            raise RuntimeError("RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID 환경변수가 설정되지 않았습니다.")
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        payload = {"input": {"messages": messages, "model": self.model, **kwargs}}
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            # vLLM OpenAI-compatible 응답 형식
            return resp.json()["output"]["choices"][0]["message"]["content"]
