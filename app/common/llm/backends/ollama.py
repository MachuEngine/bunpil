import json
import os

import httpx

from ..base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    async def generate(self, messages: list[dict], **kwargs) -> str:
        # stream=True로 토큰 단위 수신 → CPU 느린 환경에서도 timeout 회피
        payload = {"model": self.model, "messages": messages, "stream": True, **kwargs}
        tokens = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=300)) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    tokens.append(chunk["message"]["content"])
                    if chunk.get("done"):
                        break
        return "".join(tokens)
