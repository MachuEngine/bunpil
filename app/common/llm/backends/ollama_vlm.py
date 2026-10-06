"""Ollama 로컬 VLM 백엔드 (VLMBackend 인터페이스).

2026-10: VLM 모델 비교 평가 전용(qwen3-vl:8b, gemma3:12b 등 로컬 후보). 기존
`OllamaBackend`(backends/ollama.py)와 같은 패턴으로 httpx를 직접 써서 /api/chat을
호출하고 LangChain은 거치지 않는다 — 이 경로가 다루는 마스킹 전 원본 이미지가
LangSmith로 새지 않아야 하는 이유는 openai_vlm.py 모듈 docstring과 동일하다.
"""
import base64
import os

import httpx

from ..base import VLMBackend
from .openai_vlm import _EXTRACT_PROMPT


class OllamaVLMBackend(VLMBackend):
    def __init__(self, model: str | None = None):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_VLM_MODEL")
        if not self.model:
            raise RuntimeError(
                "OLLAMA_VLM_MODEL 환경변수가 설정되지 않았습니다 — 평가 비교 목적상 "
                "기본 모델을 임의로 지정하지 않습니다."
            )

    async def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user", "content": "", "images": [b64]},
            ],
            "stream": False,
            # num_ctx: 이미지 토큰(고해상도 캡처 기준 수백~수천)을 담을 수 있도록 넉넉히 잡음
            "options": {"temperature": 0, "num_predict": 2048, "num_ctx": 8192},
        }
        # 로컬 VLM은 이미지 처리 특성상 텍스트 전용 모델보다 느릴 수 있어 타임아웃을 넉넉히 둠
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=300)) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return (data.get("message", {}).get("content") or "").strip()
