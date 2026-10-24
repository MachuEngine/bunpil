"""OpenRouter(OpenAI 호환 API) Chat Completions 백엔드 어댑터 (LangChain 호환).

OpenRouter는 OpenAI와 동일한 Chat Completions 스펙을 쓰므로 langchain_openai의
ChatOpenAI를 base_url만 바꿔 그대로 재사용한다 — chat_openai.py와 같은 패턴.
2026-10-18부터 운영 생성·Judge 경로(gpt-6-luna 생성, claude-sonnet-5.5 Judge)이자
평가(모델 비교, gemini-3.8-flash 생성·Claude/gemini Judge)용으로도 쓴다.
"""
import os

from langchain_openai import ChatOpenAI

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ChatOpenRouterBackend(ChatOpenAI):
    """OPENROUTER_API_KEY/OPENROUTER_MODEL 환경변수 기본값을 프로젝트 관례에 맞춘 얇은 래퍼."""

    def __init__(self, temperature: float = 0.7, max_tokens: int = 2048, model: str | None = None, **kwargs):
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")
        resolved_model = model or os.getenv("OPENROUTER_MODEL", "")
        if not resolved_model:
            # OpenAI 백엔드와 달리 기본 모델을 임의로 정하지 않는다 — 운영과 평가 모두
            # 쓸 모델을 명시값(OPENROUTER_MODEL)으로 요구한다.
            raise RuntimeError("OPENROUTER_MODEL 환경변수가 설정되지 않았습니다.")
        super().__init__(
            model=resolved_model,
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
