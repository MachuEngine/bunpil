"""OpenRouter 텍스트 전용 백엔드 (LLMBackend 인터페이스). OpenAIBackend의 generate()
로직을 그대로 재사용하고 chat 어댑터·모델 env만 OpenRouter용으로 교체한다.
평가(모델 비교, 2026-10) 전용 — 런타임 생성/Judge 경로에서는 아직 쓰지 않는다."""
from .chat_openrouter import ChatOpenRouterBackend
from .openai import OpenAIBackend


class OpenRouterBackend(OpenAIBackend):
    _chat_cls = ChatOpenRouterBackend
    _model_env = "OPENROUTER_MODEL"
    _default_model = None  # OpenAI 백엔드와 달리 기본 모델을 임의로 정하지 않음
