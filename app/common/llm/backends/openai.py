"""OpenAI 텍스트 전용 백엔드 (LLMBackend 인터페이스). chat_openai.py의 LangChain
어댑터를 내부적으로 재사용 — tool calling이 필요없는 단순 generate() 호출용."""
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..base import LLMBackend
from .chat_openai import ChatOpenAIBackend

_ROLE_MAP = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


class OpenAIBackend(LLMBackend):
    # 하위 클래스(OpenRouterBackend 등)가 chat 어댑터·모델 env·기본 모델만 바꿔
    # generate() 로직을 그대로 재사용하도록 클래스 속성으로 분리해 둠.
    _chat_cls = ChatOpenAIBackend
    _model_env = "OPENAI_MODEL"
    _default_model = "gpt-4o-mini"

    def __init__(self, model=None):
        self.model = model or os.getenv(self._model_env, self._default_model)

    async def generate(self, messages: list[dict], **kwargs) -> str:
        chat = self._chat_cls(
            model=self.model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        lc_messages = [_ROLE_MAP.get(m["role"], HumanMessage)(content=m["content"]) for m in messages]
        result = await chat.ainvoke(lc_messages)
        return result.content or ""
