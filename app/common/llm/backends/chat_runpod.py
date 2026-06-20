"""LangChain BaseChatModel adapter for RunPod serverless vLLM.

LangGraph ReAct 에이전트는 LangChain 인터페이스가 필요하므로
기존 RunPodBackend(단순 HTTP) 위에 래퍼를 씌운다.
"""
import asyncio
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .runpod import RunPodBackend


def _to_runpod_messages(messages: List[BaseMessage]) -> List[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]


class ChatRunPod(BaseChatModel):
    """RunPod 서버리스 vLLM 백엔드를 LangChain 인터페이스로 감싼 어댑터."""

    max_tokens: int = 512
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "chat-runpod"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        import os
        return {"endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID", "")}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """동기 컨텍스트용 폴백 — 별도 이벤트 루프에서 비동기 백엔드 실행."""
        loop = asyncio.new_event_loop()
        try:
            text = loop.run_until_complete(self._call_backend(messages))
        finally:
            loop.close()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """FastAPI/LangGraph 비동기 컨텍스트에서 호출되는 주 경로."""
        text = await self._call_backend(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _call_backend(self, messages: List[BaseMessage]) -> str:
        backend = RunPodBackend()
        return await backend.generate(
            _to_runpod_messages(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
