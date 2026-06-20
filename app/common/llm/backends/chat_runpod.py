"""LangChain BaseChatModel adapter for RunPod serverless vLLM.

LangGraph ReAct 에이전트는 LangChain 인터페이스가 필요하므로
기존 RunPodBackend(단순 HTTP) 위에 래퍼를 씌운다.
"""
import asyncio
import json
from typing import Any, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .runpod import RunPodBackend


def _to_runpod_messages(messages: List[BaseMessage]) -> List[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    result = []
    for m in messages:
        msg: dict = {"role": role_map.get(m.type, "user"), "content": m.content or ""}
        # AIMessage가 tool_calls를 가질 때 OpenAI 호환 형식으로 포함
        if isinstance(m, AIMessage) and m.tool_calls:
            msg["content"] = m.content or None
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        # ToolMessage는 tool_call_id가 있어야 vLLM이 결과를 매핑 가능
        if isinstance(m, ToolMessage):
            msg["tool_call_id"] = m.tool_call_id
        result.append(msg)
    return result


class ChatRunPod(BaseChatModel):
    """RunPod 서버리스 vLLM 백엔드를 LangChain 인터페이스로 감싼 어댑터."""

    max_tokens: int = 2048
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "chat-runpod"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        import os
        return {"endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID", "")}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        from langchain_core.utils.function_calling import convert_to_openai_tool
        tool_defs = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=tool_defs, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """동기 컨텍스트용 폴백."""
        text = asyncio.run(self._call_backend(messages, stop=stop, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """FastAPI/LangGraph 비동기 컨텍스트에서 호출되는 주 경로."""
        text = await self._call_backend(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _call_backend(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        backend = RunPodBackend()
        return await backend.generate(
            _to_runpod_messages(messages),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
