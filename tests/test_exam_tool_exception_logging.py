"""agent_node가 도구 실행 예외를 조용히 삼키지 않고 경고 로그를 남기는지 검증
(2026-10-07). LLM 호출 없음 — get_langchain_model/TOOLS를 가짜로 대체.

하드룰 4(로그·캐시에 PII·사용자 입력 원문 금지): 로그에는 도구 이름과 예외 타입만
남아야 하고, 예외 메시지(사용자 입력이 섞일 수 있음)는 남지 않아야 한다."""
import logging

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.modules.exam.graph import agent_node
from app.modules.exam.tools import init_session

_SECRET_PASSAGE_SNIPPET = "교사가-입력한-예시-지문-원문-조각"


@tool
def _broken_tool(x: str) -> str:
    """항상 예외를 던지는 가짜 도구 — 예외 메시지에 입력 원문 흔적을 심어 로그에
    새지 않는지 확인한다."""
    raise ValueError(f"처리 실패: {_SECRET_PASSAGE_SNIPPET}")


class _FakeLLM:
    """bind_tools는 자기 자신을 반환하고, invoke는 매 호출 messages를 기록한 뒤
    준비된 AIMessage를 순서대로 반환한다(test_exam_invalid_tool_calls.py와 동일 패턴)."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


def test_tool_exception_logs_warning_without_exception_message(monkeypatch, caplog):
    init_session("합성 지문", target_num=1)

    first = AIMessage(
        content="",
        tool_calls=[{"name": "_broken_tool", "args": {"x": "아무값"}, "id": "call1"}],
    )
    second = AIMessage(
        content="",
        tool_calls=[{"name": "submit_for_review", "args": {}, "id": "call2"}],
    )
    fake_llm = _FakeLLM([first, second])
    monkeypatch.setattr("app.modules.exam.graph.get_langchain_model", lambda: fake_llm)
    monkeypatch.setattr("app.modules.exam.graph.TOOLS", [_broken_tool])

    state = {"spec": {"passage_text": "합성 지문", "num_items": 1}, "budget": 1}
    with caplog.at_level(logging.WARNING, logger="app.modules.exam.graph"):
        agent_node(state)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("_broken_tool" in r.getMessage() for r in warnings)
    assert any("ValueError" in r.getMessage() for r in warnings)
    # 예외 메시지(사용자 입력 원문 흔적 포함)는 로그에 남지 않아야 한다(하드룰 4).
    assert all(_SECRET_PASSAGE_SNIPPET not in r.getMessage() for r in caplog.records)
    assert all("처리 실패" not in r.getMessage() for r in caplog.records)
