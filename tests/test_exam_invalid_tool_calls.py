"""agent_node가 invalid_tool_calls(인자 JSON이 깨진 도구 호출)에도 ToolMessage로
응답하는지 검증한다. LLM 호출 없음 — get_langchain_model을 가짜 LLM으로 대체.

배경(2026-10-03): langchain-openai는 다음 요청을 보낼 때 정상 tool_calls뿐 아니라
invalid_tool_calls도 assistant 메시지의 tool_calls로 함께 직렬화한다
(langchain_openai/chat_models/base.py _convert_message_to_dict). agent_node가
tool_calls만 ToolMessage로 응답하면, 응답 없는 tool call이 대화에 남아 OpenAI 계열
API가 "No tool output found for function call ..." 400으로 거부한다."""
from langchain_core.messages import AIMessage, ToolMessage

from app.modules.exam.graph import agent_node
from app.modules.exam.tools import init_session


def _invalid_tool_call(call_id: str, error: str = "json parse error") -> dict:
    return {
        "type": "invalid_tool_call",
        "id": call_id,
        "name": "save_item",
        "args": "{bad json",
        "error": error,
    }


class _FakeLLM:
    """bind_tools는 자기 자신을 반환하고, invoke는 매 호출 messages를 기록한 뒤
    준비된 AIMessage를 순서대로 반환한다."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.invoke_calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invoke_calls.append(list(messages))
        return self._responses.pop(0)


def _answered_tool_call_ids(messages) -> set:
    return {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}


def _requested_tool_call_ids(messages) -> set:
    """모든 AIMessage의 tool_calls·invalid_tool_calls id 합집합."""
    ids = set()
    for m in messages:
        if isinstance(m, AIMessage):
            ids |= {tc["id"] for tc in m.tool_calls if tc.get("id")}
            ids |= {tc["id"] for tc in (m.invalid_tool_calls or []) if tc.get("id")}
    return ids


def _assert_every_requested_id_answered(messages):
    assert _requested_tool_call_ids(messages) <= _answered_tool_call_ids(messages)


def test_invalid_tool_call_gets_tool_message_and_existing_broken_format_flow_kept(monkeypatch):
    """tool_calls는 비고 invalid_tool_calls만 있는 응답 — invalid 쪽은 ToolMessage로
    답하고, 기존 흐름(도구 호출 형식이 손상 HumanMessage + malformed_streak)은 유지돼야 한다."""
    init_session("합성 지문", target_num=1)

    # content에 깨진 tool_call JSON 흔적을 남겨 _looks_like_broken_tool_call이
    # "도구 호출 형식이 손상" 분기를 타도록 한다(기존 휴리스틱, graph.py 참고).
    first = AIMessage(
        content='{"name": "save_item", 이 뒤가 깨짐',
        invalid_tool_calls=[_invalid_tool_call("bad1")],
    )
    second = AIMessage(
        content="",
        tool_calls=[{"name": "submit_for_review", "args": {}, "id": "ok_final"}],
    )
    fake_llm = _FakeLLM([first, second])
    monkeypatch.setattr("app.modules.exam.graph.get_langchain_model", lambda: fake_llm)

    state = {"spec": {"passage_text": "합성 지문", "num_items": 1}, "budget": 1}
    result = agent_node(state)

    # 두 번째 invoke에 넘어간 messages에 bad1용 ToolMessage와, 그 뒤에 기존
    # "도구 호출 형식이 손상" HumanMessage가 모두 있어야 한다.
    second_call_messages = fake_llm.invoke_calls[1]
    assert "bad1" in _answered_tool_call_ids(second_call_messages)
    broken_format_notices = [
        m for m in second_call_messages
        if hasattr(m, "content") and "도구 호출 형식이 손상" in str(m.content)
    ]
    assert len(broken_format_notices) == 1

    _assert_every_requested_id_answered(result["agent_messages"])


def test_mixed_valid_and_invalid_tool_calls_both_get_tool_messages(monkeypatch):
    """정상 tool_call(ok1)과 invalid(bad2)가 한 응답에 섞여 있으면 둘 다 ToolMessage를
    받아야 한다 — 정상 호출은 그대로 실행되고, invalid는 오류 ToolMessage만 받는다."""
    init_session("합성 지문", target_num=1)

    response = AIMessage(
        content="",
        tool_calls=[{"name": "submit_for_review", "args": {}, "id": "ok1"}],
        invalid_tool_calls=[_invalid_tool_call("bad2")],
    )
    fake_llm = _FakeLLM([response])
    monkeypatch.setattr("app.modules.exam.graph.get_langchain_model", lambda: fake_llm)

    state = {"spec": {"passage_text": "합성 지문", "num_items": 1}, "budget": 1}
    result = agent_node(state)

    messages = result["agent_messages"]
    answered = _answered_tool_call_ids(messages)
    assert {"ok1", "bad2"} <= answered
    _assert_every_requested_id_answered(messages)
