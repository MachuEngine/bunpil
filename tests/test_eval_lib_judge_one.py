"""evals/eval_lib.py judge_one()의 stimulus 포함·parse_failed 플래그 단위 테스트.

LLM 호출 없음 — generate()가 고정 문자열을 반환하는 가짜 llm으로 검증한다. eval_lib는
app.common.rag/app.modules.exam.judge를 import해 다소 무겁지만(로컬에서 import만 약 2초),
실제 모델 호출은 전혀 하지 않는다.
"""
import json

from evals.eval_lib import judge_one


class _FakeLLM:
    """generate()에 전달된 messages를 기록하고 고정 raw 문자열을 반환한다."""

    def __init__(self, raw: str):
        self.raw = raw
        self.calls: list[list[dict]] = []

    async def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.raw


_FIXED_JSON = '{"정답유일성": 5, "오답매력도": 4, "근거성": 3}'


def test_judge_one_includes_stimulus_when_present():
    item = {
        "question": "<보기>를 참고할 때 옳은 것은?",
        "stimulus": "<보기> 갑, 을이 계약을 체결하였다.",
        "options": ["①", "②", "③", "④"],
        "answer": "①",
    }
    llm = _FakeLLM(_FIXED_JSON)
    judge_one(item, llm)

    user_turn = llm.calls[0][-1]["content"]
    sent = json.loads(user_turn)
    assert sent["stimulus"] == item["stimulus"]


def test_judge_one_omits_stimulus_when_missing_or_empty():
    llm = _FakeLLM(_FIXED_JSON)
    judge_one({"question": "q", "options": ["①", "②"], "answer": "①"}, llm)
    sent_no_key = json.loads(llm.calls[0][-1]["content"])
    assert "stimulus" not in sent_no_key

    llm2 = _FakeLLM(_FIXED_JSON)
    judge_one({"question": "q", "stimulus": "", "options": ["①", "②"], "answer": "①"}, llm2)
    sent_empty = json.loads(llm2.calls[0][-1]["content"])
    assert "stimulus" not in sent_empty


def test_judge_one_valid_json_sets_parse_failed_false():
    llm = _FakeLLM(_FIXED_JSON)
    result = judge_one({"question": "q", "options": ["①"], "answer": "①"}, llm)
    assert result["parse_failed"] is False
    assert result["정답유일성"] == 5
    assert result["오답매력도"] == 4
    assert result["근거성"] == 3


def test_judge_one_unparseable_raw_sets_parse_failed_true_with_default_scores():
    llm = _FakeLLM("이것은 JSON이 아닙니다.")
    result = judge_one({"question": "q", "options": ["①"], "answer": "①"}, llm)
    assert result["parse_failed"] is True
    assert result["정답유일성"] == 3
    assert result["오답매력도"] == 3
    assert result["근거성"] == 3
    assert result["overall"] == 3.0
