"""해설(/exam/explain) 경로 테스트 — 2026-09 기능 추가. 모델 호출 없음(가짜 백엔드).

브라우저가 문항을 다시 보내는 경로라, 그 문항도 사용자 입력으로 보고 **모델 호출 전에**
마스킹되는지(하드룰 2), 인증·형식 검증·슬롯 해제가 기존 엔드포인트와 같은지 확인한다.
"""
import json

import app.main as main_module
import app.modules.exam.explain as explain_module
from fastapi.testclient import TestClient

from app.main import app

HEADERS = {"X-Bunpil-Api-Key": "synthetic-secret"}
ITEM = {
    "item_id": "abc12345",
    "question": "김철수 학생이 조사한 시장 실패의 사례로 옳은 것은?",
    "stimulus": "",
    "options": ["① 공공재 무임승차", "② 완전 경쟁", "③ 가격 신축성", "④ 자유 진입"],
    "answer": "①",
    "item_type": "객관식",
    "difficulty": "중",
    "standard": "",
}


class _FakeBackend:
    def __init__(self, reply="① 공공재는 비배제성 때문에 무임승차가 생긴다.", fail=False):
        self.messages = None
        self.kwargs = None
        self.reply = reply
        self.fail = fail

    async def generate(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        return self.reply


def _post_explain(item):
    return TestClient(app).post("/exam/explain", headers=HEADERS, data={"item": json.dumps(item, ensure_ascii=False)})


def test_explain_requires_api_key(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post("/exam/explain", data={"item": json.dumps(ITEM)})
    assert response.status_code == 401


def test_explain_masks_item_before_model_call(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    backend = _FakeBackend()
    monkeypatch.setattr(explain_module, "get_llm_backend", lambda: backend)

    response = _post_explain(ITEM)

    assert response.status_code == 200
    body = response.json()
    assert body["explanation"].startswith("①")
    assert body["pii_found"] == ["이름"]
    model_input = "\n".join(m["content"] for m in backend.messages)
    assert "김철수" not in model_input
    assert "[정답] ①" in model_input
    assert backend.kwargs["max_tokens"] >= 1024  # RunPod 기본 256으로 잘리지 않게


def test_explain_rejects_malformed_item(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    for bad in ({**ITEM, "item_type": "주관식"}, {**ITEM, "options": ["①"] * 6}, {**ITEM, "question": ""}, ["not", "a", "dict"]):
        assert _post_explain(bad).status_code == 400
    bad_json = TestClient(app).post("/exam/explain", headers=HEADERS, data={"item": "{not json"})
    assert bad_json.status_code == 400


def test_explain_releases_slot_and_hides_error_on_backend_failure(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    monkeypatch.setattr(explain_module, "get_llm_backend", lambda: _FakeBackend(fail=True))

    response = _post_explain(ITEM)

    assert response.status_code == 502
    assert "synthetic backend failure" not in response.text
    assert main_module._REQUEST_SLOTS._value == 2


# ── 챗봇 수정(/exam/revise) ──────────────────────────────────────────────
import app.modules.exam.revise as revise_module

PASSAGE = "1. 다음 중 시장 실패의 사례로 옳은 것은?\n① 공공재 ② 완전 경쟁 ③ 가격 신축성 ④ 자유 진입"
REVISED_2 = {
    "number": 2,
    "question": "헌법이 보장하는 기본권 중 자유권에 해당하는 것은?",
    "stimulus": "",
    "options": ["① 신체의 자유", "② 선거권", "③ 교육을 받을 권리", "④ 청원권"],
    "answer": "①",
    "item_type": "객관식",
    "difficulty": "중",
}
ITEM_2 = {**ITEM, "item_id": "def67890", "question": "민주 선거의 기본 원칙으로 옳은 것은?", "answer": "②"}


class _ScriptedBackend:
    """호출될 때마다 준비된 응답을 차례로 돌려주는 가짜 백엔드."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self.replies.pop(0)


def _post_revise(instruction, backend, monkeypatch, history=None, items=None):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    monkeypatch.setattr(revise_module, "get_llm_backend", lambda: backend)
    return TestClient(app).post("/exam/revise", headers=HEADERS, data={
        "passage_text": PASSAGE,
        "items": json.dumps(items or [ITEM, ITEM_2], ensure_ascii=False),
        "history": json.dumps(history or [], ensure_ascii=False),
        "instruction": instruction,
    })


def test_revise_masks_every_input_before_model_call(monkeypatch):
    backend = _ScriptedBackend([json.dumps({"message": "2번을 바꿨습니다.", "changes": [REVISED_2]}, ensure_ascii=False)])
    history = [{"role": "user", "content": "이영희 학생이 푼 문제야"}, {"role": "assistant", "content": "네"}]

    response = _post_revise("2번을 01012345678 번호 없이 자유권 문제로 바꿔줘", backend, monkeypatch, history)

    assert response.status_code == 200
    body = response.json()
    assert body["changes"] == [{"number": 2, "item": {**{k: ITEM_2[k] for k in ("standard",)}, **{k: v for k, v in REVISED_2.items() if k != "number"}}}]
    model_input = json.dumps(backend.calls[0], ensure_ascii=False)
    for raw in ("김철수", "이영희", "01012345678"):
        assert raw not in model_input
    assert set(body["pii_found"]) >= {"이름", "전화번호"}


def test_revise_keeps_original_when_change_fails_gates_twice(monkeypatch):
    bad = {**REVISED_2, "options": ["① 신체의 자유", "② 선거권"]}  # 선지 2개 → 형식 오류
    reply = json.dumps({"message": "바꿨습니다.", "changes": [bad]}, ensure_ascii=False)
    backend = _ScriptedBackend([reply, reply])

    body = _post_revise("2번 바꿔줘", backend, monkeypatch).json()

    assert body["changes"] == []
    assert "2번" in body["message"] and "원래 문항을 유지" in body["message"]
    assert len(backend.calls) == 2  # 한 번은 오류를 알려 다시 쓰게 한다
    assert "형식 검사에 걸렸습니다" in backend.calls[1][-1]["content"]


def test_revise_accepts_fix_on_retry(monkeypatch):
    bad = {**REVISED_2, "answer": "⑤"}
    backend = _ScriptedBackend([
        json.dumps({"message": "", "changes": [bad]}, ensure_ascii=False),
        json.dumps({"message": "고쳤습니다.", "changes": [REVISED_2]}, ensure_ascii=False),
    ])

    body = _post_revise("2번 바꿔줘", backend, monkeypatch).json()

    assert [c["number"] for c in body["changes"]] == [2]
    assert body["message"] == "고쳤습니다."


def test_revise_rejects_duplicate_of_other_item(monkeypatch):
    item_1 = {**ITEM, "question": "다음 중 시장 실패의 원인으로 가장 적절한 것은?"}
    dup = {**REVISED_2, "question": item_1["question"]}  # 1번과 같은 발문
    reply = json.dumps({"message": "", "changes": [dup]}, ensure_ascii=False)

    body = _post_revise("2번 바꿔줘", _ScriptedBackend([reply, reply]), monkeypatch, items=[item_1, ITEM_2]).json()

    assert body["changes"] == []


def test_revise_rejects_malformed_request(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    client = TestClient(app)
    base = {"passage_text": PASSAGE, "items": json.dumps([ITEM]), "instruction": "바꿔줘"}
    assert client.post("/exam/revise", data=base).status_code == 401
    for override in (
        {"items": "[]"},
        {"items": "{bad"},
        {"instruction": " "},
        {"instruction": "가" * 1001},
        {"history": json.dumps([{"role": "system", "content": "x"}])},
        {"history": json.dumps([{"role": "user", "content": "x"}] * 7)},
    ):
        assert client.post("/exam/revise", headers=HEADERS, data={**base, **override}).status_code == 400


def test_revise_releases_slot_and_hides_error_on_backend_failure(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    monkeypatch.setattr(revise_module, "get_llm_backend", lambda: _FakeBackend(fail=True))

    response = TestClient(app).post("/exam/revise", headers=HEADERS, data={
        "passage_text": PASSAGE, "items": json.dumps([ITEM]), "instruction": "바꿔줘",
    })

    assert response.status_code == 502
    assert "synthetic backend failure" not in response.text
    assert main_module._REQUEST_SLOTS._value == 2


def test_revise_retries_once_when_response_is_not_json(monkeypatch):
    backend = _ScriptedBackend(["죄송합니다, 수정하겠습니다.", json.dumps({"message": "바꿨습니다.", "changes": [REVISED_2]}, ensure_ascii=False)])

    body = _post_revise("2번 바꿔줘", backend, monkeypatch).json()

    assert [c["number"] for c in body["changes"]] == [2]
    assert "JSON 하나만" in backend.calls[1][-1]["content"]


def test_revise_applies_combo_rules_when_teacher_asks_for_them(monkeypatch):
    # <보기>에 선지를 복사하고 선지는 서술문 — 합답형 요청에는 불합격이어야 한다
    not_combo = {**REVISED_2, "stimulus": "<보기>\n① 신체의 자유\n② 선거권", "difficulty": "상"}
    reply = json.dumps({"message": "", "changes": [not_combo]}, ensure_ascii=False)

    body = _post_revise("2번을 <보기> 합답형으로 바꿔줘", _ScriptedBackend([reply, reply]), monkeypatch).json()

    assert body["changes"] == []
