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
