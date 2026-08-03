"""API 인증·요청 크기 경계 테스트. 모델 호출 없음.

2026-08-03: 생기부 모듈 제거로 `/record`가 사라져 `/exam` 기준으로 옮겼다.
세 케이스 모두 **핸들러 본문에 도달하기 전**(인증 의존성 · 요청 검증 · 크기 미들웨어)에
차단되는 경로만 검증하므로 실제 문항 생성(LLM 호출)은 일어나지 않는다.
"""
from fastapi.testclient import TestClient

from app.main import app


def test_health_does_not_require_authentication():
    response = TestClient(app).get("/health")
    assert response.status_code == 200


def test_protected_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post("/exam", data={"passage_text": "합성 지문"})
    assert response.status_code == 401


def test_valid_key_reaches_request_validation(monkeypatch):
    """인증은 통과하되 필수 필드가 없어 422 — 인증 계층이 검증보다 앞임을 확인."""
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/exam",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        data={},
    )
    assert response.status_code == 422


def test_oversized_request_is_rejected_before_endpoint(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/exam",
        headers={
            "X-Bunpil-Api-Key": "synthetic-secret",
            "Content-Length": str(65 * 1024),
        },
        data={"passage_text": "합성"},
    )
    assert response.status_code == 413


def test_stream_endpoint_also_requires_key(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post("/exam/stream", data={"passage_text": "합성 지문"})
    assert response.status_code == 401
