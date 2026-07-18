"""API 인증·요청 크기 경계 테스트. 모델 호출 없음."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_does_not_require_authentication():
    response = TestClient(app).get("/health")
    assert response.status_code == 200


def test_protected_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post("/record", json={"memo": "합성 메모"})
    assert response.status_code == 401


def test_valid_key_reaches_request_validation(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/record",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        json={},
    )
    assert response.status_code == 422


def test_oversized_request_is_rejected_before_endpoint(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/record",
        headers={
            "X-Bunpil-Api-Key": "synthetic-secret",
            "Content-Length": str(65 * 1024),
        },
        json={"memo": "합성"},
    )
    assert response.status_code == 413


def test_record_memo_length_is_bounded(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/record",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        json={"memo": "가" * 4001},
    )
    assert response.status_code == 422
