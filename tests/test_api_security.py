"""API 인증·요청 크기 경계 테스트. 모델 호출 없음.

2026-08-03: 생기부 모듈 제거로 `/record`가 사라져 `/exam` 기준으로 옮겼다.
세 케이스 모두 **핸들러 본문에 도달하기 전**(인증 의존성 · 요청 검증 · 크기 미들웨어)에
차단되는 경로만 검증하므로 실제 문항 생성(LLM 호출)은 일어나지 않는다.
"""
import app.common.llm as common_llm
import app.main as main_module
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


# ── 2026-08-04: 동시성 슬롯·에러 핸들링 배선 확인 ────────────────────────
# 세마포어 순서·예외 처리 수정(app/main.py) 회귀 방지용. LLM은 FakeBackend로 대체.


def test_exam_slot_covers_num_items_llm_call(monkeypatch):
    """세마포어가 그래프 실행뿐 아니라 _extract_num_items()의 LLM 호출까지
    커버하는지 확인한다 — 이전엔 _build_spec()이 슬롯 확보 전에 실행돼 이
    LLM 호출이 동시요청 제한을 우회할 수 있었다."""
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    initial = main_module._REQUEST_SLOTS._value
    observed = {}

    class _FakeBackend:
        async def generate(self, messages, **kwargs):
            observed["value_during_call"] = main_module._REQUEST_SLOTS._value
            return "2"

    monkeypatch.setattr(common_llm, "get_llm_backend", lambda: _FakeBackend())

    async def fake_run_exam(spec):
        return {"items": [], "validation_passed": True}

    monkeypatch.setattr(main_module, "_run_exam", fake_run_exam)

    response = TestClient(app).post(
        "/exam",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        data={"passage_text": "합성 지문"},
    )
    assert response.status_code == 200
    assert observed["value_during_call"] < initial
    assert main_module._REQUEST_SLOTS._value == initial  # 요청 종료 후 슬롯 반납 확인


def test_exam_returns_graceful_error_instead_of_raw_500(monkeypatch):
    """`_run_exam`이 예외를 던져도 /exam은 /exam/stream과 동일하게 로깅 후
    graceful한 JSON 에러로 응답해야 한다(수정 전엔 처리되지 않은 500이 그대로 샜다)."""
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    initial = main_module._REQUEST_SLOTS._value

    class _FakeBackend:
        async def generate(self, messages, **kwargs):
            return "2"

    monkeypatch.setattr(common_llm, "get_llm_backend", lambda: _FakeBackend())

    async def failing_run_exam(spec):
        raise RuntimeError("합성 오류")

    monkeypatch.setattr(main_module, "_run_exam", failing_run_exam)

    response = TestClient(app).post(
        "/exam",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        data={"passage_text": "합성 지문"},
    )
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    # 슬롯이 예외 상황에서도 반납됐는지 확인(누수 방지)
    assert main_module._REQUEST_SLOTS._value == initial


# ── 2026-08-19: /exam/extract(이미지 입력) 경계 테스트 ────────────────────
# VLM 호출 없음 — 인증·MIME 화이트리스트·크기 제한 세 경계가 핸들러 본문(VLM 호출)에
# 도달하기 전에 각각 차단하는지만 검증한다.

_PNG_BYTES = b"\x89PNG\r\n\x1a\n"  # 시그니처만 있는 최소 더미 — 실제 디코딩은 하지 않음


def test_exam_extract_requires_api_key(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/exam/extract",
        files={"image": ("test.png", _PNG_BYTES, "image/png")},
    )
    assert response.status_code == 401


def test_exam_extract_rejects_unsupported_mime(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/exam/extract",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        files={"image": ("test.gif", b"GIF89a", "image/gif")},
    )
    assert response.status_code == 400


def test_exam_extract_rejects_oversized_image(monkeypatch):
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    response = TestClient(app).post(
        "/exam/extract",
        headers={
            "X-Bunpil-Api-Key": "synthetic-secret",
            "Content-Length": str(6 * 1024 * 1024),
        },
        files={"image": ("test.png", _PNG_BYTES, "image/png")},
    )
    assert response.status_code == 413


def test_exam_extract_releases_slot_and_masks_error_on_vlm_failure(monkeypatch):
    """VLM 호출이 예외를 던져도 (1) 세마포어가 반납되고 (2) 스택트레이스 없이
    generic 502로 응답해야 한다 — /exam의 test_exam_returns_graceful_error_instead_of_raw_500
    과 동일한 회귀 방지 목적."""
    monkeypatch.setenv("BUNPIL_API_KEY", "synthetic-secret")
    initial = main_module._REQUEST_SLOTS._value

    class _FailingVLMBackend:
        async def extract_text(self, image_bytes, mime_type):
            raise RuntimeError("합성 오류")

    monkeypatch.setattr(common_llm, "get_vlm_backend", lambda: _FailingVLMBackend())

    response = TestClient(app).post(
        "/exam/extract",
        headers={"X-Bunpil-Api-Key": "synthetic-secret"},
        files={"image": ("test.png", _PNG_BYTES, "image/png")},
    )
    assert response.status_code == 502
    assert "합성 오류" not in response.text
    assert main_module._REQUEST_SLOTS._value == initial
