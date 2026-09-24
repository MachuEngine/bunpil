"""OpenRouter 요청 고정·실행 기록 — 모델 선정 평가 전용.

정식 비교 조건(사용자 지시 2026-09-24):
- 정확한 모델 ID 고정(`auto`·`latest`·`:free` 등 실행 대상이 바뀔 수 있는 별칭 금지)
- provider endpoint 고정 + provider·모델 fallback 비활성화 + `require_parameters`
- 자동 context compression 비활성화
- 오픈 모델은 quantization 고정
- 실제 처리 모델·provider·토큰·비용·지연·오류 기록, 요청과 다르면 집계 제외
- 인증은 `OPENROUTER_API_KEY` 환경변수에서만 읽고 어디에도 기록하지 않는다
"""
import os
import re

_FORBIDDEN_MODEL_PATTERNS = re.compile(r"(^openrouter/auto$|:latest$|:free$|(^|/)auto$)")


class RoutingConfigError(ValueError):
    pass


def api_key() -> str:
    """키를 읽기만 한다. 값은 반환 외에 어디에도 남기지 않는다(로그·결과·예외 메시지 포함)."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RoutingConfigError("OPENROUTER_API_KEY 환경변수가 없습니다 — --mock 또는 --dry-run으로 실행하세요.")
    return key


def provider_block(spec: dict, *, sensitive: bool = False) -> dict:
    """모델 설정에서 OpenRouter `provider` 객체를 만든다. fallback은 항상 끈다."""
    model = spec["model"]
    if _FORBIDDEN_MODEL_PATTERNS.search(model):
        raise RoutingConfigError(f"실행 대상이 바뀔 수 있는 모델 별칭은 쓸 수 없습니다: {model}")
    if not spec.get("endpoint"):
        raise RoutingConfigError(f"{model}: provider endpoint를 고정해야 합니다.")
    block = {
        "order": [spec["endpoint"]],
        "only": [spec["endpoint"]],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    if spec.get("quantization"):
        block["quantizations"] = [spec["quantization"]]
    if sensitive:
        # 민감 데이터 평가 모드 — 적격 endpoint가 없으면 OpenRouter가 요청을 거부한다(= 실패 처리)
        block["data_collection"] = "deny"
        block["zdr"] = True
    return block


def extra_body(spec: dict, *, sensitive: bool = False) -> dict:
    """chat.completions 요청에 덧붙일 OpenRouter 전용 필드."""
    body = {
        "provider": provider_block(spec, sensitive=sensitive),
        # 8K 이하 context endpoint는 기본으로 압축이 켜진다 — 명시적으로 끈다
        "plugins": [{"id": "context-compression", "enabled": False}],
    }
    if spec.get("reasoning") is not None:
        body["reasoning"] = spec["reasoning"]
    return body


def sampling_params(spec: dict, temperature: float | None) -> dict:
    """endpoint가 지원하는 파라미터만 보낸다(require_parameters로 미지원 파라미터가 있으면 라우팅 실패)."""
    return {"temperature": temperature} if spec.get("supports_temperature") and temperature is not None else {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def route_matches(spec: dict, actual_model: str | None, actual_provider: str | None) -> tuple[bool, str]:
    """요청한 모델·endpoint로 실제 처리됐는지. 반환 (일치 여부, 불일치 사유)."""
    want_model = spec["model"]
    if actual_model and not _norm(actual_model).startswith(_norm(want_model.split(":")[0])):
        return False, f"model mismatch: requested={want_model} actual={actual_model}"
    want_provider = spec["endpoint"].split("/")[0]
    if actual_provider and not _norm(actual_provider).startswith(_norm(want_provider)):
        return False, f"provider mismatch: requested={spec['endpoint']} actual={actual_provider}"
    if not actual_provider:
        return False, "provider unknown (응답·generation 조회 모두에서 확인 불가)"
    return True, ""
