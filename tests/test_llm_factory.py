"""app/common/llm/factory.py — 백엔드 분기 로직. 네트워크 호출 없음(생성자만 확인).

2026-08-04: LLM_BACKEND/JUDGE_BACKEND에 오타 등 인식 못 하는 값이 들어오면
조용히 local로 새지 않고 그 자리에서 실패하는지 확인한다(fail-fast 회귀 방지).
"""
import pytest

from app.common.llm.backends.ollama import OllamaBackend
from app.common.llm.backends.openai import OpenAIBackend
from app.common.llm.backends.openai_vlm import OpenAIVLMBackend
from app.common.llm.backends.openrouter import OpenRouterBackend
from app.common.llm.backends.runpod import RunPodBackend
from app.common.llm.factory import get_judge_backend, get_llm_backend, get_vlm_backend


@pytest.mark.parametrize(
    "env_value,expected_cls",
    [
        (None, OllamaBackend),
        ("local", OllamaBackend),
        ("runpod", RunPodBackend),
        ("openai", OpenAIBackend),
        ("openrouter", OpenRouterBackend),
    ],
)
def test_get_llm_backend_recognized_values(monkeypatch, env_value, expected_cls):
    if env_value is None:
        monkeypatch.delenv("LLM_BACKEND", raising=False)
    else:
        monkeypatch.setenv("LLM_BACKEND", env_value)
    assert isinstance(get_llm_backend(), expected_cls)


def test_get_llm_backend_rejects_unrecognized_value(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "runpdo")  # 오타
    with pytest.raises(ValueError, match="runpdo"):
        get_llm_backend()


@pytest.mark.parametrize(
    "env_value,expected_cls",
    [
        (None, OllamaBackend),
        ("local", OllamaBackend),
        ("openai", OpenAIBackend),
        ("openrouter", OpenRouterBackend),
    ],
)
def test_get_judge_backend_recognized_values(monkeypatch, env_value, expected_cls):
    if env_value is None:
        monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("JUDGE_BACKEND", env_value)
    if env_value == "openrouter":
        # openrouter judge는 OPENROUTER_JUDGE_MODEL이 없으면 ValueError로 실패하므로
        # (생성 모델로 조용히 폴백 금지, 아래 별도 테스트) 여기서는 명시값을 세팅해 둔다.
        monkeypatch.setenv("OPENROUTER_JUDGE_MODEL", "dummy/judge-model")
    assert isinstance(get_judge_backend(), expected_cls)


def test_get_judge_backend_rejects_unrecognized_value(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "opneai")  # 오타
    with pytest.raises(ValueError, match="opneai"):
        get_judge_backend()


def test_get_judge_backend_openrouter_without_judge_model_fails(monkeypatch):
    # 생성 모델(OPENROUTER_MODEL)로 조용히 폴백하지 않고 즉시 실패해야 한다.
    monkeypatch.setenv("JUDGE_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-3.8-flash")
    monkeypatch.delenv("OPENROUTER_JUDGE_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_JUDGE_MODEL"):
        get_judge_backend()


def test_get_judge_backend_openrouter_with_judge_model_set(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("OPENROUTER_JUDGE_MODEL", "anthropic/claude-judge")
    backend = get_judge_backend()
    assert isinstance(backend, OpenRouterBackend)
    assert backend.model == "anthropic/claude-judge"


# ── 2026-08-19: get_vlm_backend() — /exam/extract 전용 세 번째 축 ────────────


@pytest.mark.parametrize(
    "env_value,expected_cls",
    [(None, OpenAIVLMBackend), ("openai", OpenAIVLMBackend)],
)
def test_get_vlm_backend_recognized_values(monkeypatch, env_value, expected_cls):
    if env_value is None:
        monkeypatch.delenv("VLM_BACKEND", raising=False)
    else:
        monkeypatch.setenv("VLM_BACKEND", env_value)
    assert isinstance(get_vlm_backend(), expected_cls)


def test_get_vlm_backend_rejects_unrecognized_value(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "local")  # 아직 지원하지 않는 값
    with pytest.raises(ValueError, match="local"):
        get_vlm_backend()


# ── 2026-10: get_langchain_model() LLM_BACKEND=openrouter 분기 (평가 전용) ──────


def test_get_langchain_model_openrouter_returns_chat_openrouter_backend(monkeypatch):
    from app.common.llm.backends.chat_openrouter import ChatOpenRouterBackend
    from app.modules.exam.llm import get_langchain_model

    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy_key")
    monkeypatch.setenv("OPENROUTER_MODEL", "dummy/model")
    assert isinstance(get_langchain_model(), ChatOpenRouterBackend)


def test_get_langchain_model_openrouter_requires_api_key(monkeypatch):
    from app.modules.exam.llm import get_langchain_model

    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "dummy/model")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_langchain_model()
