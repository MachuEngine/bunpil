"""app/common/llm/factory.py — 백엔드 분기 로직. 네트워크 호출 없음(생성자만 확인).

2026-08-04: LLM_BACKEND/JUDGE_BACKEND에 오타 등 인식 못 하는 값이 들어오면
조용히 local로 새지 않고 그 자리에서 실패하는지 확인한다(fail-fast 회귀 방지).
"""
import pytest

from app.common.llm.backends.ollama import OllamaBackend
from app.common.llm.backends.openai import OpenAIBackend
from app.common.llm.backends.openai_vlm import OpenAIVLMBackend
from app.common.llm.backends.runpod import RunPodBackend
from app.common.llm.factory import get_judge_backend, get_llm_backend, get_vlm_backend


@pytest.mark.parametrize(
    "env_value,expected_cls",
    [(None, OllamaBackend), ("local", OllamaBackend), ("runpod", RunPodBackend), ("openai", OpenAIBackend)],
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
    [(None, OllamaBackend), ("local", OllamaBackend), ("openai", OpenAIBackend)],
)
def test_get_judge_backend_recognized_values(monkeypatch, env_value, expected_cls):
    if env_value is None:
        monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("JUDGE_BACKEND", env_value)
    assert isinstance(get_judge_backend(), expected_cls)


def test_get_judge_backend_rejects_unrecognized_value(monkeypatch):
    monkeypatch.setenv("JUDGE_BACKEND", "opneai")  # 오타
    with pytest.raises(ValueError, match="opneai"):
        get_judge_backend()


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
