"""app/common/llm/tracing.py init_langsmith_project()의 dev/prod 분기 단위 테스트. LLM·네트워크 없음."""
import os

from app.common.llm.tracing import init_langsmith_project


def test_local_backend_goes_to_dev(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "bunpil")  # 테스트 후 환경 원복용 등록
    monkeypatch.setenv("LLM_BACKEND", "local")
    init_langsmith_project()
    assert os.environ["LANGCHAIN_PROJECT"] == "bunpil-dev"


def test_openrouter_backend_goes_to_prod(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "bunpil")
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    init_langsmith_project()
    assert os.environ["LANGCHAIN_PROJECT"] == "bunpil-prod"


def test_custom_project_is_kept(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "custom-proj")
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    init_langsmith_project()
    assert os.environ["LANGCHAIN_PROJECT"] == "custom-proj"


def test_tracing_disabled_leaves_project_unchanged(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "bunpil")
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    init_langsmith_project()
    assert os.environ["LANGCHAIN_PROJECT"] == "bunpil"
