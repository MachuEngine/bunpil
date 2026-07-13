import os

from .backends.ollama import OllamaBackend
from .backends.openai import OpenAIBackend
from .backends.runpod import RunPodBackend
from .base import LLMBackend


def get_llm_backend() -> LLMBackend:
    # 새 백엔드(local 이외) 추가 시 app/common/llm/tracing.py의 _PROD_BACKENDS도 확인 —
    # 거기서 실제 서빙 백엔드인지(dev/prod LangSmith 분기 기준)를 별도로 판단한다.
    backend = os.getenv("LLM_BACKEND", "local")
    if backend == "runpod":
        return RunPodBackend()
    if backend == "openai":
        return OpenAIBackend()
    return OllamaBackend()


def get_judge_backend() -> LLMBackend:
    judge_model = os.getenv("OLLAMA_JUDGE_MODEL")
    if judge_model:
        return OllamaBackend(model=judge_model)
    return OllamaBackend()  # OLLAMA_MODEL 폴백
