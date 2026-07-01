import os

from .backends.ollama import OllamaBackend
from .backends.runpod import RunPodBackend
from .base import LLMBackend


def get_llm_backend() -> LLMBackend:
    backend = os.getenv("LLM_BACKEND", "local")
    if backend == "runpod":
        return RunPodBackend()
    return OllamaBackend()


def get_judge_backend() -> LLMBackend:
    judge_model = os.getenv("OLLAMA_JUDGE_MODEL")
    if judge_model:
        return OllamaBackend(model=judge_model)
    return OllamaBackend()  # OLLAMA_MODEL 폴백
