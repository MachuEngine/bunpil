import os

from langchain_ollama import ChatOllama


def get_langchain_model():
    """LangGraph ReAct 에이전트용 LangChain 호환 LLM을 반환한다."""
    backend = os.getenv("LLM_BACKEND", "local")
    if backend == "local":
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        )
    # RunPod 어댑터는 Phase 8에서 구현
    raise NotImplementedError(f"LangChain adapter for '{backend}': Phase 8에서 구현")
