import os

from langchain_ollama import ChatOllama

from app.common.llm.backends.chat_runpod import ChatRunPod


def get_langchain_model():
    """LangGraph ReAct 에이전트용 LangChain 호환 LLM을 반환한다."""
    backend = os.getenv("LLM_BACKEND", "local")
    if backend == "local":
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            num_predict=2048,  # RunPod 백엔드(max_tokens=2048)와 동일 캡 — 미설정 시 폭주 생성 위험
        )
    if backend == "runpod":
        return ChatRunPod()
    raise NotImplementedError(f"지원하지 않는 LLM_BACKEND: '{backend}'")
