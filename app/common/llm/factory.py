import os

from .backends.ollama import OllamaBackend
from .backends.openai import OpenAIBackend
from .backends.openai_vlm import OpenAIVLMBackend
from .backends.runpod import RunPodBackend
from .base import LLMBackend, VLMBackend


def get_llm_backend() -> LLMBackend:
    # 새 백엔드(local 이외) 추가 시 app/common/llm/tracing.py의 _PROD_BACKENDS도 확인 —
    # 거기서 실제 서빙 백엔드인지(dev/prod LangSmith 분기 기준)를 별도로 판단한다.
    backend = os.getenv("LLM_BACKEND", "local")
    if backend == "runpod":
        return RunPodBackend()
    if backend == "openai":
        return OpenAIBackend()
    if backend != "local":
        raise ValueError(
            f"LLM_BACKEND={backend!r}은 지원하지 않는 값입니다 (local|runpod|openai)."
        )
    return OllamaBackend()


def get_judge_backend() -> LLMBackend:
    # LLM_BACKEND(생성용)와 독립 — Judge만 별도로 OpenAI 등으로 바꿔보고 싶을 때 사용.
    # 미설정 시 로컬 개발 편의를 위해 Ollama로 폴백한다(OLLAMA_JUDGE_MODEL, 폴백 OLLAMA_MODEL) —
    # 프로덕션은 .env.example이 JUDGE_BACKEND=openai를 명시값으로 요구하므로 이 폴백을 안 탄다.
    # 단, 오타 등으로 인식 못 하는 값이 명시적으로 들어오면(예: "opneai") 조용히 local로
    # 새지 않고 그 자리에서 실패한다 — judge.py의 fail-fast 철학(신뢰도 검증 안 된 채
    # 게이트를 통과시키지 않는다)과 동일한 이유.
    judge_backend = os.getenv("JUDGE_BACKEND", "local")
    if judge_backend == "openai":
        return OpenAIBackend(model=os.getenv("OPENAI_JUDGE_MODEL"))
    if judge_backend != "local":
        raise ValueError(
            f"JUDGE_BACKEND={judge_backend!r}은 지원하지 않는 값입니다 (local 또는 openai)."
        )
    judge_model = os.getenv("OLLAMA_JUDGE_MODEL")
    if judge_model:
        return OllamaBackend(model=judge_model)
    return OllamaBackend()  # OLLAMA_MODEL 폴백


def get_vlm_backend() -> VLMBackend:
    # 2026-08-19: 이미지 → 텍스트 추출 전용(/exam/extract). 생성(LLM_BACKEND)·Judge
    # (JUDGE_BACKEND)와 완전히 독립된 세 번째 축. OpenAIVLMBackend는 langchain_openai를
    # 쓰지 않고 openai SDK를 직접 호출해 LangChain Runnable이 아니다 — 마스킹 전
    # 원본 이미지·VLM 원문이 LangSmith로 새지 않도록 트레이싱을 원천 차단하기 위함
    # (자세한 이유는 backends/openai_vlm.py 모듈 docstring). tracing.py의
    # _PROD_BACKENDS는 LLM_BACKEND 기준 LangSmith 프로젝트명(dev/prod) 분기용일 뿐이라
    # 이 함수와 무관하다.
    # local/runpod VLM 경로는 요청받은 적이 없어 만들지 않았다 — 필요해지면 그때 추가.
    backend = os.getenv("VLM_BACKEND", "openai")
    if backend != "openai":
        raise ValueError(f"VLM_BACKEND={backend!r}은 지원하지 않는 값입니다 (openai만 지원).")
    return OpenAIVLMBackend()
