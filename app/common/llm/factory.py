import os

from .backends.ollama import OllamaBackend
from .backends.ollama_vlm import OllamaVLMBackend
from .backends.openai import OpenAIBackend
from .backends.openai_vlm import OpenAIVLMBackend, OpenRouterVLMBackend
from .backends.openrouter import OpenRouterBackend
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
    if backend == "openrouter":
        return OpenRouterBackend()  # 평가(모델 비교) 전용 — gemini-3.8-flash 등
    if backend != "local":
        raise ValueError(
            f"LLM_BACKEND={backend!r}은 지원하지 않는 값입니다 (local|runpod|openai|openrouter)."
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
    if judge_backend == "openrouter":
        # OpenRouterBackend(OpenAIBackend 상속)는 model=None이면 OPENROUTER_MODEL(생성 모델)로
        # 폴백한다 — openai 경로의 OPENAI_JUDGE_MODEL→OPENAI_MODEL 폴백과 같은 코드지만,
        # 평가에서 생성·Judge 모델을 분리하는 목적과 반대 결과가 나오므로 여기서는 미설정을
        # 조용히 넘기지 않고 즉시 실패시킨다.
        judge_model = os.getenv("OPENROUTER_JUDGE_MODEL")
        if not judge_model:
            raise ValueError(
                "OPENROUTER_JUDGE_MODEL이 설정되지 않았습니다 — "
                "JUDGE_BACKEND=openrouter에서 생성 모델(OPENROUTER_MODEL)로 조용히 "
                "폴백하지 않도록 명시값이 필요합니다(생성·Judge 모델 분리 원칙)."
            )
        return OpenRouterBackend(model=judge_model)  # 평가(모델 비교) 전용 — Claude/gemini Judge
    if judge_backend != "local":
        raise ValueError(
            f"JUDGE_BACKEND={judge_backend!r}은 지원하지 않는 값입니다 (local|openai|openrouter)."
        )
    judge_model = os.getenv("OLLAMA_JUDGE_MODEL")
    if judge_model:
        return OllamaBackend(model=judge_model)
    return OllamaBackend()  # OLLAMA_MODEL 폴백


def get_vlm_backend() -> VLMBackend:
    # 2026-08-19: 이미지 → 텍스트 추출 전용(/exam/extract). 생성(LLM_BACKEND)·Judge
    # (JUDGE_BACKEND)와 완전히 독립된 세 번째 축. 여기서 만드는 세 백엔드 모두
    # langchain_openai/LangChain Runnable을 쓰지 않는다 — 마스킹 전 원본 이미지·VLM
    # 원문이 LangSmith로 새지 않도록 트레이싱을 원천 차단하기 위함(자세한 이유는
    # backends/openai_vlm.py 모듈 docstring). tracing.py의 _PROD_BACKENDS는 LLM_BACKEND
    # 기준 LangSmith 프로젝트명(dev/prod) 분기용일 뿐이라 이 함수와 무관하다.
    # 2026-10: openrouter/local은 VLM 모델 비교 평가 전용으로 추가했다(runpod VLM 경로는
    # 여전히 요청받은 적이 없어 미구현).
    backend = os.getenv("VLM_BACKEND", "openai")
    if backend == "openrouter":
        return OpenRouterVLMBackend()
    if backend == "local":
        return OllamaVLMBackend()
    if backend != "openai":
        raise ValueError(
            f"VLM_BACKEND={backend!r}은 지원하지 않는 값입니다 (openai|openrouter|local)."
        )
    return OpenAIVLMBackend()
