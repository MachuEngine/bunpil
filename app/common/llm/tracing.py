"""LangSmith 프로젝트를 LLM_BACKEND에 따라 dev/prod로 자동 분기.

로컬(Ollama)과 프로덕션(RunPod) 트레이스가 같은 LangSmith 프로젝트로 섞이면,
로컬 개발 중 발생하는 노이즈(실험적 프롬프트 변경·재시도·모델 비교 실험 등)가
프로덕션 통계를 오염시킨다. .env에 LANGCHAIN_PROJECT를 정적으로 박아두고 사람이
환경마다 다르게 관리하는 방식은 실수로 같은 값이 배포될 위험이 있어, 대신
LLM_BACKEND 값을 보고 매 실행 시점에 코드가 자동으로 결정한다.

호출 시점: load_dotenv() 직후, LangChain 트레이스가 발생할 수 있는 모든
진입점(app/main.py, scripts/eval_*.py 등)에서 다른 로직보다 먼저 호출한다.
"""
import os


def init_langsmith_project() -> None:
    if os.getenv("LANGCHAIN_TRACING_V2") != "true":
        return
    base = os.getenv("LANGCHAIN_PROJECT", "bunpil")
    if base.endswith("-dev") or base.endswith("-prod"):
        return  # 이미 분기 처리됨 — 같은 프로세스에서 두 번 호출돼도 중복 접미사 방지
    backend = os.getenv("LLM_BACKEND", "local")
    # runpod만 실제 프로덕션 트래픽 — local/openai(모델 비교 실험 등)는 전부 dev로 취급
    suffix = "prod" if backend == "runpod" else "dev"
    os.environ["LANGCHAIN_PROJECT"] = f"{base}-{suffix}"
