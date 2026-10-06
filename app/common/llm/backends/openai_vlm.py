"""OpenAI Vision 백엔드 (VLMBackend 인터페이스).

2026-08-19: 시험 문제 캡처 이미지에서 텍스트를 추출하는 전용 경로.

⚠️ langchain_openai.ChatOpenAI(LangChain `Runnable`)를 의도적으로 쓰지 않고, 공식
`openai` SDK(`AsyncOpenAI`)를 직접 호출한다 — chat_openai.py의 ChatOpenAIBackend를
재사용하지 않는 것은 실수가 아니라 이 파일의 핵심 설계 결정이다.

이유: `LANGCHAIN_TRACING_V2=true`일 때 LangChain은 "그래프 안/밖"이 아니라
"호출 대상이 LangChain Runnable인가"를 기준으로 전역 트레이싱을 자동으로 건다
(app/common/llm/tracing.py, 환경변수 하나로 프로세스 전체에 걸리는 스위치).
이 경로는 **마스킹 전** 원본 이미지(base64)와 VLM이 반환한 마스킹 전 원문을 다루는데,
CLAUDE.md 하드룰 3의 승인된 LangSmith 예외는 "PII 마스킹 후"만 전제로 하므로, 만약
ChatOpenAI를 통해 이 호출이 트레이싱되면 마스킹 전 이미지·원문이 LangSmith로 나가
그 예외 범위를 벗어난다. OllamaBackend/RunPodBackend가 LangChain Runnable이 아니라서
트레이싱을 원천적으로 피하는 것(tests/test_exam_input_privacy.py의
test_plain_backends_are_not_langchain_traceable)과 동일한 이유로, 이 백엔드도
LangChain을 거치지 않는다.

2026-10: 아래 `OpenRouterVLMBackend`(VLM 모델 비교 평가 전용)도 같은 이유로 openai
SDK를 직접 호출한다 — `base_url`과 키만 OpenRouter용으로 바꾸고 메시지 구성(`_build_messages`)
과 `_EXTRACT_PROMPT`는 OpenAI 직접 경로와 공유하지만, 실제 요청 파라미터(`_call_vlm` vs
`_call_openai_vlm`)는 gpt-6-luna가 OpenAI 직접 호출에서만 거부하는 파라미터(아래 참고)
때문에 분리돼 있다.
"""
import base64
import os

from openai import AsyncOpenAI, BadRequestError

from ..base import VLMBackend

_EXTRACT_PROMPT = """당신은 사회 교사가 첨부한 시험 문제 이미지에서 텍스트를 추출하는 도구입니다. 아래 원칙을 지키세요.

- 발문, <보기>, 선지(①~⑤)를 원문 그대로 옮기세요. 오탈자가 있어도 그대로 옮기고 임의로 고치지 마세요.
- 표·그래프·지도 등 텍스트가 아닌 자료는 내용을 서술하여 "[자료: ...]" 형식으로 표기하세요. 사회탐구 문항은 자료 제시형 비중이 높아 이 서술이 빠지면 문항이 성립하지 않습니다.
- 요약하거나 해설을 덧붙이거나 정답을 추론하지 마세요. 추출만 하세요.
- 설명이나 마크다운 코드 펜스 없이, 추출된 문제 본문만 응답하세요."""


def _build_messages(image_bytes: bytes, mime_type: str) -> list[dict]:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return [
        {"role": "system", "content": _EXTRACT_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
            ],
        },
    ]


async def _call_vlm(client: AsyncOpenAI, model: str, image_bytes: bytes, mime_type: str) -> str:
    """OpenRouter VLM 전용 요청(기존 동작 유지 — 2026-10 VLM 비교 평가 때부터 검증된
    경로라 파라미터를 바꾸지 않는다. OpenAI 직접 경로는 아래 `_call_openai_vlm` 참고)."""
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2048,
        messages=_build_messages(image_bytes, mime_type),
    )
    return (response.choices[0].message.content or "").strip()


async def _call_openai_vlm(client: AsyncOpenAI, model: str, image_bytes: bytes, mime_type: str) -> str:
    """OpenAI 직접 경로 전용 요청.

    2026-10: gpt-6-luna로 전환하면서 실측된 두 가지 파라미터 차이 때문에 OpenRouter와
    요청을 분리했다 — ① `max_tokens`를 거부하고 `max_completion_tokens`를 요구함(이
    파라미터는 openai SDK 2.45의 chat.completions.create가 gpt-4o-mini 등 기존 모델도
    동일하게 지원하므로 모델 분기 없이 항상 사용 가능 — 2026-10-06 실호출로 gpt-4o-mini·
    gpt-6-luna 모두 max_completion_tokens로 성공 확인), ② `temperature`도 거부함.
    어떤 모델이 ②를 거부할지 모델명 패턴으로 미리 판별하는 규칙은 신모델이 나올 때마다
    깨지기 쉬워, 400 오류 메시지로 감지해 temperature 없이 1회만 재시도한다."""
    kwargs = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 2048,
        "messages": _build_messages(image_bytes, mime_type),
    }
    try:
        response = await client.chat.completions.create(**kwargs)
    except BadRequestError as exc:
        if "temperature" not in str(exc):
            raise
        kwargs.pop("temperature")
        response = await client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


class OpenAIVLMBackend(VLMBackend):
    # 2026-10: 기본 모델 gpt-4o-mini → gpt-6-luna. 그림 문항 CER 0.035(gpt-4o-mini
    # 0.143)·서술 실패율 0%(gpt-4o-mini 30%)로 우세(MODEL_SELECTION.md §7 2026-10 비교).
    # 이미지가 전달되는 대상은 그대로 OpenAI라 개인정보 전달 경로는 바뀌지 않는다.
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_VLM_MODEL", "gpt-6-luna")

    async def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        client = AsyncOpenAI(api_key=api_key)
        return await _call_openai_vlm(client, self.model, image_bytes, mime_type)


class OpenRouterVLMBackend(VLMBackend):
    """OpenRouter 경유 VLM — 2026-10 VLM 모델 비교 평가 전용(gpt-6-luna, gemini-3.8-flash 등).
    OpenAIVLMBackend와 동일하게 openai SDK를 직접 호출해 LangChain을 거치지 않는다
    (트레이싱 차단 이유는 이 파일 상단 docstring 참고)."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENROUTER_VLM_MODEL")
        if not self.model:
            raise RuntimeError(
                "OPENROUTER_VLM_MODEL 환경변수가 설정되지 않았습니다 — 평가 비교 목적상 "
                "기본 모델을 임의로 지정하지 않습니다."
            )

    async def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")

        client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        return await _call_vlm(client, self.model, image_bytes, mime_type)
