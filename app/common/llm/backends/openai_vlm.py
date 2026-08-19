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
"""
import base64
import os

from openai import AsyncOpenAI

from ..base import VLMBackend

_EXTRACT_PROMPT = """당신은 사회 교사가 첨부한 시험 문제 이미지에서 텍스트를 추출하는 도구입니다. 아래 원칙을 지키세요.

- 발문, <보기>, 선지(①~⑤)를 원문 그대로 옮기세요. 오탈자가 있어도 그대로 옮기고 임의로 고치지 마세요.
- 표·그래프·지도 등 텍스트가 아닌 자료는 내용을 서술하여 "[자료: ...]" 형식으로 표기하세요. 사회탐구 문항은 자료 제시형 비중이 높아 이 서술이 빠지면 문항이 성립하지 않습니다.
- 요약하거나 해설을 덧붙이거나 정답을 추론하지 마세요. 추출만 하세요.
- 설명이나 마크다운 코드 펜스 없이, 추출된 문제 본문만 응답하세요."""


class OpenAIVLMBackend(VLMBackend):
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")

    async def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        client = AsyncOpenAI(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = await client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=2048,
            messages=[
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
            ],
        )
        return (response.choices[0].message.content or "").strip()
