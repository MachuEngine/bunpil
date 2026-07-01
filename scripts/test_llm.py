#!/usr/bin/env python
"""Phase 2 LLM 추상화 레이어 검증: Ollama 로컬 → 응답 수신, 백엔드 전환 확인."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.common.llm import PromptTemplate, get_llm_backend

# 모듈별 주입 예시: 사회 교과 전문가 Few-shot + CoT 템플릿
SOCIAL_TEMPLATE = PromptTemplate(
    system="당신은 한국 고등학교 사회 교과 전문가입니다. 질문에 정확하고 간결하게 답하세요.",
    few_shots=[
        {
            "user": "민주주의의 핵심 원리 3가지는?",
            "assistant": "국민 주권, 기본권 보장, 권력 분립입니다.",
        },
        {
            "user": "시장 실패의 대표적 원인은?",
            "assistant": "외부효과, 공공재, 정보 비대칭, 독과점입니다.",
        },
    ],
    cot_prefix="단계적으로 생각해 보겠습니다.\n",
)


async def main():
    llm = get_llm_backend()
    print(f"[백엔드] {llm.__class__.__name__} | backend={os.getenv('LLM_BACKEND')}")

    messages = SOCIAL_TEMPLATE.build("사회계약론을 주장한 사상가 2명과 각각의 핵심 주장은?")
    print(f"[메시지 수] {len(messages)}개 (system 1 + few-shot {len(SOCIAL_TEMPLATE.few_shots)*2} + user 1 + cot 1)\n")
    print("[생성 중...]")
    response = await llm.generate(messages)
    print(f"[응답]\n{response}\n")
    print("\n[완료] Phase 2 LLM 추상화 레이어 검증 통과")


asyncio.run(main())
