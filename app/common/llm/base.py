from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], **kwargs) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": "..."}]"""
        ...


class VLMBackend(ABC):
    """이미지 → 텍스트 추출 전용 인터페이스. LLMBackend와 별도 —
    생성 모델과 완전히 분리된 백엔드이며(2026-08-19), 그래프 밖(app/main.py의
    /exam/extract)에서만 호출된다."""

    @abstractmethod
    async def extract_text(self, image_bytes: bytes, mime_type: str) -> str:
        ...
