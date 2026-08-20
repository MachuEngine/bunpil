#!/usr/bin/env python
"""VLM(이미지 → 텍스트 추출) 배선 검증: 합성 이미지를 생성해 get_vlm_backend()로
실제 추출까지 되는지 확인한다. 추출 결과의 정확도(CER/WER 등)는 이 스크립트 범위가
아니며 evals/eval_vlm.py가 담당한다(scripts/test_llm.py와 evals/eval_exam.py의
관계와 동일한 역할 분담).

이미지는 이 스크립트가 PIL로 즉석에서 합성한다 — 실제 스크린샷 미사용(하드룰 1과
같은 이유), 디스크에도 쓰지 않는다(메모리 내 BytesIO).
"""
import asyncio
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from PIL import Image, ImageDraw, ImageFont

from app.common.llm import get_vlm_backend

# 한글 렌더링 가능한 폰트를 환경별로 탐색 — 못 찾으면 tofu(빈 네모)로 렌더링돼
# VLM이 아예 못 읽는 이미지가 만들어지므로, 조용히 넘어가지 않고 바로 실패한다.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux (fonts-noto-cjk)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:\\Windows\\Fonts\\malgun.ttf",  # Windows
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise RuntimeError(
        "한글 렌더링 가능한 폰트를 찾지 못했습니다. _FONT_CANDIDATES에 이 환경의 "
        "한글 폰트 경로를 추가하세요(예: fonts-noto-cjk 설치)."
    )


_QUESTION_LINES = [
    "1. 다음 <보기>의 정치 참여 형태에 해당하는 것은?",
    "",
    "<보기>",
    "시민들이 특정 법안에 반대하며 온라인 서명 운동을 벌였다.",
    "",
    "① 선거 참여   ② 청원   ③ 시민 불복종   ④ 로비   ⑤ 언론 투고",
]


def _make_synthetic_image() -> bytes:
    img = Image.new("RGB", (700, 260), "white")
    d = ImageDraw.Draw(img)
    font = _load_font(20)
    y = 15
    for line in _QUESTION_LINES:
        d.text((20, y), line, fill="black", font=font)
        y += 34
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def main():
    vlm = get_vlm_backend()
    print(f"[백엔드] {vlm.__class__.__name__} | model={vlm.model} | backend={os.getenv('VLM_BACKEND', 'openai')}")

    image_bytes = _make_synthetic_image()
    print(f"[합성 이미지] {len(image_bytes)} bytes (PIL 렌더링, 디스크 미저장)")

    print("[추출 중...]")
    text = await vlm.extract_text(image_bytes, "image/png")
    print(f"[추출 결과]\n{text}\n")

    if not text.strip():
        raise RuntimeError("VLM 추출 결과가 비어 있습니다.")
    if "보기" not in text:
        raise RuntimeError("추출 결과에 <보기>가 없습니다 — 원문 재현 실패로 의심됩니다.")
    if "①" not in text:
        raise RuntimeError("추출 결과에 선지(①)가 없습니다 — 원문 재현 실패로 의심됩니다.")

    print("[완료] VLM 배선 검증 통과 (정확도 측정은 evals/eval_vlm.py 참고)")


asyncio.run(main())
