"""생성된 문항 하나의 정답 해설을 만든다 — '해설 보기' 버튼(2026-09 기능 추가).

그래프(plan→agent→judge→validate) 밖의 단발 호출이다. 서버는 문항을 저장하지 않으므로
(하드룰 3) 브라우저가 문항을 다시 보내고, 호출부(main.py)가 **mask_pii()를 거친 문항**만
여기로 넘긴다(하드룰 2).

정답은 다시 정하지 않는다 — 생성 단계에서 이미 정한 answer를 주고 그 근거만 쓰게 한다.
해설이 정답을 뒤집으면 교사 화면에 정답과 해설이 서로 다른 말을 하게 되기 때문이다.
"""
from langsmith import traceable

from app.common.llm import get_llm_backend

# RunPodBackend 기본 max_tokens(256)로는 선지별 해설이 중간에 잘린다.
_MAX_TOKENS = 1024

_SYSTEM = (
    "당신은 한국 고등학교 사회 교사입니다. 주어진 문항의 해설을 한국어로 작성하세요.\n"
    "- 정답은 이미 정해져 있습니다. 정답을 바꾸거나 의심하지 말고, 왜 정답인지 설명하세요.\n"
    "- 객관식은 정답의 근거를 먼저 쓰고, 이어서 나머지 선지가 틀린 이유를 선지 기호별로 한 줄씩 쓰세요.\n"
    "- <보기>가 있으면 각 진술(ㄱ, ㄴ, ...)이 옳은지 그른지와 그 이유를 먼저 쓰세요.\n"
    "- 서술형은 '예시 답안'과 '채점 요소'(핵심 개념 2~3개)를 쓰세요. 객관식에는 채점 요소를 쓰지 마세요.\n"
    "- 교육과정 수준을 벗어난 내용이나 확인할 수 없는 수치는 지어내지 마세요.\n"
    "- 인사말이나 머리말 없이 해설만 쓰세요."
)


def _format_item(item: dict) -> str:
    lines = [f"[유형] {item.get('item_type', '')}", f"[발문] {item.get('question', '')}"]
    if item.get("stimulus"):
        lines.append(f"[제시문]\n{item['stimulus']}")
    if item.get("options"):
        lines.append("[선지]\n" + "\n".join(item["options"]))
    if item.get("answer"):
        lines.append(f"[정답] {item['answer']}")
    return "\n".join(lines)


@traceable(name="explain_item", run_type="chain")
async def explain_item(item: dict) -> str:
    """마스킹된 문항 dict를 받아 해설 텍스트를 반환한다. 호출 실패는 그대로 전파한다."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _format_item(item)},
    ]
    text = await get_llm_backend().generate(messages, max_tokens=_MAX_TOKENS)
    return text.strip()
