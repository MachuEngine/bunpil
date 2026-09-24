"""교사의 수정 요청으로 생성된 문항 일부를 고친다 — 챗봇 수정(2026-09 기능 추가).

그래프 밖의 단발 호출이다. 서버는 문항·대화를 저장하지 않으므로(하드룰 3) 브라우저가
매 요청에 예시 문제·현재 문항·최근 대화를 함께 보내고, 호출부(main.py)가 **전부
mask_pii()를 거친 값**만 여기로 넘긴다(하드룰 2).

모델은 바뀐 문항만 돌려준다. 바뀐 문항은 생성 경로와 같은 결정론적 게이트
(형식·한국어·예시 복사·세트 내 중복)로 다시 검사한다. 단, "예시 형식을 따르라"는
규칙(선지 수·<보기> 필수·합답형)은 적용하지 않는다 — 교사가 "4지로 바꿔줘"처럼
예시와 다른 형식을 일부러 요청할 수 있기 때문이다. 대신 수정 요청에 "합답형"·"<보기>"가
있으면 그 형식 규칙을 적용한다(교사가 명시한 형식). 같은 이유로 Judge 재채점도 하지 않는다.
"""
import json

from langsmith import traceable

from app.common.llm import get_llm_backend

from .tools import _check_korean, _check_similarity, _format_errors, init_session, _get_ctx

_MAX_TOKENS = 2048
_EDIT_FIELDS = ("question", "stimulus", "options", "answer", "item_type", "difficulty")

_SYSTEM = (
    "당신은 한국 고등학교 사회 문항 출제를 돕는 조수입니다. 교사가 이미 만들어진 문항 세트의 "
    "일부를 고쳐 달라고 요청합니다. 한국어로만 답하세요.\n"
    "반드시 아래 JSON 하나만 출력하세요. 다른 텍스트는 쓰지 마세요.\n"
    '{"message": "교사에게 할 짧은 답변", "changes": [{"number": 문항번호, "question": "...", '
    '"stimulus": "...", "options": ["①...", ...], "answer": "①", "item_type": "객관식", "difficulty": "중"}]}\n'
    "- changes에는 **실제로 바꾼 문항만** 넣고, 바꾼 문항은 모든 필드를 빠짐없이 쓰세요.\n"
    "- 요청받지 않은 문항은 넣지 마세요.\n"
    "- 객관식 선지는 4개 또는 5개이고 ①②③④⑤ 기호로 시작합니다. answer는 정답 선지 기호 하나입니다.\n"
    "- 서술형은 options를 []로, answer는 예시 답안으로 쓰세요.\n"
    "- stimulus는 <보기>·자료 제시문이고, 없으면 \"\"입니다. 합답형 선지 기호는 <보기>에 있어야 합니다.\n"
    "- 정답이 하나만 되도록 하고, 오답도 같은 개념 범주의 그럴듯한 선지로 쓰세요.\n"
    "- 문항을 새로 추가하거나 세트 전체를 새로 만들어 달라는 요청이면 changes를 []로 두고, "
    "message에 '새로 생성' 기능을 쓰라고 안내하세요."
)


def _format_items(items: list) -> str:
    blocks = []
    for n, item in enumerate(items, 1):
        lines = [f"[{n}번] ({item['item_type']}/{item['difficulty']}) {item['question']}"]
        if item.get("stimulus"):
            lines.append(item["stimulus"])
        lines.extend(item["options"])
        if item.get("answer"):
            lines.append(f"정답: {item['answer']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_response(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}") + 1
    try:
        data = json.loads(raw[s:e]) if s >= 0 and e > s else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def _requested_format(instruction: str) -> tuple[bool, bool]:
    """수정 요청이 명시한 형식. 반환: (<보기> 필수, 합답형 선지).

    14B는 "<보기> 합답형으로 바꿔줘"에 <보기>에 선지를 복사하고 선지는 서술문으로 두는
    문항을 돌려줬다(2026-09 실측) — 교사가 형식을 명시하면 생성 경로와 같은 규칙으로 검사한다."""
    combo = "합답형" in instruction
    return combo or "<보기>" in instruction, combo


def _check_change(change: dict, items: list, needs_stimulus: bool = False, combo: bool = False) -> tuple[int, dict, list[str]]:
    """바뀐 문항 하나를 원본에 병합하고 게이트로 검사한다. 반환: (0-based 위치, 병합된 문항, 오류)."""
    number = change.get("number")
    if not isinstance(number, int) or not 1 <= number <= len(items):
        return -1, {}, [f"없는 문항 번호입니다: {number}"]
    index = number - 1
    merged = {**items[index], **{k: change[k] for k in _EDIT_FIELDS if k in change}}
    if not isinstance(merged["options"], list) or not all(isinstance(v, str) for k, v in merged.items() if k != "options"):
        return index, merged, ["필드 형식이 올바르지 않습니다"]

    options = merged["options"]
    errors = _format_errors(
        merged["question"], options, merged["answer"], merged["item_type"],
        len(options) if len(options) in (4, 5) else 4,
        merged["stimulus"], needs_stimulus, combo,
    )
    if merged["difficulty"] not in ("상", "중", "하"):
        errors.append("난이도는 상·중·하 중 하나여야 합니다")
    if not errors:
        # 세트 내 중복 검사는 나머지 문항과만 비교한다
        _get_ctx()["items"] = [it for i, it in enumerate(items) if i != index]
        rejection = (
            _check_korean(merged["question"], options, merged["answer"], merged["stimulus"])
            or _check_similarity(merged["question"], merged["stimulus"])
        )
        if rejection:
            errors.append(rejection)
    return index, merged, errors


@traceable(name="revise_items", run_type="chain")
async def revise_items(passage_text: str, items: list, history: list, instruction: str) -> dict:
    """마스킹된 입력으로 수정 요청을 처리한다.

    반환: {"message": str, "changes": [{"number": int, "item": dict}]}.
    게이트를 통과하지 못한 변경은 한 번 오류를 알려 다시 쓰게 하고, 그래도 실패하면 버리고
    원본을 유지한다. 호출 실패는 그대로 전파한다."""
    init_session(passage_text)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"[예시 문제]\n{passage_text}\n\n[현재 문항 세트]\n{_format_items(items)}"},
        *history,
        {"role": "user", "content": instruction},
    ]
    backend = get_llm_backend()
    needs_stimulus, combo = _requested_format(instruction)

    accepted: dict[int, dict] = {}
    rejected: dict[int, list[str]] = {}
    message = ""
    for attempt in range(2):
        raw = await backend.generate(messages, max_tokens=_MAX_TOKENS)
        data = _parse_response(raw)
        if not data and attempt == 0:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "형식이 틀렸습니다. 설명 없이 지정한 JSON 하나만 다시 출력하세요."})
            continue
        message = str(data.get("message", "")).strip() or message
        rejected = {}
        for change in data.get("changes") or []:
            if not isinstance(change, dict):
                continue
            index, merged, errors = _check_change(change, items, needs_stimulus, combo)
            if errors:
                rejected[index] = errors
            else:
                accepted[index] = merged
                rejected.pop(index, None)
        if not rejected or attempt == 1:
            break
        messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
        messages.append({
            "role": "user",
            "content": "다음 문항은 형식 검사에 걸렸습니다. 오류를 고쳐 해당 문항만 다시 JSON으로 쓰세요.\n"
            + "\n".join(f"{i + 1}번: {' / '.join(errs)}" for i, errs in rejected.items() if i >= 0),
        })

    if rejected:
        numbers = ", ".join(f"{i + 1}번" for i in sorted(rejected) if i >= 0) or "일부 문항"
        note = f"{numbers}은 형식 검사를 통과하지 못해 원래 문항을 유지했습니다."
        # 반영된 변경이 없으면 모델의 "수정했습니다" 같은 답변은 사실과 달라 버린다
        message = f"{message}\n{note}" if message and accepted else note
    if not message:
        message = "요청을 반영했습니다." if accepted else "바꾼 문항이 없습니다."
    return {
        "message": message,
        "changes": [{"number": i + 1, "item": accepted[i]} for i in sorted(accepted)],
    }
