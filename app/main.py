import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# 사용자 입력 비저장 하드룰(CLAUDE.md 3번):
# 출제 모듈은 2026-07-24부터 하드룰 3의 예외: passage_text·생성 문항은 실존 인물 정보가
# 아니고 PII 마스킹(하드룰 2)도 LLM 호출 전에 이미 거치므로, .env의 LANGCHAIN_TRACING_V2를
# 그대로 존중해 프로덕션에서도 관측성을 확보한다(agent_node의 ChatOllama/ChatRunPod,
# judge_node 둘 다 트레이싱 대상 — 사용자 승인, 자세한 배경은 LANGSMITH_GUIDE.md 1절).
from app.common.llm.tracing import init_langsmith_project
init_langsmith_project()

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI(title="분필 API", version="0.1.0")

MAX_REQUEST_BYTES = 64 * 1024
# 2026-08-19: 이미지 업로드(/exam/extract)만 별도 한도. 스크린샷은 보통 300KB~2MB —
# 텍스트 경로(MAX_REQUEST_BYTES)는 그대로 두고 이 경로만 완화한다.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_IMAGE_UPLOAD_PATHS = {"/exam/extract"}
_REQUEST_SLOTS = asyncio.Semaphore(2)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    limit = MAX_IMAGE_BYTES if request.url.path in _IMAGE_UPLOAD_PATHS else MAX_REQUEST_BYTES
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            too_large = int(content_length) > limit
        except ValueError:
            return JSONResponse({"detail": "Content-Length 형식이 올바르지 않습니다."}, status_code=400)
        if too_large:
            return JSONResponse({"detail": "요청 본문이 너무 큽니다."}, status_code=413)
    return await call_next(request)


async def verify_api_key(x_bunpil_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("BUNPIL_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="서버 인증이 설정되지 않았습니다.")
    if not x_bunpil_api_key or not hmac.compare_digest(x_bunpil_api_key, expected):
        raise HTTPException(status_code=401, detail="인증에 실패했습니다.")


async def _acquire_request_slot() -> None:
    try:
        await asyncio.wait_for(_REQUEST_SLOTS.acquire(), timeout=0.05)
    except TimeoutError:
        raise HTTPException(status_code=429, detail="동시에 처리할 수 있는 요청 수를 초과했습니다.")


@asynccontextmanager
async def request_slot():
    await _acquire_request_slot()
    try:
        yield
    finally:
        _REQUEST_SLOTS.release()

# 예시 문제 붙여넣기 최대 길이. 현재 스택(Qwen2.5-14B, 32K 네이티브 context) 기준.
MAX_PASSAGE_LENGTH = 8000


DEFAULT_NUM_ITEMS = 2


# 지원 형식 5종 — 이 밖의 형식은 가장 가까운 지원 형식으로 만들고 교사에게 안내한다(2026-09-24 결정).
SUPPORTED_FORMATS = ("4지 선다", "5지 선다", "<보기> 합답형", "자료 제시형", "서술형")

_ANALYZE_PROMPT = (
    "다음은 교사가 문항 생성 서비스에 입력한 텍스트(예시 문제와 요청)입니다. 설명 없이 아래 JSON 하나만 출력하세요.\n"
    '{"requested_num_items": 정수 또는 null, "num_options": 4 또는 5, "has_stimulus": true/false, "combo": true/false, '
    '"has_essay": true/false, "unsupported": [문자열], "nearest": 문자열}\n'
    "- requested_num_items: 교사가 '3문제 만들어 줘'처럼 **명시적으로 요청한** 생성 문항 수. 요청 문장이 없으면 null. "
    "예시 문제의 문항 수를 세지 않는다. 범위로 요청하면(예: 3~5문제) 큰 값.\n"
    "- num_options: 예시 객관식의 선지 수. 선지 자리에 있는 기호(①~⑤)만 센다. 본문 속 '⑤번' 같은 언급은 세지 않는다. "
    "객관식이 없으면 4.\n"
    "- has_stimulus: 발문과 선지 사이에 <보기>, (가)·(나) 같은 자료, 표, 그래프·그림 설명, 제시문의 **내용이 실제로 "
    "적혀 있으면** true. '다음 자료의', '다음 지도에서'처럼 자료를 언급만 하고 내용이 없으면 false.\n"
    "- combo: 선지가 '① ㄱ, ㄴ'처럼 <보기> 기호의 조합이면 true.\n"
    "- has_essay: 답을 글로 쓰는 서술형 문항이 있으면 true.\n"
    f"- unsupported: 지원 형식({', '.join(SUPPORTED_FORMATS)})으로 나타낼 수 없는 예시 형식의 이름. "
    "예: OX, 빈칸 채우기, 연결형, 단답형. 선지가 순서 조합인 순서 배열 문항은 선다형이므로 넣지 않는다. "
    "사회 과목 문제가 아니면 '사회 과목 아님'을 넣는다. 없으면 [].\n"
    f"- nearest: unsupported가 있을 때 대신 만들 지원 형식({', '.join(SUPPORTED_FORMATS)} 중 하나). 없으면 \"\"."
)


def _parse_analysis(raw: str, masked_text: str) -> dict:
    """요청 분석 응답을 검증한다. 잘못된 필드는 필드별로 규칙 판정(rule_format)·기본값으로 대체한다.

    이전 `_extract_num_items`는 응답의 숫자를 모두 이어 붙여 "3~5문제"를 35(→20)로 읽었다 —
    이제 num_items는 JSON 정수 하나만 인정한다. 응답이 숫자 하나뿐이면 그 값을 쓴다(하위 호환)."""
    from app.modules.exam.revise import _parse_response
    from app.modules.exam.tools import rule_format

    rule = rule_format(masked_text)
    data = _parse_response(raw or "")
    if not data and (raw or "").strip().isdigit():
        data = {"num_items": int(raw.strip())}

    # 기본값은 코드가 채운다 — 모델에게 "요청이 없으면 2"를 맡겼더니 14B가 예시 문항 수(1)를 답했다(2026-09-24 실측)
    n = data.get("requested_num_items", data.get("num_items"))
    num_items = n if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 20 else DEFAULT_NUM_ITEMS
    fmt = {
        "num_options": data["num_options"] if data.get("num_options") in (4, 5) else rule["num_options"],
        **{k: data[k] if isinstance(data.get(k), bool) else rule[k] for k in ("has_stimulus", "combo", "has_essay")},
    }
    if fmt["combo"]:
        fmt["has_stimulus"] = True  # 합답형은 <보기>가 있어야 성립한다

    unsupported = data.get("unsupported")
    if isinstance(unsupported, str):
        unsupported = [unsupported]  # "OX"처럼 문자열 하나로 답해도 안내가 꺼지지 않게
    unsupported = [u.strip()[:30] for u in unsupported if isinstance(u, str) and u.strip()][:5] \
        if isinstance(unsupported, list) else []
    nearest = data.get("nearest") if data.get("nearest") in SUPPORTED_FORMATS else "4지 선다"
    notice = instruction = ""
    if unsupported:
        names = ", ".join(unsupported)
        notice = f"예시의 {names} 형식은 지원하지 않아 {nearest} 형식으로 만들었습니다."
        instruction = f"예시 중 {names} 형식은 지원하지 않으므로, 해당 문항은 {nearest} 형식으로 작성하세요."
    return {"num_items": num_items, "format": fmt, "format_notice": notice, "format_instruction": instruction}


async def _analyze_request(masked_text: str) -> dict:
    """마스킹된 입력에서 요청 문항 수와 예시 형식을 한 번의 LLM 호출로 판정한다(2026-09-24).

    이전의 문항 수 추출 호출을 확장한 것이라 요청당 LLM 호출 수는 그대로다. 형식 판정 결과는
    spec["format"] 하나에 담겨 저장 게이트(init_session)와 생성 프롬프트가 함께 쓴다. 호출이 실패하면
    전부 규칙 판정·기본값으로 대체한다 — 생성 자체는 막지 않는다."""
    from app.common.llm import get_llm_backend

    messages = [
        {"role": "system", "content": _ANALYZE_PROMPT},
        {"role": "user", "content": masked_text},
    ]
    try:
        raw = await get_llm_backend().generate(messages, max_tokens=300)
    except Exception:
        raw = ""
    return _parse_analysis(raw, masked_text)


async def _build_spec(passage_text: str):
    """예시 문제를 길이 제한·PII 마스킹 후 ExamSpec으로 구성한다."""
    from app.common.privacy import mask_pii

    truncated = len(passage_text) > MAX_PASSAGE_LENGTH
    text = passage_text[:MAX_PASSAGE_LENGTH] if truncated else passage_text
    masked_text, pii_found = mask_pii(text)
    spec = {"passage_text": masked_text, **await _analyze_request(masked_text)}
    return spec, truncated, pii_found


async def _run_exam(spec) -> dict:
    from app.modules.exam import get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    # graph.invoke가 실행될 스레드로 contextvars가 전파되도록,
    # to_thread 호출 전에 세션 dict를 먼저 만들어둔다.
    init_session()
    graph = get_exam_graph()
    state = await asyncio.to_thread(graph.invoke, {"spec": spec, "budget": 5})
    return {
        "items": get_draft_items(),
        "validation_passed": state.get("validation_passed", False),
    }


_NODE_MESSAGES = {
    "plan": "준비 중...",
    "judge": "생성된 문항의 구조적 유사도를 채점하고 있습니다...",
    "validate": "채점 결과가 기준을 통과했는지 확인하고 있습니다...",
}


async def _run_exam_events(spec):
    """그래프를 노드 단위(graph.stream)로 실행하며 진행 이벤트를 순서대로 yield한다.

    graph.stream()은 동기 제너레이터이고 각 노드는 LangGraph가 별도 context로
    격리 실행하므로, init_session()/get_draft_items() 호출도 같은 워커 스레드
    안에서 함께 끝내야 세션 dict가 올바르게 공유된다. 이 실행 전체를 단일
    executor 스레드에서 돌리고, 이벤트만 asyncio.Queue로 async 쪽에 전달한다.
    """
    from app.modules.exam import get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    DONE = object()

    def worker():
        try:
            init_session()
            graph = get_exam_graph()
            attempt = 0
            validation_passed = False
            for step in graph.stream({"spec": spec, "budget": 5}, stream_mode="updates"):
                for node_name, node_output in step.items():
                    if node_name == "agent":
                        attempt += 1
                        msg = (
                            "AI가 문항을 생성하고 있습니다. 수 분 소요됩니다..."
                            if attempt == 1
                            else f"문항 세트를 다시 생성하고 있습니다 ({attempt}번째 시도)..."
                        )
                    else:
                        msg = _NODE_MESSAGES.get(node_name, f"{node_name} 처리 중...")
                    if node_name == "validate":
                        validation_passed = node_output.get("validation_passed", False)
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"status": "progress", "msg": msg}
                    )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "status": "done",
                    "items": get_draft_items(),
                    "validation_passed": validation_passed,
                },
            )
        except Exception:
            logger.exception("출제 worker 오류")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"status": "error", "msg": "문항 생성 중 오류가 발생했습니다."},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    future = loop.run_in_executor(None, worker)
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                yield {"status": "heartbeat"}
                continue
            if item is DONE:
                break
            yield item
    finally:
        await future


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 문항 출제: SSE 스트리밍 ──────────────────────────────────────────────

@app.post("/exam/stream")
async def exam_stream(
    passage_text: str = Form(...),
    _: None = Depends(verify_api_key),
):
    """예시 문제 텍스트를 받아 SSE로 진행 상황과 결과를 스트리밍한다."""

    # 2026-08-04: 동시요청 제한이 그래프 실행뿐 아니라 _build_spec()의 num_items
    # 추출 LLM 호출까지 커버하도록 슬롯을 먼저 확보한다(이전엔 그래프 실행만 커버해
    # LLM 호출 총량이 세마포어(2)를 우회할 수 있었다).
    await _acquire_request_slot()
    try:
        spec, truncated, pii_found = await _build_spec(passage_text)
    except Exception:
        _REQUEST_SLOTS.release()
        raise

    async def generate():
        def evt(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            if pii_found:
                yield evt({
                    "status": "pii_masked",
                    "msg": "개인정보가 감지되어 모델 호출 전에 마스킹되었습니다.",
                    "pii_found": pii_found,
                })
            if truncated:
                yield evt({"status": "truncated", "msg": "입력이 길어 앞부분만 반영되었습니다."})
            if spec.get("format_notice"):
                yield evt({"status": "format_notice", "msg": spec["format_notice"]})

            async for event in _run_exam_events(spec):
                if event.get("status") == "done":
                    event = {
                        **event,
                        "truncated": truncated,
                        "format_notice": spec.get("format_notice", ""),
                        "pii_found": pii_found,
                    }
                yield evt(event)

        except Exception:
            logger.exception("/exam/stream 오류")
            yield evt({"status": "error", "msg": "문항 생성 중 오류가 발생했습니다."})
        finally:
            _REQUEST_SLOTS.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 이미지 → 텍스트 추출 (예시 문제 캡처 붙여넣기) ──────────────────────────
# 2026-08-19: passage_text를 텍스트로 직접 입력하는 대신 화면 캡처를 붙여넣을 수
# 있도록 추가. 추출된 텍스트는 기존 /exam/stream으로 그대로 흘러간다 — 그래프는
# 이미지를 모른다(app/modules/exam/graph.py 미수정).
#
# PII 마스킹 순서 예외(하드룰 2): 다른 모든 LLM 호출은 mask_pii() 이후에 이뤄지지만,
# 이 경로는 원본 이미지가 마스킹 전에 VLM으로 먼저 전달된다 — 이미지 자체를 마스킹할
# 방법이 없어 텍스트로 변환된 뒤에야 마스킹이 가능하기 때문이다(DESIGN.md 6절에
# 동일하게 문서화). 추출된 텍스트는 반환 전 반드시 mask_pii()를 거친다.
#
# 이 순서 예외가 LangSmith 트레이싱(하드룰 3의 승인된 예외, "마스킹 후"만 전제)까지
# 어기지 않도록, get_vlm_backend()(OpenAIVLMBackend)는 langchain_openai를 쓰지 않고
# openai SDK를 직접 호출한다 — LangChain Runnable이 아니므로 LANGCHAIN_TRACING_V2가
# 켜져 있어도 이 호출은 트레이싱되지 않는다(이유는 backends/openai_vlm.py 참고).
#
# 이미지 원본은 애플리케이션 코드가 디스크에 명시적으로 쓰지 않는다 — UploadFile.read()로
# 읽어 VLM 호출에만 쓰고 즉시 버린다(하드룰 3: 사용자 입력 비저장). 단, Starlette의
# multipart 파서가 1MB 초과 업로드를 내부적으로 SpooledTemporaryFile로 OS temp에
# 일시 스풀할 수 있다(starlette/formparsers.py, max_file_size=1MB) — 요청 종료 시
# 자동 삭제되는 임시 파일이며 애플리케이션이 관리하는 영구 경로에는 쓰이지 않는다.
_ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}


@app.post("/exam/extract")
async def exam_extract(
    image: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """시험 문제 캡처 이미지에서 텍스트를 추출해 마스킹 후 반환한다."""
    if image.content_type not in _ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=400, detail="지원하지 않는 이미지 형식입니다 (png/jpeg/webp만 허용)."
        )

    image_bytes = await image.read()

    # VLM 호출도 LLM 호출이므로 세마포어(2) 밖에 두지 않는다
    # (_build_spec의 num_items 추출을 슬롯 안으로 넣은 2026-08-04 변경과 같은 이유).
    async with request_slot():
        try:
            from app.common.llm import get_vlm_backend
            from app.common.privacy import mask_pii

            raw_text = await get_vlm_backend().extract_text(image_bytes, image.content_type)
            masked_text, pii_found = mask_pii(raw_text)
        except Exception:
            logger.exception("/exam/extract 오류")
            raise HTTPException(status_code=502, detail="이미지에서 문제를 읽지 못했습니다.")

    return {"text": masked_text, "pii_found": pii_found}


# ── 생성 문항을 다시 받는 경로 공통 (해설 /exam/explain, 수정 /exam/revise) ─────────────────────
# 서버는 생성 문항을 저장하지 않는다(하드룰 3) — 해설·수정은 브라우저가 문항을 다시
# 보내는 방식이다. 브라우저가 보낸 값은 사용자가 편집할 수 있는 입력이므로 형태를 검증하고,
# 모델 호출 전에 문자열 필드 전부를 mask_pii()로 마스킹한다(하드룰 2).
_MAX_ITEM_FIELD_LENGTH = 3000
_ITEM_STR_FIELDS = ("question", "stimulus", "answer", "item_type", "difficulty", "standard")


def _parse_item(raw: dict) -> dict:
    """브라우저가 보낸 문항 dict를 검증해 필요한 필드만 남긴다. 형태가 틀리면 400."""
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="문항 형식이 올바르지 않습니다.")
    item = {k: raw.get(k, "") for k in _ITEM_STR_FIELDS}
    item["options"] = raw.get("options", [])
    if (
        not all(isinstance(item[k], str) and len(item[k]) <= _MAX_ITEM_FIELD_LENGTH for k in _ITEM_STR_FIELDS)
        or not isinstance(item["options"], list)
        or len(item["options"]) > 5
        or not all(isinstance(o, str) and len(o) <= _MAX_ITEM_FIELD_LENGTH for o in item["options"])
        or item["item_type"] not in ("객관식", "서술형")
        or not item["question"].strip()
    ):
        raise HTTPException(status_code=400, detail="문항 형식이 올바르지 않습니다.")
    return item


def _mask_item(item: dict) -> tuple[dict, list[str]]:
    """문항의 모든 문자열 필드를 마스킹한다. 반환: (마스킹된 문항, 발견된 PII 유형)."""
    from app.common.privacy import mask_pii

    found: list[str] = []

    def mask(text: str) -> str:
        masked, pii = mask_pii(text)
        found.extend(p for p in pii if p not in found)
        return masked

    masked = {k: mask(item[k]) for k in _ITEM_STR_FIELDS}
    masked["options"] = [mask(o) for o in item["options"]]
    return masked, found


def _load_json_field(raw: str, name: str):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{name} 형식이 올바르지 않습니다.")


# ── 해설 보기 ─────────────────────────────────────────────────────────────
# 2026-09: '해설 보기' 버튼을 누를 때만 생성한다(생성 에이전트의 도구 인자·턴 부담을 늘리지
# 않기 위함). 그래프 밖의 단발 호출이다 — app/modules/exam/explain.py.

@app.post("/exam/explain")
async def exam_explain(
    item: str = Form(...),
    _: None = Depends(verify_api_key),
):
    """생성된 문항 하나(JSON 문자열)를 받아 마스킹 후 해설을 생성한다."""
    parsed = _parse_item(_load_json_field(item, "문항"))
    masked_item, pii_found = _mask_item(parsed)

    async with request_slot():
        try:
            from app.modules.exam.explain import explain_item

            explanation = await explain_item(masked_item)
        except Exception:
            logger.exception("/exam/explain 오류")
            raise HTTPException(status_code=502, detail="해설을 생성하지 못했습니다.")

    return {"explanation": explanation, "pii_found": pii_found}


# ── 챗봇 수정 ─────────────────────────────────────────────────────────────
# 2026-09: 생성된 문항 일부를 대화로 고친다. 서버는 대화·문항을 저장하지 않는다(하드룰 3) —
# 브라우저가 매 요청에 예시 문제·현재 문항·최근 대화를 함께 보내고, 여기서 전부 마스킹한 뒤
# 모델에 넘긴다(하드룰 2). 수정 요청 문장은 마스킹 후 트레이싱을 허용한다(CLAUDE.md 하드룰 3 예외).
_MAX_REVISE_ITEMS = 20
_MAX_HISTORY_TURNS = 6
_MAX_CHAT_LENGTH = 1000


@app.post("/exam/revise")
async def exam_revise(
    passage_text: str = Form(...),
    items: str = Form(...),
    instruction: str = Form(...),
    history: str = Form("[]"),
    _: None = Depends(verify_api_key),
):
    """현재 문항 세트와 수정 요청을 받아 바뀐 문항만 돌려준다."""
    from app.common.privacy import mask_pii

    raw_items = _load_json_field(items, "문항")
    raw_history = _load_json_field(history, "대화")
    if (
        not isinstance(raw_items, list)
        or not 1 <= len(raw_items) <= _MAX_REVISE_ITEMS
        or not isinstance(raw_history, list)
        or len(raw_history) > _MAX_HISTORY_TURNS
        or not all(
            isinstance(t, dict)
            and t.get("role") in ("user", "assistant")
            and isinstance(t.get("content"), str)
            and len(t["content"]) <= _MAX_CHAT_LENGTH
            for t in raw_history
        )
        or not instruction.strip()
        or len(instruction) > _MAX_CHAT_LENGTH
    ):
        raise HTTPException(status_code=400, detail="수정 요청 형식이 올바르지 않습니다.")

    pii_found: list[str] = []

    def mask(text: str) -> str:
        masked, pii = mask_pii(text)
        pii_found.extend(p for p in pii if p not in pii_found)
        return masked

    masked_items = []
    for raw in raw_items:
        masked_item, pii = _mask_item(_parse_item(raw))
        pii_found.extend(p for p in pii if p not in pii_found)
        masked_items.append(masked_item)
    masked_passage = mask(passage_text[:MAX_PASSAGE_LENGTH])
    masked_history = [{"role": t["role"], "content": mask(t["content"])} for t in raw_history]
    masked_instruction = mask(instruction)

    async with request_slot():
        try:
            from app.modules.exam.revise import revise_items

            result = await revise_items(masked_passage, masked_items, masked_history, masked_instruction)
        except Exception:
            logger.exception("/exam/revise 오류")
            raise HTTPException(status_code=502, detail="수정 요청을 처리하지 못했습니다.")

    return {**result, "pii_found": pii_found}


# ── 기존 JSON 엔드포인트 (하위 호환) ────────────────────────────────────

@app.post("/exam")
async def exam(
    passage_text: str = Form(...),
    _: None = Depends(verify_api_key),
):
    # 2026-08-04: /exam/stream과 동일하게 (1) 슬롯을 num_items 추출 LLM 호출까지
    # 커버하도록 _build_spec()도 감싸고 (2) 예외를 잡아 로깅 후 graceful하게
    # 응답한다(이전엔 이 엔드포인트만 예외가 처리 안 된 500으로 그대로 새어나갔다).
    async with request_slot():
        try:
            spec, truncated, pii_found = await _build_spec(passage_text)
            result = await _run_exam(spec)
        except Exception:
            logger.exception("/exam 오류")
            return JSONResponse(
                {"status": "error", "msg": "문항 생성 중 오류가 발생했습니다."},
                status_code=500,
            )
    return {"truncated": truncated, "pii_found": pii_found, "format_notice": spec.get("format_notice", ""), **result}

