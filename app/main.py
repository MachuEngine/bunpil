import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# 사용자 입력 비저장 하드룰: 실제 요청을 처리하는 서버에서는 LangSmith가
# 프롬프트·응답을 외부에 기록하지 못하도록 환경 설정과 무관하게 차단한다.
# 합성 데이터 평가 스크립트는 각 진입점에서 tracing을 별도로 초기화한다.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="분필 API", version="0.1.0")

MAX_REQUEST_BYTES = 64 * 1024
MAX_MEMO_LENGTH = 4000
_REQUEST_SLOTS = asyncio.Semaphore(2)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            too_large = int(content_length) > MAX_REQUEST_BYTES
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

# 예시 문제 붙여넣기 최대 길이. 현재 스택(Qwen2.5-7B, 32K 네이티브 context) 기준.
MAX_PASSAGE_LENGTH = 8000


def _parse_standards(standards: str) -> list:
    std_list = [s.strip() for s in standards.splitlines() if s.strip()]
    if not std_list and standards:
        std_list = [s.strip() for s in standards.split(",") if s.strip()]
    return std_list


DEFAULT_NUM_ITEMS = 5


async def _extract_num_items(passage_text: str) -> int:
    """passage_text 안에 교사가 명시적으로 요청한 문항 개수가 있으면 그 값을,
    없으면 기본값(DEFAULT_NUM_ITEMS)을 반환한다. 예시 문제 자체의 문항 개수와는 무관 —
    "5문제 만들어줘" 같은 지시문이 같은 텍스트에 섞여 들어올 수 있어 정규식이 아니라
    LLM 판단으로 추출한다."""
    from app.common.llm import get_llm_backend

    messages = [
        {
            "role": "system",
            "content": (
                "다음은 교사가 문항 생성 서비스에 입력한 텍스트입니다. "
                "이 텍스트에서 교사가 명시적으로 요청한 생성 문항 개수를 찾으세요. "
                "명시적인 개수 요청이 있으면 그 숫자만 응답하고, 없으면 5라고만 응답하세요. "
                "설명 없이 숫자만 응답하세요."
            ),
        },
        {"role": "user", "content": passage_text[:2000]},
    ]
    try:
        raw = await get_llm_backend().generate(messages)
        digits = "".join(ch for ch in raw if ch.isdigit())
        n = int(digits) if digits else DEFAULT_NUM_ITEMS
    except Exception:
        n = DEFAULT_NUM_ITEMS
    return max(1, min(n, 20))  # 폭주 생성 방지


async def _build_spec(passage_text: str, standards: str):
    """예시 문제를 길이 제한·PII 마스킹 후 ExamSpec으로 구성한다."""
    from app.common.privacy import mask_pii

    truncated = len(passage_text) > MAX_PASSAGE_LENGTH
    text = passage_text[:MAX_PASSAGE_LENGTH] if truncated else passage_text
    masked_text, passage_pii = mask_pii(text)
    masked_standards, standards_pii = mask_pii(standards)
    pii_found = list(dict.fromkeys([*passage_pii, *standards_pii]))
    spec = {
        "passage_text": masked_text,
        "standards": _parse_standards(masked_standards),
        "num_items": await _extract_num_items(masked_text),
    }
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
    "validate": "생성된 문항의 구조적 유사도를 검증하고 있습니다...",
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
    standards: str = Form(""),
    _: None = Depends(verify_api_key),
):
    """예시 문제 텍스트를 받아 SSE로 진행 상황과 결과를 스트리밍한다."""

    spec, truncated, pii_found = await _build_spec(passage_text, standards)

    await _acquire_request_slot()

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

            async for event in _run_exam_events(spec):
                if event.get("status") == "done":
                    event = {
                        **event,
                        "truncated": truncated,
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


# ── 기존 JSON 엔드포인트 (하위 호환) ────────────────────────────────────

@app.post("/exam")
async def exam(
    passage_text: str = Form(...),
    standards: str = Form(""),
    _: None = Depends(verify_api_key),
):
    spec, truncated, pii_found = await _build_spec(passage_text, standards)
    async with request_slot():
        result = await _run_exam(spec)
    return {"truncated": truncated, "pii_found": pii_found, **result}


# ── 생기부 윤문 ──────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    memo: str = Field(min_length=1, max_length=MAX_MEMO_LENGTH)


@app.post("/record")
async def record(req: RecordRequest, _: None = Depends(verify_api_key)):
    from app.modules.record import get_record_chain
    chain = get_record_chain()
    async with request_slot():
        result = await chain.run(req.memo)
    return result
