import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    logger.info("LangSmith tracing enabled (project: %s)", os.getenv("LANGCHAIN_PROJECT", "default"))

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="분필 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 예시 문제 붙여넣기 최대 길이. 현재 스택(Qwen2.5-7B, 32K 네이티브 context) 기준.
MAX_PASSAGE_LENGTH = 8000


def _parse_standards(standards: str) -> list:
    std_list = [s.strip() for s in standards.splitlines() if s.strip()]
    if not std_list and standards:
        std_list = [s.strip() for s in standards.split(",") if s.strip()]
    return std_list


def _build_spec(passage_text: str, standards: str):
    """예시 문제 입력을 길이 제한에 맞게 잘라 ExamSpec으로 구성한다. (spec, truncated) 반환."""
    truncated = len(passage_text) > MAX_PASSAGE_LENGTH
    text = passage_text[:MAX_PASSAGE_LENGTH] if truncated else passage_text
    spec = {
        "passage_text": text,
        "standards": _parse_standards(standards),
    }
    return spec, truncated


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
    import traceback

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
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"status": "error", "msg": str(e), "detail": traceback.format_exc()},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    future = loop.run_in_executor(None, worker)
    try:
        while True:
            item = await queue.get()
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
):
    """예시 문제 텍스트를 받아 SSE로 진행 상황과 결과를 스트리밍한다."""

    spec, truncated = _build_spec(passage_text, standards)

    async def generate():
        def evt(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            if truncated:
                yield evt({"status": "truncated", "msg": "입력이 길어 앞부분만 반영되었습니다."})

            async for event in _run_exam_events(spec):
                if event.get("status") == "done":
                    event = {**event, "truncated": truncated}
                yield evt(event)

        except Exception as e:
            logger.exception("/exam/stream 오류")
            import traceback
            yield evt({"status": "error", "msg": str(e), "detail": traceback.format_exc()})

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
):
    spec, truncated = _build_spec(passage_text, standards)
    result = await _run_exam(spec)
    return {"truncated": truncated, **result}


# ── 생기부 윤문 ──────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    memo: str


@app.post("/record")
async def record(req: RecordRequest):
    from app.modules.record import get_record_chain
    chain = get_record_chain()
    result = await chain.run(req.memo)
    return result

