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

    truncated = len(passage_text) > MAX_PASSAGE_LENGTH
    text = passage_text[:MAX_PASSAGE_LENGTH] if truncated else passage_text

    async def generate():
        from app.modules.exam import ExamSpec, get_exam_graph
        from app.modules.exam.tools import get_draft_items, init_session

        def evt(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            if truncated:
                yield evt({"status": "truncated", "msg": "입력이 길어 앞부분만 반영되었습니다."})

            yield evt({"status": "generating", "msg": "AI가 문항을 생성하고 있습니다. 수 분 소요됩니다..."})

            spec: ExamSpec = {
                "passage_text": text,
                "standards": _parse_standards(standards),
            }

            # graph.invoke가 실행될 스레드로 contextvars가 전파되도록,
            # to_thread 호출 전에 세션 dict를 먼저 만들어둔다.
            init_session()
            graph = get_exam_graph()
            state = await asyncio.to_thread(graph.invoke, {"spec": spec, "budget": 5})
            items = get_draft_items()

            yield evt({
                "status": "done",
                "items": items,
                "validation_passed": state.get("validation_passed", False),
                "truncated": truncated,
            })

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
    from app.modules.exam import ExamSpec, get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    truncated = len(passage_text) > MAX_PASSAGE_LENGTH
    text = passage_text[:MAX_PASSAGE_LENGTH] if truncated else passage_text

    spec: ExamSpec = {
        "passage_text": text,
        "standards": _parse_standards(standards),
    }

    init_session()
    graph = get_exam_graph()
    state = await asyncio.to_thread(graph.invoke, {"spec": spec, "budget": 5})
    items = get_draft_items()
    return {
        "items": items,
        "validation_passed": state.get("validation_passed", False),
        "truncated": truncated,
    }


# ── 생기부 윤문 ──────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    memo: str


@app.post("/record")
async def record(req: RecordRequest):
    from app.modules.record import get_record_chain
    chain = get_record_chain()
    result = await chain.run(req.memo)
    return result

