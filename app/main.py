import asyncio
import json
import logging
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="분필 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 문항 출제: SSE 스트리밍 ──────────────────────────────────────────────

@app.post("/exam/stream")
async def exam_stream(
    pdf: UploadFile = File(...),
    unit: str = Form(...),
    num_mc: int = Form(1),
    num_sa: int = Form(0),
    num_hard: int = Form(0),
    num_med: int = Form(1),
    num_easy: int = Form(0),
    standards: str = Form(""),
):
    """PDF + 파라미터를 받아 SSE로 진행 상황과 결과를 스트리밍한다."""

    # UploadFile은 StreamingResponse 반환 즉시 닫히므로 제너레이터 밖에서 미리 읽는다
    pdf_bytes = await pdf.read()

    async def generate():
        from app.common.rag import BGEEmbedder, RAGStore, chunk_document, parse_pdf
        from app.modules.exam import ExamSpec, get_exam_graph
        from app.modules.exam.tools import get_draft_items, init_session

        def evt(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        store = RAGStore()
        embedder = BGEEmbedder()
        col: str | None = None

        try:
            yield evt({"status": "parsing", "msg": "PDF를 분석하고 있습니다..."})

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            try:
                doc = parse_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)

            chunks = chunk_document(doc)
            if not chunks:
                yield evt({"status": "error", "msg": "PDF에서 텍스트를 추출할 수 없습니다."})
                return

            yield evt({"status": "indexing", "msg": "텍스트를 인덱싱하고 있습니다..."})

            col = store.create_temp_collection()
            embeddings = embedder.embed([c["text"] for c in chunks])
            store.add_chunks(col, chunks, embeddings)

            yield evt({"status": "generating", "msg": "AI가 문항을 생성하고 있습니다. 수 분 소요됩니다..."})

            std_list = [s.strip() for s in standards.splitlines() if s.strip()]
            if not std_list and standards:
                std_list = [s.strip() for s in standards.split(",") if s.strip()]

            spec: ExamSpec = {
                "unit": unit,
                "num_items": num_mc + num_sa,
                "type_dist": {"객관식": num_mc, "서술형": num_sa},
                "difficulty_dist": {"상": num_hard, "중": num_med, "하": num_easy},
                "target": "고2 사회문화",
                "standards": std_list,
            }

            init_session(col)
            graph = get_exam_graph()
            state = await asyncio.to_thread(
                graph.invoke,
                {"spec": spec, "source_collection": col, "budget": 3},
            )
            items = get_draft_items()
            # graph 수정으로 pair당 1회 생성이 보장되므로 별도 trim 불필요
            # 빈 선지 항목은 프론트에서 표시 — 조용히 제거하면 결과 0개가 될 수 있음

            yield evt({
                "status": "done",
                "items": items,
                "validation_passed": state.get("validation_passed", False),
            })

        except Exception as e:
            logger.exception("/exam/stream 오류")
            import traceback
            yield evt({"status": "error", "msg": str(e), "detail": traceback.format_exc()})

        finally:
            if col:
                try:
                    store.delete_collection(col)
                except Exception:
                    pass

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
    pdf: UploadFile = File(...),
    unit: str = Form(...),
    num_mc: int = Form(5),
    num_sa: int = Form(2),
    num_hard: int = Form(2),
    num_med: int = Form(3),
    num_easy: int = Form(2),
    standards: str = Form(""),
):
    from app.common.rag import BGEEmbedder, RAGStore, chunk_document, parse_pdf
    from app.modules.exam import ExamSpec, get_exam_graph
    from app.modules.exam.tools import get_draft_items, init_session

    store = RAGStore()
    embedder = BGEEmbedder()
    col: str | None = None

    try:
        pdf_bytes = await pdf.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            doc = parse_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)

        chunks = chunk_document(doc)
        if not chunks:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        col = store.create_temp_collection()
        embeddings = embedder.embed([c["text"] for c in chunks])
        store.add_chunks(col, chunks, embeddings)

        std_list = [s.strip() for s in standards.splitlines() if s.strip()]
        spec: ExamSpec = {
            "unit": unit,
            "num_items": num_mc + num_sa,
            "type_dist": {"객관식": num_mc, "서술형": num_sa},
            "difficulty_dist": {"상": num_hard, "중": num_med, "하": num_easy},
            "target": "고2 사회문화",
            "standards": std_list,
        }

        init_session(col)
        graph = get_exam_graph()
        state = await asyncio.to_thread(
            graph.invoke,
            {"spec": spec, "source_collection": col, "budget": 3},
        )
        items = get_draft_items()
        return {"items": items, "validation_passed": state.get("validation_passed", False)}

    finally:
        if col:
            try:
                store.delete_collection(col)
            except Exception:
                pass


# ── 생기부 윤문 ──────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    memo: str


@app.post("/record")
async def record(req: RecordRequest):
    from app.modules.record import get_record_chain
    chain = get_record_chain()
    result = await chain.run(req.memo)
    return result

