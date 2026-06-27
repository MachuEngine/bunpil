import asyncio
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="분필 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


class RecordRequest(BaseModel):
    memo: str


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
    """PDF 지문과 파라미터를 받아 문항을 생성한다."""
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

        graph = get_exam_graph()
        state = await asyncio.to_thread(
            graph.invoke,
            {"spec": spec, "source_collection": col, "budget": 3},
        )
        items = get_draft_items()
        return {
            "items": items,
            "validation_passed": state.get("validation_passed", False),
        }

    finally:
        if col:
            try:
                store.delete_collection(col)
            except Exception:
                pass


@app.post("/record")
async def record(req: RecordRequest):
    """관찰 메모를 받아 윤문 결과를 반환한다."""
    from app.modules.record import get_record_chain
    chain = get_record_chain()
    result = await asyncio.to_thread(chain.run, req.memo)
    return result
