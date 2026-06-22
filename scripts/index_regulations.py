"""data/regulations/ 디렉토리의 PDF를 regulations 영구 컬렉션에 적재.
이미 적재된 파일(source 기준)은 스킵 — idempotent.
실행: python scripts/index_regulations.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.common.rag import BGEEmbedder, RAGStore, chunk_document, parse_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "regulations"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "regulations")


def main() -> None:
    if not os.path.isdir(DATA_DIR):
        logger.error("디렉토리가 없습니다: %s", DATA_DIR)
        sys.exit(1)

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("data/regulations/ 에 PDF 파일이 없습니다.")
        return

    store = RAGStore()
    embedder = BGEEmbedder()
    already = store.indexed_sources(COLLECTION)

    indexed = 0
    for filename in sorted(pdf_files):
        path = os.path.join(DATA_DIR, filename)
        stem = os.path.splitext(filename)[0]
        if stem in already:
            logger.info("스킵 (이미 적재됨): %s", filename)
            continue

        logger.info("파싱 중: %s", filename)
        doc = parse_pdf(path)
        chunks = chunk_document(doc)
        if not chunks:
            logger.warning("청크 없음 — 스킵: %s", filename)
            continue

        logger.info("임베딩 중 (%d 청크)...", len(chunks))
        embeddings = embedder.embed([c["text"] for c in chunks])
        store.add_chunks(COLLECTION, chunks, embeddings)
        logger.info("적재 완료: %s (%d 청크)", filename, len(chunks))
        indexed += 1

    total = store.count(COLLECTION)
    logger.info("완료 — 이번 실행 적재: %d 파일 / regulations 총 문서 수: %d", indexed, total)


if __name__ == "__main__":
    main()
