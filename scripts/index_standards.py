"""data/standards/ 디렉토리의 텍스트·PDF 파일을 standards 영구 컬렉션에 적재.
이미 적재된 파일(source 기준)은 스킵 — idempotent.
지원 형식: .txt, .pdf
실행: python scripts/index_standards.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.common.rag import BGEEmbedder, RAGStore, chunk_document, extract_year, parse_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION = "standards"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "standards")


def _parse_txt(path: str) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0]
    year = extract_year(os.path.basename(path))
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return {"source": stem, "year": year, "pages": [{"page": 1, "text": text}]}


def main() -> None:
    if not os.path.isdir(DATA_DIR):
        logger.error("디렉토리가 없습니다: %s", DATA_DIR)
        sys.exit(1)

    all_files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".pdf", ".txt"))
    ]
    if not all_files:
        logger.warning("data/standards/ 에 .txt/.pdf 파일이 없습니다.")
        return

    store = RAGStore()
    embedder = BGEEmbedder()
    already = store.indexed_sources(COLLECTION)

    indexed = 0
    for filename in sorted(all_files):
        path = os.path.join(DATA_DIR, filename)
        stem = os.path.splitext(filename)[0]
        if stem in already:
            logger.info("스킵 (이미 적재됨): %s", filename)
            continue

        logger.info("파싱 중: %s", filename)
        if filename.lower().endswith(".pdf"):
            doc = parse_pdf(path)
        else:
            doc = _parse_txt(path)

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
    logger.info("완료 — 이번 실행 적재: %d 파일 / standards 총 문서 수: %d", indexed, total)


if __name__ == "__main__":
    main()
