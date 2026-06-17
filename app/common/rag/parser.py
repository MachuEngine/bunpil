import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def parse_pdf(path: str) -> dict:
    """PDF를 파싱해 텍스트와 메타데이터를 반환한다."""
    p = Path(path)
    source = p.stem
    year = _extract_year(p.name)

    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()

    return {"source": source, "year": year, "pages": pages}


def chunk_document(doc: dict, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """페이지별 텍스트를 고정 길이 청크로 분할한다."""
    chunks = []
    for page in doc["pages"]:
        text = page["text"]
        start = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": doc["source"],
                        "year": doc["year"],
                        "page": page["page"],
                    }
                )
            start += chunk_size - overlap
    return chunks


def _extract_year(filename: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group()) if match else None
