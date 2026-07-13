import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 문장·문단 경계를 우선 존중하며 분할하기 위한 구분자 우선순위.
# 앞쪽(문단·줄바꿈)에서 먼저 자르고, 안 되면 뒤쪽(문장부호·공백)으로 내려간다.
# 한국어/영어 문장부호(. ! ?)와 전각 마침표(。)를 함께 고려.
_SEPARATORS = ["\n\n", "\n", ". ", ".\n", "。", "! ", "? ", " ", ""]


def parse_pdf(path: str) -> dict:
    """PDF를 파싱해 텍스트와 메타데이터를 반환한다."""
    p = Path(path)
    source = p.stem
    year = extract_year(p.name)

    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()

    return {"source": source, "year": year, "pages": pages}


def chunk_document(doc: dict, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """문서를 문장/문단 경계를 존중하는 청크로 분할한다.

    2026-07-14 재설계 (TROUBLESHOOTING.md 참고): 기존에는 페이지마다 청킹을
    리셋하고 글자 수 기준으로 고정 분할했는데, 이로 인해 (1) 페이지 끝에서
    800자를 못 채운 자투리 청크(심하면 1글자짜리 노이즈)가 페이지마다 발생하고,
    (2) 문장 한가운데서 잘려 임베딩 품질이 떨어졌다. 이를 해결하기 위해 페이지를
    하나로 이어붙인 뒤 RecursiveCharacterTextSplitter로 문장·문단 경계 우선 분할한다.

    페이지 메타데이터: 각 문자가 어느 페이지에서 왔는지 추적해, 청크가 시작하는
    페이지를 `page`(하위호환 — store.py의 청크 ID 생성이 이 값을 씀), 끝나는
    페이지를 `page_end`로 기록한다. 단일 페이지 청크는 둘이 같다.
    """
    # 페이지 텍스트를 이어붙이되, 각 문자 위치의 페이지 번호를 병렬로 기록한다.
    parts: list[str] = []
    char_page: list[int] = []
    for page in doc["pages"]:
        text = page["text"]
        parts.append(text)
        char_page.extend([page["page"]] * len(text))
        parts.append("\n\n")  # 페이지 사이 문단 구분자 (splitter가 여기서 우선 자르도록)
        char_page.extend([page["page"]] * 2)
    full_text = "".join(parts)
    if not full_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_SEPARATORS,
        length_function=len,
    )
    pieces = splitter.split_text(full_text)

    fallback_page = doc["pages"][0]["page"] if doc["pages"] else 1
    chunks: list[dict] = []
    search_start = 0
    for piece in pieces:
        # 청크가 원문 어디에서 왔는지 찾아 페이지 범위를 산출한다(overlap이 있어
        # 조각이 앞으로 겹칠 수 있으므로, 못 찾으면 처음부터 다시 탐색).
        idx = full_text.find(piece, search_start)
        if idx < 0:
            idx = full_text.find(piece)
        if idx >= 0:
            span = char_page[idx : idx + len(piece)]
            page_start = span[0] if span else fallback_page
            page_end = span[-1] if span else page_start
            search_start = idx + 1
        else:
            page_start = page_end = fallback_page
        chunks.append(
            {
                "text": piece,
                "source": doc["source"],
                "year": doc["year"],
                "page": page_start,
                "page_end": page_end,
            }
        )
    return chunks


def extract_year(filename: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group()) if match else None
