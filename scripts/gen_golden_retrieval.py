#!/usr/bin/env python
"""실제 ChromaDB 컬렉션에서 검색 평가 골든셋 초안을 생성한다.

출력: data/golden/retrieval_golden.json
reviewed: false — LLM 생성 초안이므로 사람이 검수 후 true로 변경해야 함.
실행: python scripts/gen_golden_retrieval.py
"""
import asyncio
import json
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.common.llm import get_llm_backend
from app.common.rag import RAGStore

SAMPLE_COUNTS = {
    "standards": 12,
    "regulations": 10,
    "past_exams": 8,
}
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden", "retrieval_golden.json")

QUERY_PROMPT = (
    "다음은 한국 고등학교 사회 교과 관련 텍스트 청크입니다.\n"
    "이 청크를 찾기 위해 교사가 입력할 법한 자연스러운 검색 쿼리를 한 문장으로만 출력하세요.\n"
    "쿼리만 출력하고 다른 설명은 쓰지 마세요.\n\n"
    "청크:\n{chunk}"
)


def sample_chunks(store: RAGStore, collection_name: str, n: int) -> list[dict]:
    col = store._collection(collection_name)
    total = col.count()
    if total == 0:
        warnings.warn(f"{collection_name} 컬렉션이 비어있습니다 — 스킵")
        return []
    if total < n:
        warnings.warn(f"{collection_name}: 요청 {n}개 < 실제 {total}개 — {total}개만 생성")
        n = total

    all_ids = col.get(include=[])["ids"]
    sampled_ids = random.sample(all_ids, n)
    result = col.get(ids=sampled_ids, include=["documents", "metadatas"])

    return [
        {
            "id": chunk_id,
            "text": doc,
            "metadata": meta,
        }
        for chunk_id, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]


async def generate_query(backend, chunk_text: str) -> str:
    prompt = QUERY_PROMPT.format(chunk=chunk_text[:500])
    response = await backend.generate([{"role": "user", "content": prompt}])
    return response.strip()


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    store = RAGStore()
    backend = get_llm_backend()

    records = []
    summary = {}
    item_counter = 1

    for collection_name, n in SAMPLE_COUNTS.items():
        print(f"\n[{collection_name}] 샘플링 중 ({n}개 요청)...")
        chunks = sample_chunks(store, collection_name, n)

        count = 0
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_id = chunk["id"]

            print(f"  쿼리 생성 중 ({count + 1}/{len(chunks)}): {chunk_id[:30]}...")
            try:
                query = asyncio.run(generate_query(backend, chunk_text))
            except Exception as e:
                warnings.warn(f"LLM 호출 실패 ({chunk_id}): {e}")
                query = ""

            records.append({
                "id": f"ret_{item_counter:03d}",
                "query": query,
                "source_collection": collection_name,
                "expected_chunk_id": chunk_id,
                "chunk_preview": chunk_text[:100],
                "reviewed": False,
            })
            item_counter += 1
            count += 1

        summary[collection_name] = count

    # 자체 검증: query 빈값 확인
    empty_queries = [r["id"] for r in records if not r["query"]]
    if empty_queries:
        warnings.warn(f"query가 비어있는 항목: {empty_queries}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    total = sum(summary.values())
    print(
        f"\n완료 — "
        + ", ".join(f"{k}: {v}개" for k, v in summary.items())
        + f", 총 {total}개 생성"
    )
    print(f"출력: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
