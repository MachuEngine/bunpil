#!/usr/bin/env python
"""Phase 1 RAG 인프라 검증: 샘플 PDF 인덱싱 → 검색 → rerank."""
import os
import shutil
import sys
import tempfile

import fitz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.common.rag import (
    BGEEmbedder,
    BGEReranker,
    RAGRetriever,
    RAGStore,
    chunk_document,
    parse_pdf,
)

SAMPLE_TEXT = """\
Chapter 1: Democracy and Constitution

Democracy is a political system in which citizens hold sovereignty and govern themselves.
The constitution serves as the supreme law of the land, guaranteeing fundamental rights.
Citizens participate in governance through elections, referendums, and civic engagement.
The separation of powers divides government authority into legislative, executive, and judicial.
Rule of law ensures that all people and institutions are accountable to the law.

Chapter 2: Market Economy and Economic Principles

A market economy relies on supply and demand to allocate resources efficiently.
Prices serve as signals coordinating the decisions of buyers and sellers in markets.
Competition among producers leads to lower prices and improved quality for consumers.
Government intervention may address market failures such as externalities and public goods.
Economic growth depends on productivity, capital investment, and technological innovation.

Chapter 3: Social Inequality and Welfare

Social stratification refers to the hierarchical arrangement of individuals in society.
Equality of opportunity ensures fair access to resources and advancement for all citizens.
Welfare policies reduce poverty and provide a safety net for vulnerable populations.
Education is a key driver of social mobility and reducing intergenerational inequality.
Globalization has increased economic interdependence while also widening income gaps.
"""

COLLECTION = "permanent"
TEST_CHROMA_DIR = "./chroma_db_test"


def create_sample_pdf(path: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT, fontsize=10)
    doc.save(path)
    doc.close()


def main():
    os.environ.setdefault("CHROMA_PERSIST_DIR", TEST_CHROMA_DIR)
    shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "2024_social_studies_sample.pdf")

        print("1. 샘플 PDF 생성...")
        create_sample_pdf(pdf_path)

        print("2. PDF 파싱 및 청킹...")
        doc = parse_pdf(pdf_path)
        chunks = chunk_document(doc)
        print(f"   → {len(chunks)}개 청크 | source={doc['source']} year={doc['year']}")

        print("3. BGE-M3 임베더 로드... (첫 실행 시 모델 다운로드)")
        embedder = BGEEmbedder()

        print("4. BGE-reranker 로드...")
        reranker = BGEReranker()

        print("5. ChromaDB 영구 컬렉션 적재...")
        store = RAGStore()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        store.add_chunks(COLLECTION, chunks, embeddings)
        print(f"   → '{COLLECTION}' 컬렉션에 {len(chunks)}개 저장")

        print("\n6. 세션 임시 컬렉션 생성 및 폐기 (업로드 지문 시뮬레이션)...")
        temp_name = store.create_temp_collection()
        store.add_chunks(temp_name, chunks[:2], embedder.embed(texts[:2]))
        store.delete_collection(temp_name)
        print(f"   → 임시 컬렉션 '{temp_name[:20]}...' 폐기 완료")

        print("\n7. 검색 + rerank 테스트...")
        retriever = RAGRetriever(store, embedder, reranker)
        query = "What is the role of government in a market economy?"
        results = retriever.retrieve(query, COLLECTION, top_k=3)

        print(f"\n[쿼리] {query}")
        print(f"[결과] {len(results)}개 청크 반환\n")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"  #{i} score={r['score']:.4f} | source={meta['source']} page={meta['page']}")
            print(f"      {r['text'][:100].strip()}...")

    shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)
    print("\n[완료] Phase 1 RAG 인프라 검증 통과")


if __name__ == "__main__":
    main()
