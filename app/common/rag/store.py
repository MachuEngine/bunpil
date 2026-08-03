import os

import chromadb
from chromadb.config import Settings


class RAGStore:
    def __init__(self):
        # 크로마DB 경로 설정 및 
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db") # .env에 CHROMA_PERSIST_DIR가 없으면 fallback 경로로 설정됨
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl=(
                    "app.common.rag.telemetry.NoOpProductTelemetry"
                ),
            ),
        )

    def _collection(self, name: str):
        # get_or_create_collection(): 있으면 가져오고 없으면 새로 만듦
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}, # 유사도 계산 방식을 코사인 유사도로 지정
        )

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """
            add_chunks()는 인덱싱 함수이다. 
        """
        col = self._collection(collection_name)
        # ids = [ "규정.pdf_p3_0", ..., "규정.pdf_p3_5", ... ] -> list[str]
        #   c = {
        #       "source" = "규정.pdf",
        #       "year" = "2026",
        #       "page" = "p4"
        #       ...
        #   }
        ids = [f"{c['source']}_p{c['page']}_{i}" for i, c in enumerate(chunks)]
        metas = [
            {
                "source": c["source"],
                "year": c["year"] if c["year"] is not None else -1,
                "page": c["page"],
                # 청크가 여러 페이지에 걸칠 수 있어(2026-07-14 청킹 재설계) 끝 페이지도
                # 기록. 단일 페이지면 page와 동일. 구버전 청크엔 없으므로 .get 폴백.
                "page_end": c.get("page_end", c["page"]),
            }
            for c in chunks
        ]
        col.upsert(
            documents=[c["text"] for c in chunks],   # ["본문1", "본문2", "본문3", ...]
            embeddings=embeddings,                   # [[0.1, 0.2,...], [0.3, 0.1,...], ...]
            metadatas=metas,                         # [{"source":...}, {"source":...}, ...]
            ids=ids,                                 # ["규정.pdf_p3_0", "규정.pdf_p3_1", ...]
        )

    def indexed_sources(self, collection_name: str) -> set[str]:
        """컬렉션에 이미 적재된 source 목록을 반환한다."""
        try:
            col = self._collection(collection_name)
            if col.count() == 0:
                return set()
            result = col.get(include=["metadatas"])
            return {m.get("source", "") for m in result["metadatas"]}
        except Exception:
            return set()

    def all_documents(self, collection_name: str) -> list[dict]:
        """컬렉션의 모든 청크를 id·본문·메타데이터로 반환한다.

        BM25 역색인(`lexical.py`)을 세울 때 쓴다 — dense 검색은 상위 후보만
        받아오면 되지만, 어휘 검색은 "정답 청크가 dense 후보에 아예 안 들어오는"
        경우를 잡는 게 목적이라 컬렉션 전체를 봐야 한다. 청크가 수백 개
        수준(regulations 472 / standards 557)이라 전량 조회해도 부담이 없다.
        """
        try:
            col = self._collection(collection_name)
            if col.count() == 0:
                return []
            res = col.get(include=["documents", "metadatas"])
            return [
                {"id": doc_id, "text": text, "metadata": meta}
                for doc_id, text, meta in zip(
                    res["ids"], res["documents"], res["metadatas"]
                )
            ]
        except Exception:
            return []

    def count(self, collection_name: str) -> int:
        """컬렉션의 문서 수를 반환한다. 컬렉션이 없거나 오류 시 0.
            문서 = 청크 
            즉, 청크 개수를 반환 함.
        """
        try:
            return self._collection(collection_name).count() # ChromaDB 라이브러리가 제공하는 Collection 객체의 .count() 메서드
        except Exception:
            return 0

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 20,
    ) -> list[dict]:
        """
        ChromaDB는 API 설계 자체가 쿼리를 여러개 Batch로 한꺼번에 검색할 수 있도록 지원함.
        그렇기 때문에 쿼리가 1개여도 리스트의 리스트 형태로 처리해야함. 
        대신 쿼리가 1개이면 바깥 리스트는 ["xxxx"][0]으로 고정해서 처리하면 됨. 
        ---
        zip은 여러 개의 리스트를 같은 인덱스끼리 짝지어서 튜플로 묶어주는 파이썬 내장 함수 
        때문에 zip으로 처리하면 아래와 같이 튜플로 묶어서 리턴할 수 있음.

        return [
            ("본문1", {"source": "a.pdf"}, 0.12),
            ("본문2", {"source": "b.pdf"}, 0.45),
            ("본문3", {"source": "c.pdf"}, 0.67),
        ]
        """
        col = self._collection(collection_name)
        res = col.query(query_embeddings=[query_embedding], n_results=n_results)
        # id는 하이브리드 검색에서 dense 순위와 BM25 순위를 같은 문서끼리 짝지을 때
        # 쓴다(2026-08-03 추가). 본문 문자열로 짝지으면 동일 텍스트 청크가 뭉개지므로
        # ChromaDB가 이미 갖고 있는 id를 그대로 조인 키로 쓴다.
        return [
            {"id": doc_id, "text": doc, "metadata": meta, "distance": dist}
            for doc_id, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
