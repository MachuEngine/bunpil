import os

import chromadb


class RAGStore:
    def __init__(self):
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        col = self._collection(collection_name)
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
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=metas,
            ids=ids,
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

    def count(self, collection_name: str) -> int:
        """컬렉션의 문서 수를 반환한다. 컬렉션이 없거나 오류 시 0."""
        try:
            return self._collection(collection_name).count()
        except Exception:
            return 0

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 20,
    ) -> list[dict]:
        col = self._collection(collection_name)
        res = col.query(query_embeddings=[query_embedding], n_results=n_results)
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
