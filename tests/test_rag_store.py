"""RAG 저장소가 외부 익명 telemetry 없이 초기화되는지 검증한다."""
from app.common.rag.store import RAGStore


def test_chroma_anonymized_telemetry_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    store = RAGStore()
    assert store.client.get_settings().anonymized_telemetry is False
    assert store.client.get_settings().chroma_product_telemetry_impl.endswith(
        ".NoOpProductTelemetry"
    )
