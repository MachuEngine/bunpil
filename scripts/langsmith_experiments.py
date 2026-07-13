"""LangSmith Experiments 연동 공용 유틸리티.

eval_exam.py(문항 품질·구조 유사도 Judge 신뢰도)와 eval_ragas.py(RAG 품질)가 공유.
골든셋 JSON이 이미 사람 라벨의 단일 진실 공급원(source of truth)이므로, LangSmith
Dataset은 실행할 때마다 그 내용으로 동기화한다(기존 데이터셋이 있으면 지우고
재생성) — 라벨이 바뀌었는데(예: 이번 세션의 str_010/047 재라벨링) Dataset이
오래된 값을 계속 들고 있는 불일치를 막기 위함. 골든셋이 최대 수십 개 수준이라
매번 재생성해도 비용이 크지 않다.

LANGCHAIN_TRACING_V2가 꺼져 있으면(로컬에서 LangSmith 없이 개발 중인 경우)
아무 것도 하지 않고 조용히 건너뛴다 — Experiments 연동은 선택 기능이지 필수
경로가 아니다.
"""
import os


def experiments_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2") == "true"


def sync_dataset(client, name: str, examples: list[dict], description: str = "") -> None:
    """examples: [{"inputs": {...}, "outputs": {...}}, ...]"""
    if client.has_dataset(dataset_name=name):
        client.delete_dataset(dataset_name=name)
    dataset = client.create_dataset(dataset_name=name, description=description)
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e.get("outputs", {}) for e in examples],
        dataset_id=dataset.id,
    )


def identity_target(inputs: dict) -> dict:
    """골든셋 항목처럼 이미 고정된 콘텐츠를 평가할 때 쓰는 target — 새로 생성하지
    않고 그대로 통과시킨다. evaluator가 inputs를 직접 참조해도 되지만, LangSmith
    UI에서 "무엇을 평가했는지"가 outputs로도 보이도록 명시적으로 통과시켜 둔다."""
    return inputs
