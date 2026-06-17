#!/usr/bin/env python
"""Phase 3 출제 모듈 통합 테스트.
샘플 PDF 인덱싱 → ReAct Agent 출제 → 제약 검증.
"""
import os
import shutil
import sys
import tempfile

import fitz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./chroma_db_exam_test")

from app.common.rag import BGEEmbedder, RAGStore, chunk_document, parse_pdf
from app.modules.exam import ExamSpec, get_exam_graph

SAMPLE_TEXT = """\
제1장 민주주의와 헌법

민주주의는 국민이 주권을 갖고 스스로 나라를 다스리는 정치 체제다.
대한민국 헌법 제1조는 "대한민국은 민주공화국이다"라고 명시한다.
국민 주권 원리는 모든 권력이 국민으로부터 나온다는 뜻이다.
기본권 보장은 개인의 자유와 평등을 국가가 보호함을 의미한다.
권력 분립은 입법·행정·사법으로 국가 권력을 나누어 견제와 균형을 이룬다.

제2장 시장 경제와 경제 원리

시장 경제는 수요와 공급에 따라 자원이 배분되는 경제 체제다.
가격은 생산자와 소비자의 결정을 조정하는 신호 역할을 한다.
시장 실패는 외부효과·공공재·독과점·정보 비대칭으로 발생한다.
정부는 시장 실패를 교정하기 위해 규제, 세금, 보조금 등을 사용한다.
경제 성장은 생산성 향상, 자본 투자, 기술 혁신에 의해 촉진된다.

제3장 사회 불평등과 복지

사회 계층은 소득, 직업, 교육 수준 등에 따라 형성된다.
기회의 평등은 모든 사람이 공정한 출발선에 설 수 있어야 함을 뜻한다.
복지 정책은 빈곤을 줄이고 취약 계층을 보호하기 위한 제도다.
교육은 사회 이동성을 높이고 세대 간 불평등을 줄이는 핵심 요인이다.
세계화는 경제적 상호 의존을 증가시키지만 소득 격차도 확대한다.
"""

COLLECTION = "exam_test_source"
CHROMA_DIR = "./chroma_db_exam_test"


def create_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT, fontsize=10)
    doc.save(path)
    doc.close()


def index_pdf(pdf_path: str) -> None:
    print("  PDF 파싱·청킹·임베딩 중...")
    store = RAGStore()
    embedder = BGEEmbedder()
    doc = parse_pdf(pdf_path)
    chunks = chunk_document(doc)
    vecs = embedder.embed([c["text"] for c in chunks])
    store.add_chunks(COLLECTION, chunks, vecs)
    print(f"  → {len(chunks)}개 청크 적재 완료")


def main() -> None:
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    spec: ExamSpec = {
        "unit": "민주주의와 헌법",
        "num_items": 2,
        "type_dist": {"객관식": 1, "서술형": 1},
        "difficulty_dist": {"중": 2},
        "target": "내신",
        "standards": ["민주주의 핵심 원리 이해"],
    }

    print("=== Phase 3 출제 모듈 통합 테스트 ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "2024_social_studies.pdf")
        print("1. 샘플 PDF 생성 및 인덱싱...")
        create_pdf(pdf_path)
        index_pdf(pdf_path)

    print("\n2. ReAct 에이전트 출제 시작...")
    print(f"   spec: {spec['num_items']}문항 | {spec['type_dist']} | {spec['difficulty_dist']}")

    graph = get_exam_graph()
    state = graph.invoke(
        {
            "spec": spec,
            "source_collection": COLLECTION,
            "budget": 2,
            "draft_items": [],
            "agent_messages": [],
            "coverage_map": {},
            "validation_passed": False,
            "error": "",
        }
    )

    print("\n3. 결과 확인")
    items = state.get("draft_items", [])
    approved = [it for it in items if it.get("status") == "approved"]

    print(f"   생성 문항: {len(items)}개 | 승인: {len(approved)}개")
    print(f"   검증 통과: {state.get('validation_passed', False)}")
    print(f"   coverage_map: {state.get('coverage_map')}")

    for i, it in enumerate(items, 1):
        print(
            f"\n  [{i}] {it.get('status', '?').upper()} | "
            f"{it.get('item_type','?')} | 난이도:{it.get('difficulty','?')} | "
            f"judge:{it.get('judge_score', 0)}/5 | dup:{it.get('is_duplicate', False)}"
        )
        print(f"       Q: {str(it.get('question',''))[:80]}")

    print("\n[완료] Phase 3 통합 테스트 종료")
    if not items:
        print("  경고: 문항이 생성되지 않았습니다. Ollama 연결 또는 모델 응답을 확인하세요.")

    shutil.rmtree(CHROMA_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
