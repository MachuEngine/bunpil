"""생성 경로 RAG 제거(2026-10-07) 회귀 테스트 — graph.py/tools.py를 import하는 것만으로
FlagEmbedding(BGE-M3 임베더, CPU에서도 로딩이 느림)을 불러오지 않는지 확인한다.

같은 pytest 프로세스 안에서는 다른 테스트 모듈(test_rag_store.py 등)이 먼저
app.common.rag를 import해 FlagEmbedding이 이미 sys.modules에 있을 수 있어(테스트
실행 순서에 따라 결과가 달라짐), 별도 서브프로세스로 격리해 확인한다."""
import subprocess
import sys


def test_importing_exam_graph_does_not_load_flagembedding():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import app.modules.exam.graph; "
         "assert 'FlagEmbedding' not in sys.modules, 'FlagEmbedding이 즉시 로딩됨'"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
